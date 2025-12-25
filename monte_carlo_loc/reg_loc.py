# Registration localization module
# Created by Ericxhzou

import numpy as np
import time
from queue import Queue
from scipy.spatial.transform import Rotation as R

import teaserpp_python

from monte_carlo_loc.loc_utils import euler_angles_from_rotation_matrix
from utils.draw_result import draw_cov_ellipse


class RegLoc:
    def __init__(self, logger, config, poses):
        self.logger = logger
        self.use_certify_algo = config['use_certify_algo']  # use teaser++ certification algo or not
        self.min_reliable_value = float(config['min_reliable_value'])
        self.max_sigma_x = config['max_sigma_x'] * 3
        self.max_sigma_y = config['max_sigma_y'] * 3
        self.max_sigma_yaw = config['max_sigma_yaw'] * 3
        # solver
        self.solver_params = teaserpp_python.RobustRegistrationSolver.Params()
        self.solver_params.cbar2 = config['cbar2']
        self.solver_params.noise_bound = config['noise_bound']
        self.solver_params.estimate_scaling = config['estimate_scaling']
        self.solver_params.rotation_estimation_algorithm = teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
        self.solver_params.rotation_gnc_factor = config['rotation_gnc_factor']
        self.solver_params.rotation_max_iterations = config['rotation_max_iterations']
        self.solver_params.rotation_cost_threshold = float(config['rotation_cost_threshold'])
        self.solver_params.rotation_only_yaw = float(config['rotation_only_yaw'])
        self.logger.info("Solver Parameters are:", self.solver_params)
        self.solver = teaserpp_python.RobustRegistrationSolver(self.solver_params)
        # certifier
        self.certifier_params = teaserpp_python.DRSCertifier.Params()
        self.certifier_params.cbar2 = config['cbar2']
        self.certifier_params.noise_bound = config['noise_bound']
        self.certifier_params.sub_optimality = float(config['sub_optimality'])
        self.logger.info("Certifier Parameters are:", self.certifier_params)
        self.certifier = teaserpp_python.DRSCertifier(self.certifier_params)
        # certify request / results
        self.certify_requests = Queue(maxsize=1)
        self.certify_results = Queue(maxsize=1)
    
    def reg_with_teaser(self, q_kpts, m_kpts):
        ''' m = R * q + t
            @Input: q_kpts: N x 3, m_kpts: N x 3
            @Return: est_R, est_t, est_theta
        '''
        self.solver.reset(self.solver_params)
        self.solver.solve(q_kpts.T, m_kpts.T)  # src: 3 x N, dst: 3 x N

        solution = self.solver.getSolution()

        # Print the inliers
        mac_inliers = self.solver.getInlierMaxClique()
        rotation_inlier_mask = self.solver.getRotationInliersMask()
        translation_inlier_mask = self.solver.getTranslationInliersMask()
        mask = rotation_inlier_mask & translation_inlier_mask
        inliers = np.array(mac_inliers)[mask]
        est_theta = np.ones(len(q_kpts), dtype=int) * -1
        est_theta[inliers] = 1  # inliers
        
        return solution.rotation, solution.translation, est_theta
    
    @staticmethod
    def get_cov(q_kpts, est_R, est_theta):
        """ @Input:  q_kpts: N x 3, source,
                     est_R: 3 x 3, est_theta: N ({1, -1}, 1: inlier, -1: outlier),
            @Description:
                4-DOF registration formulation
                / Xm \   / cosθ -sinθ  0  tx \   / Xp \
                | Ym | = | sinθ  cosθ  0  ty | x | Yp |
                | Zm |   |   0     0   1  tz |   | Zp |
                \  1 /   \   0     0   0  1  /   \ 1  /
                jacobian matrix: variable is [tx, ty, tz, θ]
                / 1  0  0 -sinθ*Xp+cosθ*Yp \
                | 0  1  0  cosθ*Xp-sinθ*Yp |
                \ 0  0  1         0        /
                Assume [tx, ty, tz, θ] ~ N(0, I)
                so, covariance matrix is: Cov = (J.T * J).inv()
            @Return: cov: 4 x 4, covariance matrix, reliable_value: float, larger is better
        """
        # covariance
        est_yaw = euler_angles_from_rotation_matrix(est_R)[2]  # in rad
        num_valid = np.sum(est_theta == 1)
        q_kpts = q_kpts[est_theta == 1]
        jacobian = np.zeros((num_valid * 3, 4))
        for i in range(num_valid):
            x, y = q_kpts[i, 0], q_kpts[i, 1]
            jacobian[3*i] = np.array([1.0, 0.0, 0.0, -np.sin(est_yaw) * x + np.cos(est_yaw) * y])
            jacobian[3*i+1] = np.array([0.0, 1.0, 0.0, np.cos(est_yaw) * x - np.sin(est_yaw) * y])
            jacobian[3*i+2] = np.array([0.0, 00, 1.0, 0.0])
        hessian = np.matmul(jacobian.T, jacobian)
        cov = np.linalg.inv(hessian)
        # reliable value
        eigen_values, eigen_vectors = np.linalg.eig(hessian)
        reliable_value = np.min(eigen_values)
        return cov, reliable_value
    
    def certify(self, q_kpts, m_kpts, est_R, est_theta):
        ''' @Input: q_kpts: N x 3, m_kpts: N x 3,
                    theta: N ({1, -1}, 1: inlier, -1: outlier),
                    est_R: 3 x 3
            @Return: is_optimal: bool, True if best_suboptimality < sub_optimality, else False
                     best_suboptimality: double
        '''
        certify_res = self.certifier.certify(est_R, q_kpts.T, m_kpts.T, est_theta.reshape(1, -1))
        return certify_res.is_optimal, certify_res.best_suboptimality
    
    def run_certify_once(self, request):
        t0 = time.time()
        certify_value = None
        if self.use_certify_algo:  # use teaser++ certification algorithm
            is_optimal, best_suboptimality = self.certify(request['q_kpts'], request['m_kpts'],
                                                        request['est_R'], request['est_theta'])
            certify_value = best_suboptimality
        else:  # use covariance and reliable value
            cov = request['cov']
            sigma_x, sigma_y, sigma_yaw = np.sqrt(cov[0,0]), np.sqrt(cov[1,1]), np.sqrt(cov[2,2])*180/np.pi
            is_optimal = request['reliable_value'] > self.min_reliable_value and \
                            sigma_x < self.max_sigma_x and sigma_y < self.max_sigma_y and sigma_yaw < self.max_sigma_yaw
            certify_value = request['reliable_value']
        result = {
            'frame_idx': request['frame_idx'],
            'pass_certify': is_optimal
        }
        t1 = time.time()
        time_cost = (t1 - t0) * 1000  # unit: ms
        frame_idx = result['frame_idx']
        self.logger.info(f'Certify frame: {frame_idx}, is_optimal: {is_optimal}, certify value: {certify_value:<.6f}, time cost: {time_cost:<.4f}ms')
        return result
    
    def run_certify(self):
        while True:
            if not self.certify_requests.empty():
                request = self.certify_requests.get()
                if request['type'] != 'certify':
                    break
                result = self.run_certify_once(request)
                self.certify_results.put(result)
            time.sleep(0.001)
            

if __name__ == '__main__':
    # params
    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver_params.cbar2 = 1
    solver_params.noise_bound = 0.05
    solver_params.estimate_scaling = False
    solver_params.rotation_estimation_algorithm = teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    solver_params.rotation_gnc_factor = 1.4
    solver_params.rotation_max_iterations = 100
    solver_params.rotation_cost_threshold = 1e-6
    solver_params.rotation_only_yaw = True
    solver = teaserpp_python.RobustRegistrationSolver(solver_params)
    # solve rotation
    t0 = time.time()
    src = np.random.rand(200, 3)
    expected_R = R.from_euler('zyx', [[45, 5, 5]], degrees=True).as_matrix()[0]
    dst = (expected_R @ src.T).T
    dst[10, 1] += 0.1
    solution = solver.solve(src.T, dst.T)
    delta_R = expected_R - solution.rotation
    t1 = time.time()
    print(f'solving rotation cost {1000*(t1-t0)} ms')
    # inliers
    mac_inliers = solver.getInlierMaxClique()
    rotation_inlier_mask = solver.getRotationInliersMask()
    translation_inlier_mask = solver.getTranslationInliersMask()
    mask = rotation_inlier_mask & translation_inlier_mask
    inliers = np.array(mac_inliers)[mask]
    est_theta = np.ones(len(src), dtype=int) * -1
    est_theta[inliers] = 1  # inliers
    # compute covariance and reliable value
    t0 = time.time()
    cov, reliable_value = RegLoc.get_cov(src, solution.rotation, est_theta)
    t1 = time.time()
    print('cov: \n', cov, '\nreliable value: ', reliable_value, '\ntime cost: ', 1000*(t1-t0), 'ms')
    # draw cov ellipse
    draw_cov_ellipse(mean_x=0.0, mean_y=0.0, cov=cov)