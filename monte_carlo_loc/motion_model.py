#!/usr/bin/env python3
# Developed by Xieyuanli Chen and Thomas Läbe
# This file is covered by the LICENSE file in the root of this project.
# Brief: this is the motion model for overlap-based Monte Carlo localization.

from loc_utils import *


class MotionModel:
    def __init__(self, config, poses):
        self.noise_dist = config['noise_dist']
        self.noise_rot1 = config['noise_rot1']
        self.noise_rot2 = config['noise_rot2']
        self.commands = None
        self.gt_headings = None
        self.commands, self.gt_headings = self.gen_commands(poses)
    
    def move(self, particles, frame_idx):
        """ MOTION performs the sampling from the proposal.
        distribution, here the rotation-translation-rotation motion model

        input:
        particles ... the particles as in the main script
        u ... the command in the form [trasl rot2] or real odometry [v, w]
        noise ... the variances for producing the Gaussian noise for
        perturbating the motion,  noise = [noiseR1 noiseTrasl noiseR2]

        output:
        the same particles, with updated poses.

        The position of the i-th particle is given by the 3D vector
        particles(i).pose which represents (x, y, z, theta, weight).

        Assume Gaussian noise in each of the three parameters of the motion model.
        These three parameters may be used as standard deviations for sampling.
        """
        u = self.commands[frame_idx]
        num_particles = len(particles)
        
        # noise in the [rot1 trasl rot2] commands when moving the particles
        r1Noise = self.noise_rot1
        transNoise = self.noise_dist
        r2Noise = self.noise_rot2

        rot1 = u[0] + r1Noise * np.random.randn(num_particles)
        tras1 = u[1] + transNoise * np.random.randn(num_particles)
        rot2 = u[2] + r2Noise * np.random.randn(num_particles)
        dz = u[3] # + transNoise * np.random.randn(num_particles)

        # update pose using motion model
        particles[:, 0] += tras1 * np.cos(particles[:, 3] + rot1)
        particles[:, 1] += tras1 * np.sin(particles[:, 3] + rot1)
        particles[:, 2] += dz
        particles[:, 3] += rot1 + rot2

        return particles

    @staticmethod
    def gen_commands(poses):
        """ Create commands out of the ground truth with noise.
        input:
            ground truth poses
        output:
            commands for each frame.
        """
        # compute noisy-free commands
        # set the default command = [0,0]'
        commands_ = np.zeros((len(poses), 4))
        
        # headings
        headings = []
        for idx in range(len(poses)):
            headings.append(euler_angles_from_rotation_matrix(poses[idx][:3, :3])[2])

        # move direction
        dx = (poses[1:, 0, 3] - poses[:-1, 0, 3])
        dy = (poses[1:, 1, 3] - poses[:-1, 1, 3])
        dz = poses[1:, 2, 3] - poses[:-1, 2, 3]
        direct = np.arctan2(dy, dx)  # atan2(dy, dx), 1X(S-1) direction of the movement
        direct = np.array([0.0] + list(direct))  # gt direction

        rot1, rot2, distance = [], [], []
        for idx in range(len(poses) - 1):
            rot1.append(direct[idx] - headings[idx])
            rot2.append(headings[idx + 1] - direct[idx])
            distance.append(np.sqrt(dx[idx] * dx[idx] + dy[idx] * dy[idx]))
        rot1, rot2, distance = np.array(rot1), np.array(rot2), np.array(distance)
        
        # add noise to commands
        commands = np.c_[rot1, distance, rot2, dz]
        commands_[1:] = commands + np.array([0.01 * np.random.randn(len(commands)),
                                             0.01 * np.random.randn(len(commands)),
                                             0.01 * np.random.randn(len(commands)),
                                             0.01 * np.random.randn(len(commands))]).T

        return commands_, headings
    
    @staticmethod
    def propagate_cov(last_yaw, dist, rot1, rot2, cov1, cov2):
        """ @Input: last_yaw: yaw in time t-1,
                    dist, rot1, rot2: motion from time t-1 to t,
                    cov1: covariance of pose in time t-1,
                    cov2: covariance of motion from time t-1 to t,
            @Description:
            cov1: 3 x 3
                / (σx(t-1))^2     σx(t-1)*σy(t-1)  σx(t-1)*σθ(t-1) \
                | σx(t-1)*σy(t-1)   (σy(t-1))^2    σy(t-1)*σθ(t-1) |
                \ σx(t-1)*σθ(t-1)  σy(t-1)*σθ(t-1)   (σθ(t-1))^2   /
            cov2: 3 x 3
                / (σdist)^2     0         0     \
                |   0       (σrot1)^2     0     |
                \   0           0     (σrot2)^2 /
            2D pose propagation formulation from time t-1 to t:
                x(t) = x(t-1) + dist*cos(θ(t-1) + rot1)
                y(t) = y(t-1) + dist*sin(θ(t-1) + rot1)
                θ(t) = θ(t-1) + rot1 + rot2
            Jacobian matrix:
                    / 1  0  0  cos(θ(t-1) + rot1)  -dist*sin(θ(t-1) + rot1)  0 \
                J = | 0  1  0  sin(θ(t-1) + rot1)   dist*cos(θ(t-1) + rot1)  0 |
                    \ 0  0  1          0                   1                 1 /
            @Return: cov: 3 x 3, J * ∑(x(t-1), y(t-1), θ(t-1), dist, rot1, rot2) * J.T
        """
        last_cov = np.zeros((6, 6))
        last_cov[:3, :3] = cov1
        last_cov[3:, 3:] = cov2
        J = np.array([
            [1, 0, 0, np.cos(last_yaw + rot1), -dist*np.sin(last_yaw + rot1), 0],
            [0, 1, 0, np.sin(last_yaw + rot1),  dist*np.cos(last_yaw + rot1), 0],
            [0, 0, 1,                       0,                             1, 1]
        ])
        cov = np.matmul(np.matmul(J, last_cov), J.T)
        return cov


if __name__ == '__main__':
  pass

