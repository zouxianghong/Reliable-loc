import numpy as np


def kitti360_calib_transform(init_T):
        cam_to_velo_data = [0.04307104361, -0.08829286498, 0.995162929, 0.8043914418,
                     -0.999004371, 0.007784614041, 0.04392796942, 0.2993489574,
                      -0.01162548558, -0.9960641394, -0.08786966659, -0.1770225824]
        cam0_to_velo = np.reshape(cam_to_velo_data, (3, 4))
        cam0_to_velo = np.vstack([cam0_to_velo, [0, 0, 0, 1]])

        cam_to_pose_data = [0.0371783278, -0.0986182135, 0.9944306009, 1.5752681039,
                             0.9992675562, -0.0053553387, -0.0378902567, 0.0043914093,
                              0.0090621821, 0.9951109327, 0.0983468786, -0.6500000000]
        cam0_to_pose = np.reshape(cam_to_pose_data, (3, 4))
        cam0_to_pose = np.vstack([cam0_to_pose, [0, 0, 0, 1]])
 
        return init_T @ cam0_to_pose @ np.linalg.inv(cam0_to_velo)


def kitti360_relative_pose(pose_1, pose_2):
    pose_1 = kitti360_calib_transform(pose_1)
    pose_2 = kitti360_calib_transform(pose_2)
    return np.linalg.inv(pose_2) @ pose_1

# def relative_pose(m1, m2):

#     return np.linalg.inv(m2) @ m1