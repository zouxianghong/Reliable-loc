import time
import os
import numpy as np
import math

from sklearn.neighbors import KDTree


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def check_makedirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def count_files():
    root = '/test/datasets/benchmark_datasets/oxford'  # datasets path
    files = os.listdir(root)
    cnt = 0
    for i in range(len(files)):
        data_path = os.path.join(root, files[i], 'pointcloud_20m_10overlap')
        cnt += len(os.listdir(data_path))
    print('data files: {}'.format(cnt))
    return cnt


def plot_point_cloud(points, label=None, output_filename=''):
    """ points is a Nx3 numpy array """
    # import matplotlib
    # matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    # ax = Axes3D(fig)
    ax = plt.subplot(111, projection='3d')
    if label is not None:
        point = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                           # cmap='RdYlBu',
                           c=label,
                           # linewidth=2,
                           alpha=0.5,
                           marker=".")
    else:
        point = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                           # cmap='RdYlBu',
                           c=points[:, 2],
                           # linewidth=2,
                           alpha=0.5,
                           marker=".")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    # fig.colorbar(point)
    # plt.axis('scaled')
    # plt.axis('off')
    if output_filename != '':
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def iou_2d(box1, box2):
    '''
        box [x1,y1,x2,y2]   分别是两对角定点的坐标
    '''
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    area_sum = area1 + area2
    # 计算重叠部分 设重叠box坐标为 [x1,y1,x2,y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x1 >= x2 or y1 >= y2:
        return 0
    else:
        inter_area = (x2 - x1) * (y2 - y1)
    return inter_area / (area_sum - inter_area)


def iou_3d(box1, box2):
    '''
        box [x1,y1,z1,x2,y2,z2]   分别是两对角定点的坐标
    '''
    area1 = (box1[3] - box1[0]) * (box1[4] - box1[1]) * (box1[5] - box1[2])
    area2 = (box2[3] - box2[0]) * (box2[4] - box2[1]) * (box2[5] - box2[2])
    area_sum = area1 + area2

    # 计算重叠部分 设重叠box坐标为 [x1,y1,z1,x2,y2,z2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    z1 = max(box1[2], box2[2])
    x2 = min(box1[3], box2[3])
    y2 = min(box1[4], box2[4])
    z2 = min(box1[5], box2[5])
    if x1 >= x2 or y1 >= y2 or z1 >= z2:
        return 0
    else:
        inter_area = (x2 - x1) * (y2 - y1) * (z2 - z1)

    return inter_area / (area_sum - inter_area)


def compute_overlap_ratio(points1, points2, use_2d=True, use_grid=True):
    """ Compute the 2D overlap ratio of two point cloud
        Input:
          points1: N x 3 point cloud, xyz only
          points2: N x 3 point cloud, xyz only
        Return:
          overlap_ratio: 0.0 ~ 1.0, overlap ratio of input point clouds
    """
    points1 = points1.reshape([-1, 3])
    points2 = points2.reshape([-1, 3])
    # min / max
    min1 = points1.min(0)
    max1 = points1.max(0)
    min2 = points2.min(0)
    max2 = points2.max(0)
    if use_2d:
        bbox1 = [min1[0], min1[1], max1[0], max1[1]]
        bbox2 = [min2[0], min2[1], max2[0], max2[1]]
        bbox_iou = iou_2d(bbox1, bbox2)
        if bbox_iou > 0.0 and use_grid:
            min12 = np.minimum(min1, min2)
            max12 = np.maximum(max1, max2)
            grid_resolution = 1.0
            size12 = np.int_((max12 - min12) / grid_resolution) + 1
            grid1 = np.zeros([size12[0], size12[1]])
            grid2 = np.zeros([size12[0], size12[1]])
            for i in range(points1.shape[0]):
                pt = points1[i, :]
                id_xyz = np.int_((pt - min12) / grid_resolution)
                grid1[id_xyz[0], id_xyz[1]] = 1
            sum_grid1 = np.sum(grid1)
            for i in range(points2.shape[0]):
                pt = points2[i, :]
                id_xyz = np.int_((pt - min12) / grid_resolution)
                grid2[id_xyz[0], id_xyz[1]] = 1
            sum_grid2 = np.sum(grid2)
            num_overlap = 0
            for i in range(size12[0]):
                for j in range(size12[1]):
                    if grid1[i, j] == 1 and grid2[i, j] == 1:
                        num_overlap += 1
            iou = float(num_overlap) / (sum_grid1 + sum_grid2 - num_overlap)
            return iou
        else:
            return bbox_iou
    else:
        bbox1 = [min1[0], min1[1], min1[2], max1[0], max1[1], max1[2]]
        bbox2 = [min2[0], min2[1], min2[2], max2[0], max2[1], max2[2]]
        bbox_iou = iou_3d(bbox1, bbox2)
        if bbox_iou > 0.0 and use_grid:
            min12 = np.minimum(min1, min2)
            max12 = np.maximum(max1, max2)
            grid_resolution = 1.0
            size12 = np.int_((max12 - min12) / grid_resolution) + 1
            grid1 = np.zeros(size12)
            grid2 = np.zeros(size12)
            for i in range(points1.shape[0]):
                pt = points1[i, :]
                id_xyz = np.int_((pt - min12) / grid_resolution)
                grid1[id_xyz[0], id_xyz[1], id_xyz[2]] = 1
            sum_grid1 = np.sum(grid1)
            for i in range(points2.shape[0]):
                pt = points2[i, :]
                id_xyz = np.int_((pt - min12) / grid_resolution)
                grid2[id_xyz[0], id_xyz[1], id_xyz[2]] = 1
            sum_grid2 = np.sum(grid2)
            num_overlap = 0
            for i in range(size12[0]):
                for j in range(size12[1]):
                    for k in range(size12[2]):
                        if grid1[i, j, k] == 1 and grid2[i, j, k] == 1:
                            num_overlap += 1
            iou = float(num_overlap) / (sum_grid1 + sum_grid2 - num_overlap)
            return iou
        else:
            return bbox_iou


def subsample_point_cloud(in_clouds, num_sample):
    """ Subsample point clouds and output T x num_sample x 3
    """
    if isinstance(in_clouds, list):
        num_cloud = len(in_clouds)
        if num_cloud == 0:
            return []
        num_points = in_clouds[0].shape[0]
        if num_sample < num_points:
            nlist = list(range(num_points))

        tmp = np.zeros((num_cloud, num_sample, 3), dtype=np.float32)
        for i in range(num_cloud):
            tidx = np.random.choice(nlist, size=num_sample, replace=False)
            tmp[i, :, :] = in_clouds[i, tidx, :]
        return tmp
    else:
        in_clouds = [in_clouds]
        return subsample_point_cloud(in_clouds, num_sample)


def get_overlap_indices(pc1, pc2, max_dist=0.2):
    pc1, pc2 = np.array(pc1), np.array(pc2)
    tree1 = KDTree(pc1)
    tree2 = KDTree(pc2)
    indices1, indices2 = set(), set()
    # nearest points: from pc2 to pc1
    dist_21 = []
    for i in range(0, pc1.shape[0]):
        dists, idxs = tree2.query(pc1[i].reshape(1, -1))
        if len(dists) == 0:
            continue
        if dists[0][0] > max_dist:
            continue
        dist_21.append(dists[0][0])
        indices1.add(i)
        indices2.add(idxs[0][0])
    # nearest points: from pc1 to pc2
    dist_12 = []
    for i in range(0, pc2.shape[0]):
        dists, idxs = tree1.query(pc2[i].reshape(1, -1))
        if len(dists) == 0:
            continue
        if dists[0][0] > max_dist:
            continue
        dist_12.append(dists[0][0])
        indices2.add(i)
        indices1.add(idxs[0][0])
    return indices1, indices2


def euler_angles_from_rotation_matrix(R):
  """ From the paper by Gregory G. Slabaugh, Computing Euler angles from a rotation matrix,
    psi, theta, phi = roll pitch yaw (x, y, z).
    Args:
      R: rotation matrix, a 3x3 numpy array
    Returns:
      a tuple with the 3 values psi, theta, phi in radians
  """

  def isclose(x, y, rtol=1.e-5, atol=1.e-8):
    return abs(x - y) <= atol + rtol * abs(y)

  phi = 0.0
  if isclose(R[2, 0], -1.0):
    theta = math.pi / 2.0
    psi = math.atan2(R[0, 1], R[0, 2])
  elif isclose(R[2, 0], 1.0):
    theta = -math.pi / 2.0
    psi = math.atan2(-R[0, 1], -R[0, 2])
  else:
    theta = -math.asin(R[2, 0])
    cos_theta = math.cos(theta)
    psi = math.atan2(R[2, 1] / cos_theta, R[2, 2] / cos_theta)
    phi = math.atan2(R[1, 0] / cos_theta, R[0, 0] / cos_theta)
  return psi, theta, phi


def get_sub_dirs(dir):
    sub_dirs = [f for f in os.listdir(dir) if os.path.isdir(os.path.join(dir, f))]
    return sub_dirs


def get_files(dir, ext_filter=None, ignore_sub_dirs=True):
    if not os.path.exists(dir):
        return []
    out_files = []
    for root, directories, files in os.walk(dir):
        if root != dir and ignore_sub_dirs:
            continue
        for filename in files:
            if os.path.splitext(filename)[1] == ext_filter:
                filepath = os.path.join(root, filename)
                out_files.append(filepath)
    return out_files


def timestamp2str(timestamp):
    if not isinstance(timestamp, float):
        return str(timestamp)
    stamp_int = int(timestamp)
    stamp_float = int((timestamp - stamp_int + 5.e-7) * 1000000)
    return str(stamp_int) + "." + str(stamp_float).zfill(6)


# 最大公约数
def gcd(m, n):
    if m <=0 or n <= 0:
        return 0
    if m > n:
        return gcd(n, m-n)
    elif n > m:
        return gcd(m, n-m)
    else:
        return m


def random_rotation_matrix():
    """
    Generates a random 3D rotation matrix from axis and angle.

    Args:
        numpy_random_state: numpy random state object

    Returns:
        Random rotation matrix.
    """
    rng = np.random.RandomState()
    axis = rng.rand(3) - 0.5
    axis /= np.linalg.norm(axis) + 1E-8
    theta = np.pi * rng.uniform(0.0, 1.0)
    thetas=axis*theta
    alpha=thetas[0]
    beta=thetas[1]
    gama=thetas[2]
    Rzalpha=np.array([[np.cos(alpha),np.sin(alpha),0],
                      [-np.sin(alpha),np.cos(alpha),0],
                      [0,0,1]])

    Rybeta=np.array([[np.cos(beta),0,-np.sin(beta)],
                     [0,1,0],
                     [np.sin(beta),0,np.cos(beta)]])

    Rzgama=np.array([[np.cos(gama),np.sin(gama),0],
                      [-np.sin(gama),np.cos(gama),0],
                      [0,0,1]])
    R=np.matmul(Rzgama,np.matmul(Rybeta,Rzalpha))
    return R


def points_to_hpoints(points):
    n,_=points.shape
    return np.concatenate([points,np.ones([n,1])],1)


def hpoints_to_points(hpoints):
    return hpoints[:,:-1]/hpoints[:,-1:]


def transform_points(pts, transform):
    h,w=transform.shape
    if h==3 and w==3:
        return pts @ transform.T
    if h==3 and w==4:
        return pts @ transform[:,:3].T + transform[:,3:].T
    elif h==4 and w==4:
        return pts @ transform[:3, :3].T + transform[:3, 3:].T
        #return hpoints_to_points(points_to_hpoints(pts) @ transform.T)
    else: raise NotImplementedError


# 最小公倍数
def lcm(m, n):
    gcd_value = gcd(m, n)
    if gcd_value == 0:
        return 0
    return m * n // gcd_value


if __name__ == "__main__":
    # test iou
    pc1 = np.load("/home/ericxhzou/Code/benchmark_datasets/whu_jisuanji/mls_submap_along_traj/pointcloud_25m_2m_clean/1637217996850017.bin")
    pc2 = np.load("/home/ericxhzou/Code/benchmark_datasets/whu_jisuanji/mls_submap_along_traj/pointcloud_25m_2m_clean/1637217997900041.bin")
    pc1 = pc1.reshape([-1, 3])
    pc2 = pc2.reshape([-1, 3])
    iou = compute_overlap_ratio(pc1, pc2, use_2d=False, use_grid=True)
    # test plot
    # points = np.random.randn(4096, 3)
    # plot_point_cloud(points, output_filename='visual/test.png')
    # test overlap indices
    start_time = time.time()
    indices1, indices2 = get_overlap_indices(pc1, pc2, max_dist=0.25)
    end_time = time.time()
    print("time cost: {:.2f}".format(end_time - start_time))  # about 0.43s
    overlap_pc1 = pc1[list(indices1)]
    overlap_pc2 = pc2[list(indices2)]
    plot_point_cloud(overlap_pc1, output_filename='overlap_pc1.png')
    plot_point_cloud(overlap_pc2, output_filename='overlap_pc2.png')
