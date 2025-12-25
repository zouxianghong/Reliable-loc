import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from monte_carlo_loc.vis_loc_result import load_from_log_file
from monte_carlo_loc.visualizer import Visualizer


# background images
images = {
    'cs_college': '/home/ericxhzou/Data/raw/cs_college.jpg',
    'info_campus': '/home/ericxhzou/Data/raw/info_campus.jpg',
    'zhongshan_park': '/home/ericxhzou/Data/raw/zhongshan_park.jpg',
    'jiefang_road': '/home/ericxhzou/Data/raw/jiefang_road.jpg',
    'yanjiang_road1': '/home/ericxhzou/Data/raw/yanjiang_road1.jpg',
    'yanjiang_road2': '/home/ericxhzou/Data/raw/yanjiang_road2.jpg'
}
image_corners = {  # xmin, xmax, ymin, ymax
    'cs_college': [-135.65, 161.24, -238.63, 186.0],
    'info_campus': [-323.42, 356.24, -481.99, 208.36],
    'zhongshan_park': [-1724.47, 202.43, -621.16, 1670.43],
    'jiefang_road': [-127.00, 3076.81, -317.10, 3051.90],
    'yanjiang_road1': [2284.68, 3696.69, 433.75, 2657.17],
    'yanjiang_road2': [500.50, 2831.35, -2299.36, 958.30]
}

class DemoVisualizer(object):
    def __init__(self, gt_locations, est_locations, loc_modes, location_errs, yaw_errs, particles):
        """ Initialization:
            map_size: the size of the given map
            poses: ground truth poses.
            numParticles: number of particles.
            start_idx: the start index.
            converge_thres: a threshold used to tell whether the localization converged or not.
        """
        # Setup the figure and axes...
        self.fig, self.ax = plt.subplots(figsize=(16, 10))
        self.font_size = 13
        
        # set background image
        self.bg_img = None
        self.bg_img_extent = None  # xmin, xmax, ymin, ymax
        
        # gt / estimated locations
        self.gt_locations = gt_locations
        self.est_locations = est_locations
        
        # location / yaw erros
        self.loc_modes = loc_modes
        self.location_errs = location_errs
        self.yaw_errs = yaw_errs
        
        # particles
        self.particles = particles
        
        # # setup
        # self.setup_plot()
        
    def setup_plot(self):
        """ Initial drawing of the scatter plot.
        """
        self.ax_gt, = self.ax.plot([], [], c='r', label='reference pose')
        self.ax_est, = self.ax.plot([], [], c='b', label='estimated pose')
        self.scat = self.ax.scatter([], [], c=[], s=1, cmap="Blues", vmin=0, vmax=1)
        self.ax.axis('square')
        border_width = 50
        map_size2 = [
            np.min(self.gt_locations[:, 0]), np.max(self.gt_locations[:, 0]),
            np.min(self.gt_locations[:, 1]), np.max(self.gt_locations[:, 1])
        ]
        if map_size2[1] - map_size2[0] > 1000 or map_size2[3] - map_size2[2] > 1000:
            border_width = 125
        map_size = [
            np.max([map_size2[0]-border_width, self.bg_img_extent[0]]), np.min([map_size2[1]+border_width, self.bg_img_extent[1]]),
            np.max([map_size2[2]-border_width, self.bg_img_extent[2]]), np.min([map_size2[3]+border_width, self.bg_img_extent[3]])
        ]
        self.ax.set(xlim=map_size[:2], ylim=map_size[2:])
        self.ax.set_xlabel('X [m]', fontsize=self.font_size)
        self.ax.set_ylabel('Y [m]', fontsize=self.font_size)
        plt.xticks(fontsize=self.font_size)  # 默认字体大小为10
        plt.yticks(fontsize=self.font_size)
        self.ax.legend()
        leg = plt.gca().get_legend()
        ltext = leg.get_texts()
        plt.setp(ltext, fontsize=self.font_size)
        # background
        if self.bg_img is not None and self.bg_img_extent is not None:
            self.ax.imshow(self.bg_img, extent=self.bg_img_extent, alpha=0.9)
        
        self.patches = [self.ax_gt, self.ax_est, self.scat]
        return self.patches
    
    def update(self, frame_idx):
        loc_mode = 'Reg' if self.loc_modes[frame_idx] else 'PF'
        is_converged = np.linalg.norm(self.particles[frame_idx][1000]) < 1e-10
        title = f"""Loc Mode: {loc_mode}, Converged: {is_converged}, Position Error: {self.location_errs[frame_idx]:<.2f}m, Yaw Error: {self.yaw_errs[frame_idx]:<.2f}deg"""
        self.ax.set_title(title, loc='center', fontsize=self.font_size)
        # set ground truth
        self.ax_gt.set_data(self.gt_locations[:frame_idx, 0],
                            self.gt_locations[:frame_idx, 1])

        # set estimated pose
        start_idx = frame_idx - 1500 if frame_idx > 1500 else 0
        est_locations = self.est_locations[start_idx:frame_idx]
        location_errs = self.location_errs[start_idx:frame_idx]
        est_locations = est_locations[location_errs < 60]
        self.ax_est.set_data(est_locations[:, 0],
                             est_locations[:, 1])

        # Set x and y data
        particle_xyc = self.particles[frame_idx]
        self.scat.set_offsets(particle_xyc[:, :2])
        # Set colors
        self.scat.set_array(particle_xyc[:, -1] * 0.8 + 0.2)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        return self.patches


# visualize from npz
def vis_from_npz(exp_dir):
    # load loc results
    exp_res = load_from_log_file(exp_dir, return_particles=True)
    # vis
    plt.ion()
    visualizer = DemoVisualizer(exp_res.gt_locations, exp_res.est_locations, exp_res.loc_modes,
                                exp_res.position_errs, exp_res.yaw_errs, exp_res.particles)
    visualizer.bg_img = mpimg.imread(images[exp_res.dataset_name])
    visualizer.bg_img_extent = image_corners[exp_res.dataset_name]
    visualizer.setup_plot()
    for frame_idx in range(len(exp_res.particles)):
        visualizer.update(frame_idx)
        visualizer.fig.canvas.draw()
        visualizer.fig.canvas.flush_events()
    
if __name__ == '__main__':
    exp_dirs = [
        # '/home/ericxhzou/Code/ppt-net-plus/exp/mc_loc/2024-04-22T20-58-57_zhongshan_park_reliable_0.0005',
        '/home/ericxhzou/Code/ppt-net-plus/exp/mc_loc/2024-04-26T21-05-22_zhongshan_park_reliable_0.0005'
        # '/home/ericxhzou/Code/ppt-net-plus/exp/mc_loc/result3/2024-03-22T16-22-58_cs_college_reliable_0.0005_param',
        # '/home/ericxhzou/Code/ppt-net-plus/exp/mc_loc/result3/2024-03-22T16-48-18_info_campus_reliable_0.0005_param',
        # '/home/ericxhzou/Code/ppt-net-plus/exp/mc_loc/result3/2024-03-22T17-30-55_zhongshan_park_reliable_0.0005_param',
        # '/home/ericxhzou/Code/ppt-net-plus/exp/mc_loc/result3/2024-03-22T18-23-40_jiefang_road_reliable_0.0005_param',
        # '/home/ericxhzou/Code/ppt-net-plus/exp/mc_loc/result3/2024-03-22T19-21-42_yanjiang_road1_reliable_0.0001_param',
        # '/home/ericxhzou/Code/ppt-net-plus/exp/mc_loc/result3/2024-03-22T20-24-43_yanjiang_road2_reliable_0.0005_param'
    ]
    for exp_dir in exp_dirs:
        vis_from_npz(exp_dir)