import os.path
import random

import matplotlib
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
# plt.rc('font', family='Times New Roman')

from utils.loading_pointclouds import normalize_point_cloud, normalize_point_clouds, load_pc_file, load_pc_files
from utils.util import get_files


# get color by value
def get_color_by_value(value, min_value=-1.0, max_value=1.0):
    norm = matplotlib.colors.Normalize(vmin=min_value, vmax=max_value)
    rgb = list(matplotlib.cm.jet(norm(value), bytes=True))[:3]
    color = '#'
    for i in range(len(rgb)):
        num = int(rgb[i])
        color += str(hex(num))[-2:].replace('x', '0').upper()
    return color


# get random color
def get_random_color():
    color_groups = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'pink', 'chocolate', 'cyan', 'lime', 'olive']
    color = random.choice(color_groups)
    return color


# draw point cloud in matplot
def draw_pc(pc_file, save_filepath=None, title_info='', pt_size=3, show_fig=False, use_np_load=True, dtype=np.float64):
    if not show_fig:
        matplotlib.use('Agg')
    pc = load_pc_file(pc_file, use_np_load=use_np_load, dtype=dtype)
    if pc.shape[0] > 4096:
        sample_idxs = np.random.choice(pc.shape[0], 4096, replace=False)
        pc = pc[sample_idxs]
    pc = normalize_point_cloud(pc)
    x = pc[:, 0]
    y = pc[:, 1]
    z = pc[:, 2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, s=pt_size, c=z,  # height data for color
               cmap='rainbow')
    ax.set_title(title_info, fontsize=30)
    ax.axis()
    # set init view
    ax.view_init(elev=65.0, azim=-45.0)
    if save_filepath:
        fig.savefig(save_filepath, transparent=False, bbox_inches='tight')
    if show_fig:
        plt.show()
    else:
        plt.close('all')


# draw point cloud with colors
def draw_pc_with_color(pc, color_values, pt_size=3, save_filepath=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    min_value, max_value = np.min(color_values), np.max(color_values)
    for i in range(pc.shape[0]):
        color = get_color_by_value(color_values[i], min_value, max_value)
        ax.scatter(pc[i, 0], pc[i, 1], pc[i, 2], s=pt_size, color=color)
    ax.axis()
    ax.set_aspect('equal')
    # set init view
    ax.view_init(elev=90.0, azim=-90.0)
    # set axis label
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    # adjust layout
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9, wspace=0.1, hspace=0.2)
    plt.tight_layout()
    if save_filepath:
        fig.savefig(save_filepath, transparent=False, bbox_inches='tight')
    else:
        plt.close('all')


def draw_pcs(pc_files, extra_infos, save_filepath=None, pt_size=20, show_fig=False):
    if not show_fig:
        matplotlib.use('Agg')
    fig = plt.figure(figsize=(75, len(pc_files)*75))
    gs = fig.add_gridspec(1, len(pc_files))
    for i in range(len(pc_files)):
        pc = load_pc_file(pc_files[i], use_np_load=True)
        pc = normalize_point_cloud(pc)
        x = pc[:, 0]
        y = pc[:, 1]
        z = pc[:, 2]
        ax = fig.add_subplot(gs[:, i], projection='3d')
        ax.scatter(x, y, z, s=pt_size, c=z,  # height data for color
                   cmap='rainbow')
        ax.set_title(extra_infos[i], fontsize=30)
        ax.axis()
    plt.tight_layout()
    if save_filepath:
        fig.savefig(save_filepath, transparent=False, bbox_inches='tight')
    if show_fig:
        plt.show()
    else:
        plt.close('all')


# draw pcs in dir
def draw_pcs_in_dir(in_dir, pt_size=3, show_fig=False):
    # get files in in_dir
    pc_files = get_files(in_dir, '.bin')
    for pc_f in pc_files:
        svg_f = os.path.join(in_dir, os.path.splitext(os.path.basename(pc_f))[0] + '.svg')
        draw_pc(pc_f, svg_f, pt_size=pt_size, show_fig=show_fig)


# add border to image
def add_img_border(src, loc='a', width=3, color=(0, 0, 0, 255)):
    """
        src: (str) 需要加边框的图片
        loc: (str) 边框添加的位置, 默认是'a'(
            四周: 'a' or 'all'
            上: 't' or 'top'
            右: 'r' or 'rigth'
            下: 'b' or 'bottom'
            左: 'l' or 'left' )
        width: (int) 边框宽度 (默认是3)
        color: (int or 3-tuple) 边框颜色 (默认是0, 表示黑色; 也可以设置为三元组表示RGB颜色)
    """
    # 读取图片
    w = src.size[0]
    h = src.size[1]

    # 添加边框
    dst = None
    if loc in ['a', 'all']:
        w += 2 * width
        h += 2 * width
        dst = Image.new('RGBA', (w, h), color)
        dst.paste(src, (width, width))
    elif loc in ['t', 'top']:
        h += width
        dst = Image.new('RGBA', (w, h), color)
        dst.paste(src, (0, width, w, h))
    elif loc in ['r', 'right']:
        w += width
        dst = Image.new('RGBA', (w, h), color)
        dst.paste(src, (0, 0, w - width, h))
    elif loc in ['b', 'bottom']:
        h += width
        dst = Image.new('RGBA', (w, h), color)
        dst.paste(src, (0, 0, w, h - width))
    elif loc in ['l', 'left']:
        w += width
        dst = Image.new('RGBA', (w, h), color)
        dst.paste(src, (width, 0, w, h))
    else:
        pass
    return dst


# draw features with t-SNE
def draw_features_with_tsne(features, labels, pc_idxs, pc_files, out_path, rewrite_img=True):
    # draw pcs and save to files
    image_files = []
    temp_path = os.path.join(os.path.dirname(out_path), 'temp')
    if not os.path.exists(temp_path):
        os.mkdir(temp_path)
    for i in range(len(pc_files)):
        save_filepath = os.path.join(temp_path, '{}.png'.format(i))
        image_files.append(save_filepath)
        if not os.path.exists(save_filepath) and rewrite_img:
            draw_pc(pc_files[i], save_filepath, 'id: {}'.format(pc_idxs[i]))
    # draw t-sne
    feat_tsne = TSNE(perplexity=5).fit_transform(features)  # high dim -> 2 dim
    tx, ty = feat_tsne[:, 0], feat_tsne[:, 1]
    tx = (tx - np.min(tx)) / (np.max(tx) - np.min(tx))
    ty = (ty - np.min(ty)) / (np.max(ty) - np.min(ty))
    bk_width, bk_height, max_dim = 4000, 3000, 175
    background = Image.new('RGBA', (bk_width, bk_height), (255, 255, 255, 255))
    for x, y, img_file, lbl in zip(tx, ty, image_files, labels):
        img = Image.open(img_file)
        if int(lbl) > 1:  # query
            border_color = (128, 0, 128, 255)  # purple
        elif int(lbl) == 1:
            border_color = (0, 255, 0, 255)  # green
        else:
            border_color = (255, 0, 0, 255)  # red
        img = add_img_border(img, width=5, color=border_color)
        rs = max(1, img.width / max_dim, img.height / max_dim)
        img = img.resize((int(img.width / rs), int(img.height / rs)), Image.ANTIALIAS)
        background.paste(img, (int((bk_width - max_dim) * x), int((bk_height - max_dim) * y)), img)
    background.save(out_path)


# draw curve
def draw_line_chart(data_list, title='', xlabel='', ylabel='', xrange=[0, 25], xstep=1,
                    yrange=[50, 100], ystep=10, font_size=15, save_filepath=None):
    # 如果要显示中文字体,则在此处设为：SimHeiplt.rcParams['axes.unicode_minus'] = False # 显示负号
    plt.rcParams['font.sans-serif'] = ['Arial']
    x = np.arange(xrange[0] + xstep, xrange[1] + xstep, step=xstep)
    y = np.arange(yrange[0], yrange[1] + ystep, step=ystep)

    # label在图示(legend)中显示。若为数学公式,则最好在字符串前后添加"$"符号
    # color：b:blue、g:green、r:red、c:cyan、m:magenta、y:yellow、k:black、w:white、、、
    # 线型：- -- -. : ,# marker：. , o v < * + 1
    plt.figure(figsize=(10, 5))
    plt.grid(linestyle="--")  # 设置背景网格线为虚线
    ax = plt.gca()
    ax.spines['top'].set_visible(False)  # 去掉上边框
    ax.spines['right'].set_visible(False)  # 去掉右边框

    for data_i in data_list:
        plt.plot(x, data_i.data, marker=data_i.fig_marker, color=data_i.fig_color, label=data_i.name, linewidth=1.5)

    xstep = 5
    x = np.arange(xrange[0] + xstep, xrange[1] + xstep, step=xstep)
    plt.xticks(x, fontsize=font_size, fontweight='bold')  # 默认字体大小为10
    plt.yticks(y, fontsize=font_size, fontweight='bold')
    plt.title(label=title, pad=20, fontsize=font_size, fontweight='bold')  # 默认字体大小为12
    plt.xlabel(xlabel=xlabel, fontsize=font_size, fontweight='bold')
    plt.ylabel(ylabel=ylabel, fontsize=font_size, fontweight='bold')
    plt.xlim(xrange[0], xrange[1])  # 设置x轴的范围
    plt.ylim(yrange[0], yrange[1])

    plt.legend()  # 显示各曲线的图例
    plt.legend(loc=4, numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=font_size, fontweight='bold')  # 设置图例字体的大小和粗细

    if save_filepath:
        plt.savefig(save_filepath, format='svg') # 建议保存为svg格式,再用inkscape转为矢量图emf后插入word中plt.show()
    plt.show()


# draw two clouds with point pairs
def draw_pc_pps(src_pc, src_kpt, tgt_pcs, tgt_kpt, tgt_states=None, offset_x=90.0, pt_size=3, title=None, save_filepath=None):
    # if save_filepath is None:
    #     matplotlib.use('Agg')
    # centralize clouds
    src_center = np.mean(src_kpt, axis=0, keepdims=True)
    src_pc = src_pc - src_center
    src_kpt = src_kpt - src_center
    tgt_center = src_center - np.array([[offset_x, 0.0, 0.0]])
    tgt_kpt = tgt_kpt - tgt_center
    # src/tgt clouds
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(src_pc[:, 0], src_pc[:, 1], src_pc[:, 2], s=pt_size, color='purple')
    if isinstance(tgt_pcs, list):
        for i in range(len(tgt_pcs)):
            tgt_pc = tgt_pcs[i] - tgt_center
            tgt_state = 1 if tgt_states is None else tgt_states[i]
            tgt_color = 'green' if tgt_state else 'red'
            ax.scatter(tgt_pc[:, 0], tgt_pc[:, 1], tgt_pc[:, 2], s=pt_size, color=tgt_color)
    else:
        tgt_pcs = tgt_pcs - tgt_center
        ax.scatter(tgt_pcs[:, 0], tgt_pcs[:, 1], tgt_pcs[:, 2], s=pt_size, color='green')
    # point pairs
    for i in range(len(src_kpt)):
        pps_color = get_random_color()
        ax.plot([src_kpt[i, 0], tgt_kpt[i, 0]], [src_kpt[i, 1], tgt_kpt[i, 1]],
                [src_kpt[i, 2], tgt_kpt[i, 2]], linewidth=2, color=pps_color)
    # title
    if title is not None:
        ax.set_title(title, fontsize=12, pad=0)
    ax.axis()
    ax.set_aspect('equal')
    # set init view
    ax.view_init(elev=90.0, azim=-90.0)
    # set axis label
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    # adjust layout
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9, wspace=0.1, hspace=0.2)
    plt.tight_layout()
    # save
    if save_filepath is not None:
        plt.savefig(save_filepath, dpi=200, bbox_inches='tight')
        plt.close('all')
    else:
        plt.show()


# draw two clouds with point pairs
def draw_pps(src_kpt, tgt_kpt, offset_x=90.0, title=None, save_filepath=None):
    # if save_filepath is None:
    #     matplotlib.use('Agg')
    # centralize clouds
    if offset_x is not None:
        src_center = np.mean(src_kpt, axis=0, keepdims=True)
        src_kpt = src_kpt - src_center
        tgt_center = src_center - np.array([[offset_x, 0.0, 0.0]])
        tgt_kpt = tgt_kpt - tgt_center
    else:
        src_center = np.mean(src_kpt, axis=0, keepdims=True)
        tgt_center = np.mean(tgt_kpt, axis=0, keepdims=True)
        center = (src_center + tgt_center) / 2
        src_kpt = src_kpt - center
        tgt_kpt = tgt_kpt - center
    # fig
    fig = plt.figure(num=2)
    fig.clear(True)
    ax = fig.add_subplot(111, projection='3d')
    # point pairs
    for i in range(len(src_kpt)):
        pps_color = get_random_color()
        ax.plot([src_kpt[i, 0], tgt_kpt[i, 0]], [src_kpt[i, 1], tgt_kpt[i, 1]],
                [src_kpt[i, 2], tgt_kpt[i, 2]], linewidth=2, color=pps_color)
    # title
    if title is not None:
        ax.set_title(title, fontsize=12, pad=0)
    ax.axis()
    ax.set_aspect('equal')
    # set init view
    ax.view_init(elev=90.0, azim=-90.0)
    # set axis label
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    # adjust layout
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9, wspace=0.1, hspace=0.2)
    plt.tight_layout()
    # save
    if save_filepath is not None:
        plt.savefig(save_filepath, dpi=200, bbox_inches='tight')
        plt.close('all')
    else:
        plt.show()

# draw anchor and top k clouds and centroid
def draw_anchor_top_k(a_k_pcs, a_pts, k_pts, a_k_states, pt_size=3, title=None, save_filepath=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # clouds
    for i in range(len(a_k_pcs)):
        tgt_pc = a_k_pcs[i]
        tgt_color = get_random_color()
        ax.scatter(tgt_pc[:, 0], tgt_pc[:, 1], tgt_pc[:, 2], s=pt_size, color=tgt_color)
    # point pairs
    max_state = np.max(a_k_states)
    for i in range(len(a_pts)):
        pps_color = get_color_by_value(a_k_states[i], min_value=0.5, max_value=max_state)
        ax.plot([a_pts[i, 0], k_pts[i, 0]], [a_pts[i, 1], k_pts[i, 1]], [a_pts[i, 2], k_pts[i, 2]],
                linewidth=2, color=pps_color)
    # title
    if title is not None:
        ax.set_title(title, fontsize=12, pad=0)
    ax.axis()
    ax.set_aspect('equal')
    # set init view
    ax.view_init(elev=90.0, azim=-90.0)
    # set axis label
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    # adjust layout
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9, wspace=0.1, hspace=0.2)
    plt.tight_layout()
    # save
    if save_filepath is not None:
        plt.savefig(save_filepath, dpi=200, bbox_inches='tight')
        plt.close('all')
    else:
        plt.show()


# draw two clouds without point pairs
def draw_two_pc(src_pc, tgt_pcs, tgt_states=None, use_diff_center=False, pt_size=3, title=None, save_filepath=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # src
    center = np.mean(src_pc, axis=0, keepdims=True)
    src_pc = src_pc - center
    # tgt
    if use_diff_center:
        center = np.mean(tgt_pcs.reshape(-1,3), axis=0, keepdims=True) - np.array([[75.0, 0.0, 0.0]])
    tgt_pcs = tgt_pcs - center
    if len(tgt_pcs.shape) == 2:
        ax.scatter(tgt_pcs[:, 0], tgt_pcs[:, 1], tgt_pcs[:, 2], s=pt_size, color='red')
    else:  # len(tgt_pcs.shape) == 3
        for i in range(tgt_pcs.shape[0]):
            tgt_pc = tgt_pcs[i]
            if tgt_states is None:
                tgt_color = 'red'
            else:
                tgt_color = 'green' if tgt_states[i] else 'red'
            ax.scatter(tgt_pc[:, 0], tgt_pc[:, 1], tgt_pc[:, 2], s=pt_size, color=tgt_color)
    ax.scatter(src_pc[:, 0], src_pc[:, 1], src_pc[:, 2], s=pt_size*3, color='purple')
    # title
    if title is not None:
        ax.set_title(title, fontsize=12, pad=0)
    ax.axis()
    ax.set_aspect('equal')
    # set init view
    ax.view_init(elev=90.0, azim=-90.0)
    # set axis label
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    # adjust layout
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9, wspace=0.1, hspace=0.2)
    plt.tight_layout()
    # save
    if save_filepath is not None:
        plt.savefig(save_filepath, dpi=200, bbox_inches='tight')
        plt.close('all')
    else:
        plt.show()


def draw_cov_ellipse(mean_x, mean_y, cov, ratio=3):
    """ @Input: mean_x: float, mean_y: float, ellipse center
        covariance matrix:
        /              \
        | σx*σx  σx*σy |
        | σx*σy  σy*σy |
        \              /
        ellipse: refer to https://cookierobotics.com/007/
        λ1 = (σx*σx + σy*σy) / 2 + ((σx*σx - σy*σy)^2 / 4 + σx*σx*σy*σy)^0.5
        λ2 = (σx*σx + σy*σy) / 2 - ((σx*σx - σy*σy)^2 / 4 + σx*σx*σy*σy)^0.5
             / 0                          if σx*σy = 0 and σx*σx >= σy*σy
        θ =  | pi/2                       if σx*σy = 0 and σx*σx < σy*σy
             \ atan(σx*σy / (λ1-σx*σx))   else
    """
    epsilon = 1e-10
    # ellipse
    ellipse_center = (mean_x, mean_y)
    sigma_xx = cov[0,0]
    sigma_xy = cov[0,1]
    sigma_yy = cov[1,1]
    part1 = (sigma_xx + sigma_yy) / 2
    part2 = np.sqrt((sigma_xx - sigma_yy)**2 / 4 + sigma_xy**2)
    ellipse_width = np.sqrt(part1 + part2)
    ellipse_height = np.sqrt(part1 - part2)
    ellipse_theta = 0
    if np.abs(sigma_xy) < epsilon:
        if sigma_xx >= sigma_yy:
            ellipse_theta = 0
        else:
            ellipse_theta = np.pi / 2
    else:
        ellipse_theta = np.arctan2(ellipse_width**2 - sigma_xx, sigma_xy)
    ellipse_theta = ellipse_theta * 180 / np.pi
    # figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ellipse = Ellipse(xy=ellipse_center,
                      width=ellipse_width * ratio,
                      height=ellipse_height * ratio,
                      angle=ellipse_theta)
    ax.add_patch(p=ellipse)    # 向子区添加形状
    ax.axis([-ellipse_width * ratio, ellipse_width * ratio, -ellipse_height * ratio, ellipse_height * ratio])
    ellipse.set(alpha=0.4, color='lightskyblue')
    plt.show()
    

# recall / precision @topN struct
class RecallPrecisionTopN:
    def __int__(self):
        self.name = 'unknown'
        self.fig_color = 'blue'
        self.fig_marker = '.'
        self.data = np.array()


# draw place recognition result: recall@top1~25 curve data A
def draw_pr_recall_top1_25_data_A():
    data_list = []
    # PointNetVLAD
    data_pointnetvlad = RecallPrecisionTopN()
    data_pointnetvlad.name = 'PointNetVLAD'
    data_pointnetvlad.fig_color = 'blue'
    data_pointnetvlad.fig_marker = 'v'
    data_pointnetvlad.data = np.array([31.68851195, 34.23284503, 37.00848111, 39.9383192, 42.48265227, 45.41249036,
                                       47.41711642, 49.42174248, 51.34926754, 53.58519661, 55.12721665, 56.20663069,
                                       57.74865073, 59.05936777, 60.0616808, 60.52428682, 61.60370085, 62.37471087,
                                       63.14572089, 64.07093292, 64.68774094, 64.84194295, 65.45875096, 65.99845798,
                                       66.615266])
    data_list.append(data_pointnetvlad)
    # PPT-Net: original version
    data_pptnet = RecallPrecisionTopN()
    data_pptnet.name = 'PPT-Net'
    data_pptnet.fig_color = 'green'
    data_pptnet.fig_marker = 's'
    data_pptnet.data = np.array([67.38627602, 73.63145721, 77.33230532, 79.8766384, 81.49575944, 82.65227448,
                                 84.04009252, 84.96530455, 85.73631457, 86.35312259, 86.8157286, 87.20123362,
                                 87.35543562, 87.43253662, 87.66383963, 88.35774865, 88.66615266, 89.05165767,
                                 89.51426369, 89.66846569, 89.82266769, 90.0539707, 90.20817271, 90.36237471,
                                 90.59367772])
    data_list.append(data_pptnet)
    # PPT-Net with L2 Norm
    data_pptnet_w_l2norm = RecallPrecisionTopN()
    data_pptnet_w_l2norm.name = 'PPT-Net w/ L2 Norm'
    data_pptnet_w_l2norm.fig_color = 'green'
    data_pptnet_w_l2norm.fig_marker = 'x'
    data_pptnet_w_l2norm.data = np.array([70.77872012, 75.32767926, 78.18041635, 79.9537394, 81.80416345, 82.88357749,
                                          83.73168851, 84.65690054, 85.35080956, 85.96761758, 86.5073246, 86.89282961,
                                          87.50963763, 87.66383963, 88.12644564, 88.43484965, 88.66615266, 89.12875867,
                                          89.36006168, 89.9768697, 90.36237471, 90.51657672, 90.74787972, 91.13338473,
                                          91.13338473])
    data_list.append(data_pptnet_w_l2norm)
    # Minkloc3D v2 without data augmentation (DA)
    data_minkloc3dv2_wo_DA = RecallPrecisionTopN()
    data_minkloc3dv2_wo_DA.name = 'Minkloc3D v2 w/o DA'
    data_minkloc3dv2_wo_DA.fig_color = 'chocolate'
    data_minkloc3dv2_wo_DA.fig_marker = '^'
    data_minkloc3dv2_wo_DA.data = np.array([72.93754819, 78.72012336, 82.26676947, 84.11719352, 86.12181958, 87.58673863,
                                            88.12644564, 88.74325366, 89.43716268, 90.1310717, 90.74787972, 90.97918273,
                                            91.28758674, 91.82729375, 92.36700077, 92.90670779, 93.2922128, 93.60061681,
                                            93.98612182, 93.98612182, 94.21742483, 94.29452583, 94.44872783, 94.52582884,
                                            94.60292984])
    data_list.append(data_minkloc3dv2_wo_DA)
    # PatchAugNet(ours)
    data_patchaugnet = RecallPrecisionTopN()
    data_patchaugnet.name = 'PatchAugNet(ours)'
    data_patchaugnet.fig_color = 'red'
    data_patchaugnet.fig_marker = 'd'
    data_patchaugnet.data = np.array([76.40709329, 80.33924441, 82.03546646, 83.73168851, 84.81110254, 85.58211257,
                                      86.5073246, 87.43253662, 88.28064765, 88.74325366, 89.28296068, 89.59136469,
                                      89.8997687, 90.20817271, 90.36237471, 90.67077872, 90.97918273, 91.13338473,
                                      91.13338473, 91.44178874, 91.59599075, 91.75019275, 91.82729375, 92.05859676,
                                      92.13569776])
    data_list.append(data_patchaugnet)
    save_filepath = '/home/ericxhzou/place recognition recall on Data A.svg'
    draw_line_chart(data_list, title='Evaluation result on Data A', xlabel='N-Number of top database candidates',
                    ylabel='Recall @N(%)', save_filepath=save_filepath, yrange=[20, 100], ystep=10)


# draw place recognition result: precision@top1~25 curve data A
def draw_pr_precision_top1_25_data_A():
    data_list = []
    # PointNetVLAD
    data_pointnetvlad = RecallPrecisionTopN()
    data_pointnetvlad.name = 'PointNetVLAD'
    data_pointnetvlad.fig_color = 'blue'
    data_pointnetvlad.fig_marker = 'v'
    data_pointnetvlad.data = np.array([31.68851195, 31.07170393, 30.55769725, 30.30069391, 30.1002313, 30.05654074,
                                       29.74997246, 29.49113338, 29.34978155, 28.98226677, 28.70260041, 28.49524544,
                                       28.14186584, 27.75636083, 27.40683629, 27.06727062, 26.77218921, 26.48847768,
                                       26.07637057, 25.66692367, 25.25975695, 24.87909161, 24.55164091, 24.27075302,
                                       23.89514264])
    data_list.append(data_pointnetvlad)
    # PPT-Net: original version
    data_pptnet = RecallPrecisionTopN()
    data_pptnet.name = 'PPT-Net'
    data_pptnet.fig_color = 'green'
    data_pptnet.fig_marker = 's'
    data_pptnet.data = np.array([67.38627602, 66.692367, 66.17836032, 65.76715497, 65.42791056, 64.99614495,
                                 64.14803392, 63.53122591, 62.71738199, 62.24363917, 61.63173758, 60.85196608,
                                 60.19215942, 59.604582, 58.95142637, 58.4232845, 57.61712549, 56.81915532,
                                 56.06054458, 55.29298381, 54.46267944, 53.66930679, 52.8175388, 52.02390131,
                                 51.23207402])
    data_list.append(data_pptnet)
    # PPT-Net with L2 Norm
    data_pptnet_w_l2norm = RecallPrecisionTopN()
    data_pptnet_w_l2norm.name = 'PPT-Net w/ L2 Norm'
    data_pptnet_w_l2norm.fig_color = 'green'
    data_pptnet_w_l2norm.fig_marker = 'x'
    data_pptnet_w_l2norm.data = np.array([70.77872012, 68.96684657, 68.20868671, 67.52120278, 66.92367001, 66.01130815,
                                          65.22744796, 64.52390131, 63.88246381, 63.09175019, 62.43078433, 61.57800051,
                                          60.91572267, 60.12776738, 59.26497044, 58.461835, 57.66701438, 56.78917159,
                                          56.01590715, 55.23130301, 54.53978045, 53.69033434, 52.7974255, 51.95001285,
                                          51.09020817])
    data_list.append(data_pptnet_w_l2norm)
    # PatchAugNet(ours)
    data_patchaugnet = RecallPrecisionTopN()
    data_patchaugnet.name = 'PatchAugNet(ours)'
    data_patchaugnet.fig_color = 'red'
    data_patchaugnet.fig_marker = 'd'
    data_patchaugnet.data = np.array([76.40709329, 75.55898227, 75.09637625, 74.26754048, 73.73939861, 73.07890003,
                                      72.59610089, 71.94487278, 71.28416003, 70.66306862, 70.03574683, 69.3459265,
                                      68.53686021, 67.76627382, 66.79002827, 65.8924441, 64.90543789, 63.87818042,
                                      62.70746257, 61.7000771, 60.63075963, 59.52547838, 58.39562871, 57.32138268,
                                      56.18195837])
    data_list.append(data_patchaugnet)
    save_filepath = '/home/ericxhzou/place recognition precision on Data A.svg'
    draw_line_chart(data_list, title='Evaluation result on Data A', xlabel='N-Number of top database candidates',
                    ylabel='Precision @N(%)', save_filepath=save_filepath, yrange=[20, 80], ystep=10)
    data_list = []
    # PointNetVLAD
    data_pointnetvlad = RecallPrecisionTopN()
    data_pointnetvlad.name = 'PointNetVLAD'
    data_pointnetvlad.fig_color = 'blue'
    data_pointnetvlad.fig_marker = 'v'
    data_pointnetvlad.data = np.array([31.68851195, 34.23284503, 37.00848111, 39.9383192, 42.48265227, 45.41249036,
                                       47.41711642, 49.42174248, 51.34926754, 53.58519661, 55.12721665, 56.20663069,
                                       57.74865073, 59.05936777, 60.0616808, 60.52428682, 61.60370085, 62.37471087,
                                       63.14572089, 64.07093292, 64.68774094, 64.84194295, 65.45875096, 65.99845798,
                                       66.615266])
    data_list.append(data_pointnetvlad)
    # PPT-Net: original version
    data_pptnet = RecallPrecisionTopN()
    data_pptnet.name = 'PPT-Net'
    data_pptnet.fig_color = 'green'
    data_pptnet.fig_marker = 's'
    data_pptnet.data = np.array([67.38627602, 73.63145721, 77.33230532, 79.8766384, 81.49575944, 82.65227448,
                                 84.04009252, 84.96530455, 85.73631457, 86.35312259, 86.8157286, 87.20123362,
                                 87.35543562, 87.43253662, 87.66383963, 88.35774865, 88.66615266, 89.05165767,
                                 89.51426369, 89.66846569, 89.82266769, 90.0539707, 90.20817271, 90.36237471,
                                 90.59367772])
    data_list.append(data_pptnet)
    # PPT-Net with L2 Norm
    data_pptnet_w_l2norm = RecallPrecisionTopN()
    data_pptnet_w_l2norm.name = 'PPT-Net w/ L2 Norm'
    data_pptnet_w_l2norm.fig_color = 'green'
    data_pptnet_w_l2norm.fig_marker = 'x'
    data_pptnet_w_l2norm.data = np.array([70.77872012, 75.32767926, 78.18041635, 79.9537394, 81.80416345, 82.88357749,
                                          83.73168851, 84.65690054, 85.35080956, 85.96761758, 86.5073246, 86.89282961,
                                          87.50963763, 87.66383963, 88.12644564, 88.43484965, 88.66615266, 89.12875867,
                                          89.36006168, 89.9768697, 90.36237471, 90.51657672, 90.74787972, 91.13338473,
                                          91.13338473])
    data_list.append(data_pptnet_w_l2norm)
    # Minkloc3D v2 without data augmentation (DA)
    data_minkloc3dv2_wo_DA = RecallPrecisionTopN()
    data_minkloc3dv2_wo_DA.name = 'Minkloc3D v2 w/o DA'
    data_minkloc3dv2_wo_DA.fig_color = 'chocolate'
    data_minkloc3dv2_wo_DA.fig_marker = '^'
    data_minkloc3dv2_wo_DA.data = np.array(
        [72.93754819, 78.72012336, 82.26676947, 84.11719352, 86.12181958, 87.58673863,
         88.12644564, 88.74325366, 89.43716268, 90.1310717, 90.74787972, 90.97918273,
         91.28758674, 91.82729375, 92.36700077, 92.90670779, 93.2922128, 93.60061681,
         93.98612182, 93.98612182, 94.21742483, 94.29452583, 94.44872783, 94.52582884,
         94.60292984])
    data_list.append(data_minkloc3dv2_wo_DA)
    # PatchAugNet(ours)
    data_patchaugnet = RecallPrecisionTopN()
    data_patchaugnet.name = 'PatchAugNet(ours)'
    data_patchaugnet.fig_color = 'red'
    data_patchaugnet.fig_marker = 'd'
    data_patchaugnet.data = np.array([76.40709329, 80.33924441, 82.03546646, 83.73168851, 84.81110254, 85.58211257,
                                      86.5073246, 87.43253662, 88.28064765, 88.74325366, 89.28296068, 89.59136469,
                                      89.8997687, 90.20817271, 90.36237471, 90.67077872, 90.97918273, 91.13338473,
                                      91.13338473, 91.44178874, 91.59599075, 91.75019275, 91.82729375, 92.05859676,
                                      92.13569776])
    data_list.append(data_patchaugnet)
    save_filepath = '/home/ericxhzou/place recognition recall on Data A.svg'
    draw_line_chart(data_list, title='Evaluation result on Data A', xlabel='N-Number of top database candidates',
                    ylabel='Recall @N(%)', save_filepath=save_filepath, yrange=[20, 100], ystep=10)


# draw rerank result: recall@top1~25 curve data A
def draw_rerank_recall_top1_25_data_A():
    data_list = []
    # Init(PatchAugNet)
    data_init = RecallPrecisionTopN()
    data_init.name = 'Init(PatchAugNet)'
    data_init.fig_color = 'blue'
    data_init.fig_marker = 'v'
    data_init.data = np.array([76.40709329, 80.33924441, 82.03546646, 83.73168851, 84.81110254, 85.58211257,
                               86.5073246, 87.43253662, 88.28064765, 88.74325366, 89.28296068, 89.59136469,
                               89.8997687, 90.20817271, 90.36237471, 90.67077872, 90.97918273, 91.13338473,
                               91.13338473, 91.44178874, 91.59599075, 91.75019275, 91.82729375, 92.05859676,
                               92.13569776])
    data_list.append(data_init)
    # Rerank-MS
    data_rerank_ms = RecallPrecisionTopN()
    data_rerank_ms.name = 'Rerank-MS'
    data_rerank_ms.fig_color = 'green'
    data_rerank_ms.fig_marker = 's'
    data_rerank_ms.data = np.array([73.70855821, 78.56592136, 81.03315343, 82.65227448, 84.27139553, 85.35080956,
                                    86.12181958, 87.12413261, 87.81804163, 88.43484965, 89.12875867, 89.51426369,
                                    90.1310717, 90.43947571, 90.67077872, 91.21048574, 91.36468774, 91.36468774,
                                    91.51888975, 91.67309175, 91.82729375, 91.82729375, 92.05859676, 92.13569776,
                                    92.13569776])
    data_list.append(data_rerank_ms)
    # Rerank-MS-Top25
    data_rerank_ms_top25 = RecallPrecisionTopN()
    data_rerank_ms_top25.name = 'Rerank-MS-Top25'
    data_rerank_ms_top25.fig_color = 'green'
    data_rerank_ms_top25.fig_marker = 'x'
    data_rerank_ms_top25.data = np.array([77.10100231, 80.57054742, 82.34387047, 83.88589052, 84.81110254, 85.65921357,
                                         85.96761758, 86.5844256, 87.35543562, 88.20354665, 89.05165767, 89.43716268,
                                         89.8997687, 90.20817271, 90.36237471, 90.43947571, 90.59367772, 91.13338473,
                                         91.36468774, 91.51888975, 91.59599075, 91.82729375, 91.90439476, 92.05859676,
                                         92.13569776])
    data_list.append(data_rerank_ms_top25)
    # Rerank-MS-Top25-Dist
    data_rerank_ms_top25_dist = RecallPrecisionTopN()
    data_rerank_ms_top25_dist.name = 'Rerank-MS-Top25-Dist'
    data_rerank_ms_top25_dist.fig_color = 'red'
    data_rerank_ms_top25_dist.fig_marker = 'd'
    data_rerank_ms_top25_dist.data = np.array([91.44178874, 91.82729375, 91.82729375, 91.90439476, 91.98149576, 92.13569776,
                                              92.13569776, 92.13569776, 92.13569776, 92.13569776, 92.13569776, 92.13569776,
                                              92.13569776, 92.13569776, 92.13569776, 92.13569776, 92.13569776, 92.13569776,
                                              92.13569776, 92.13569776, 92.13569776, 92.13569776, 92.13569776, 92.13569776,
                                              92.13569776])
    data_list.append(data_rerank_ms_top25_dist)
    # Rerank-MS-Top50
    data_rerank_ms_top50 = RecallPrecisionTopN()
    data_rerank_ms_top50.name = 'Rerank-MS-Top50'
    data_rerank_ms_top50.fig_color = 'green'
    data_rerank_ms_top50.fig_marker = '+'
    data_rerank_ms_top50.data = np.array([69.8535081, 74.09406322, 76.17579029, 78.18041635, 79.18272938, 80.49344641,
                                          81.49575944, 82.57517348, 83.96299152, 84.65690054, 85.50501157, 86.5073246,
                                          87.27833462, 87.89514264, 88.28064765, 88.51195066, 89.05165767, 89.74556669,
                                          90.0539707, 90.1310717, 90.51657672, 90.82498072, 90.97918273, 91.28758674,
                                          91.67309175])
    data_list.append(data_rerank_ms_top50)
    # Rerank-MS-Top50-Dist
    data_rerank_ms_top50_dist = RecallPrecisionTopN()
    data_rerank_ms_top50_dist.name = 'Rerank-MS-Top50-Dist'
    data_rerank_ms_top50_dist.fig_color = 'red'
    data_rerank_ms_top50_dist.fig_marker = '*'
    data_rerank_ms_top50_dist.data = np.array([93.4464148, 94.21742483, 94.60292984, 94.91133385, 95.06553585, 95.21973786,
                                               95.21973786, 95.21973786, 95.21973786, 95.29683886, 95.37393986, 95.37393986,
                                               95.37393986, 95.37393986, 95.37393986, 95.37393986, 95.37393986, 95.37393986,
                                               95.37393986, 95.37393986, 95.37393986, 95.37393986, 95.37393986, 95.37393986,
                                               95.37393986])
    data_list.append(data_rerank_ms_top50_dist)
    save_filepath = '/home/ericxhzou/rerank recall on Data A.svg'
    draw_line_chart(data_list, title='Evaluation result on Data A', xlabel='N-Number of top database candidates',
                    ylabel='Recall @N(%)', save_filepath=save_filepath, yrange=[60, 100], ystep=5)


# draw rerank result: precision@top1~25 curve data A
def draw_rerank_precision_top1_25_data_A():
    data_list = []
    # Init(PatchAugNet)
    data_init = RecallPrecisionTopN()
    data_init.name = 'Init(PatchAugNet)'
    data_init.fig_color = 'blue'
    data_init.fig_marker = 'v'
    data_init.data = np.array([76.40709329, 75.55898227, 75.09637625, 74.26754048, 73.73939861, 73.07890003,
                               72.59610089, 71.94487278, 71.28416003, 70.66306862, 70.03574683, 69.3459265,
                               68.53686021, 67.76627382, 66.79002827, 65.8924441, 64.90543789, 63.87818042,
                               62.70746257, 61.7000771, 60.63075963, 59.52547838, 58.39562871, 57.32138268,
                               56.18195837])
    data_list.append(data_init)
    # Rerank-MS
    data_rerank_ms = RecallPrecisionTopN()
    data_rerank_ms.name = 'Rerank-MS'
    data_rerank_ms.fig_color = 'green'
    data_rerank_ms.fig_marker = 's'
    data_rerank_ms.data = np.array([73.70855821, 73.2845027, 72.86044719, 72.45566692, 72.12027756, 71.89668466,
                                    71.29639828, 70.66306862, 70.18761244, 69.40632228, 68.7600757, 67.88100745,
                                    67.16683471, 66.44454235, 65.4690311, 64.68774094, 63.82602386, 62.89300094,
                                    62.05819097, 61.15651503, 60.29298381, 59.28716619, 58.26489223, 57.23143151,
                                    56.17887433])
    data_list.append(data_rerank_ms)
    # Rerank-MS-Top25
    data_rerank_ms_topk = RecallPrecisionTopN()
    data_rerank_ms_topk.name = 'Rerank-MS-Top25'
    data_rerank_ms_topk.fig_color = 'green'
    data_rerank_ms_topk.fig_marker = 'x'
    data_rerank_ms_topk.data = np.array([77.10100231, 76.5612953, 76.30429196, 75.80956052, 75.25057826, 74.68517091,
                                         73.93986122, 73.35196608, 72.68911163, 72.02004626, 71.24833532, 70.50886662,
                                         69.66372101, 68.80713735, 67.96196351, 66.98631457, 65.99392263, 64.90619378,
                                         63.75035507, 62.59444873, 61.41645556, 60.2193874, 58.82806476, 57.51413518,
                                         56.17887433])
    data_list.append(data_rerank_ms_topk)
    # Rerank-MS-Top25-Dist
    data_rerank_ms_topk_dist = RecallPrecisionTopN()
    data_rerank_ms_topk_dist.name = 'Rerank-MS-Top25-Dist'
    data_rerank_ms_topk_dist.fig_color = 'red'
    data_rerank_ms_topk_dist.fig_marker = 'd'
    data_rerank_ms_topk_dist.data = np.array([91.44178874, 90.90208173, 89.84836803, 89.09020817, 88.24980725, 87.12413261,
                                              86.15486287, 84.95566692, 83.62888718, 82.16653816, 80.74577697, 79.29838088,
                                              77.7652571, 76.142747, 74.35106656, 72.50385505, 70.6653363, 68.69270967,
                                              66.692367, 64.76869699, 62.86301722, 60.96586528, 59.22697865, 57.66833719,
                                              56.17887433])
    data_list.append(data_rerank_ms_topk_dist)
    # Rerank-MS-Top50
    data_rerank_ms_top50 = RecallPrecisionTopN()
    data_rerank_ms_top50.name = 'Rerank-MS-Top50'
    data_rerank_ms_top50.fig_color = 'green'
    data_rerank_ms_top50.fig_marker = '+'
    data_rerank_ms_top50.data = np.array([69.8535081, 69.9691596, 69.87920843, 69.91133385, 69.54510409, 69.10819841,
                                          68.47670448, 67.88743254, 67.48907736, 66.87740941, 66.23677017, 65.57440247,
                                          64.87159718, 64.24716378, 63.34104343, 62.6156515, 61.66266044, 60.75558982,
                                          59.91965264, 58.90516577, 57.90652421, 56.99516366, 56.09265529, 55.29105628,
                                          54.50424056])
    data_list.append(data_rerank_ms_top50)
    # Rerank-MS-Top50-Dist
    data_rerank_ms_top50_dist = RecallPrecisionTopN()
    data_rerank_ms_top50_dist.name = 'Rerank-MS-Top50-Dist'
    data_rerank_ms_top50_dist.fig_color = 'red'
    data_rerank_ms_top50_dist.fig_marker = '*'
    data_rerank_ms_top50_dist.data = np.array([93.4464148, 92.59830378, 91.80159342, 91.48033924, 91.02544333, 90.58082755,
                                               89.92179755, 89.2925983, 88.71755333, 88.00308404, 87.17319689, 86.39809818,
                                               85.48128818, 84.40907589, 83.34104343, 82.18003084, 80.95605243, 79.60678489,
                                               78.20070608, 76.6615266, 75.10739068, 73.50178734, 71.76091985, 70.04304806,
                                               68.26522745])
    data_list.append(data_rerank_ms_top50_dist)
    save_filepath = '/home/ericxhzou/rerank precision on Data A.svg'
    draw_line_chart(data_list, title='Evaluation result on Data A', xlabel='N-Number of top database candidates',
                    ylabel='Precision @N(%)', save_filepath=save_filepath, yrange=[50, 95], ystep=5)


# draw place recognition result: recall@top1~25 curve data B
def draw_pr_recall_top1_25_data_B():
    data_list = []
    # PointNetVLAD
    data_pointnetvlad = RecallPrecisionTopN()
    data_pointnetvlad.name = 'PointNetVLAD'
    data_pointnetvlad.fig_color = 'blue'
    data_pointnetvlad.fig_marker = 'v'
    data_pointnetvlad.data = np.array([58.47140567, 61.3575628, 63.38856227, 64.93853554, 66.54195617, 67.55745591,
                                       68.78674506, 69.69535008, 71.03153394, 71.7797969, 72.42116515, 72.9556387,
                                       73.6504543, 74.34526991, 74.77284874, 75.20042758, 75.78834848, 76.16247996,
                                       76.75040086, 77.17797969, 77.49866382, 78.08658471, 78.46071619, 78.67450561,
                                       78.94174238])
    data_list.append(data_pointnetvlad)
    # PPT-Net: original version
    data_pptnet = RecallPrecisionTopN()
    data_pptnet.name = 'PPT-Net'
    data_pptnet.fig_color = 'green'
    data_pptnet.fig_marker = 's'
    data_pptnet.data = np.array([45.80438268, 50.3474078, 53.87493319, 55.79903795, 58.09727418, 59.43345804,
                                 60.92998397, 62.15927312, 63.49545697, 64.35061464, 64.9919829, 65.74024586,
                                 66.54195617, 67.18332443, 67.66435061, 68.1453768, 68.78674506, 69.3212186,
                                 69.69535008, 69.9091395, 70.33671833, 70.55050775, 71.13842865, 71.61945484,
                                 72.15392838])
    data_list.append(data_pptnet)
    # PPT-Net with L2 Norm
    data_pptnet_w_l2norm = RecallPrecisionTopN()
    data_pptnet_w_l2norm.name = 'PPT-Net w/ L2 Norm'
    data_pptnet_w_l2norm.fig_color = 'green'
    data_pptnet_w_l2norm.fig_marker = 'x'
    data_pptnet_w_l2norm.data = np.array([50.88188135, 57.0283271, 60.39551042, 63.06787814, 64.72474613, 66.00748263,
                                          67.61090326, 68.62640299, 70.22982362, 71.61945484, 72.31427044, 73.27632282,
                                          74.13148049, 74.77284874, 75.414217, 75.62800641, 75.73490112, 75.84179583,
                                          76.42971673, 76.64350615, 76.91074292, 77.65900588, 77.8727953, 78.08658471,
                                          78.35382149])
    data_list.append(data_pptnet_w_l2norm)
    # Minkloc3D v2 without data augmentation (DA)
    data_minkloc3dv2_wo_DA = RecallPrecisionTopN()
    data_minkloc3dv2_wo_DA.name = 'Minkloc3D v2 w/o DA'
    data_minkloc3dv2_wo_DA.fig_color = 'chocolate'
    data_minkloc3dv2_wo_DA.fig_marker = '^'
    data_minkloc3dv2_wo_DA.data = np.array([44.8423303, 50.66809193, 53.50080171, 56.17316943, 57.99037948, 59.59380011,
                                            61.19722074, 61.99893105, 63.60235168, 64.45750935, 65.25921967, 65.6867985,
                                            66.16782469, 66.75574559, 67.4505612, 68.19882416, 68.7332977, 69.10742918,
                                            69.64190273, 69.96258685, 70.6039551, 70.92463923, 71.45911277, 71.94013896,
                                            72.20737573])
    data_list.append(data_minkloc3dv2_wo_DA)
    # PatchAugNet(ours)
    data_patchaugnet = RecallPrecisionTopN()
    data_patchaugnet.name = 'PatchAugNet(ours)'
    data_patchaugnet.fig_color = 'red'
    data_patchaugnet.fig_marker = 'd'
    data_patchaugnet.data = np.array([60.34206307, 65.04543025, 67.82469268, 69.58845537, 71.61945484, 72.74184928,
                                      73.59700695, 74.77284874, 75.5211117, 76.48316408, 76.91074292, 77.49866382,
                                      77.92624265, 78.19347942, 78.62105826, 78.88829503, 79.04863709, 79.52966328,
                                      79.95724212, 80.27792624, 80.81239979, 81.07963656, 81.29342598, 81.66755746,
                                      82.04168894])
    data_list.append(data_patchaugnet)
    save_filepath = '/home/ericxhzou/place recognition recall on Data B.svg'
    draw_line_chart(data_list, title='Evaluation result on Data B', xlabel='N-Number of top database candidates',
                    ylabel='Recall @N(%)', save_filepath=save_filepath, yrange=[40, 90], ystep=5)


# draw place recognition result: precision@top1~25 curve data B
def draw_pr_precision_top1_25_data_B():
    data_list = []
    # PointNetVLAD
    data_pointnetvlad = RecallPrecisionTopN()
    data_pointnetvlad.name = 'PointNetVLAD'
    data_pointnetvlad.fig_color = 'blue'
    data_pointnetvlad.fig_marker = 'v'
    data_pointnetvlad.data = np.array([58.47140567, 59.05932656, 59.04151078, 58.51149118, 58.09727418, 57.16194548,
                                       56.34878216, 55.48503474, 54.6825821, 53.84286478, 52.88372771, 51.96419027,
                                       51.04222341, 50.13743605, 49.24995546, 48.37319615, 47.65932028, 46.86739118,
                                       46.20101831, 45.40887226, 44.71761981, 43.96530781, 43.29003323, 42.60422234,
                                       41.97327632])
    data_list.append(data_pointnetvlad)
    # PPT-Net: original version
    data_pptnet = RecallPrecisionTopN()
    data_pptnet.name = 'PPT-Net'
    data_pptnet.fig_color = 'green'
    data_pptnet.fig_marker = 's'
    data_pptnet.data = np.array([45.80438268, 44.94922501, 44.7354356, 44.30785676, 44.0726884, 43.60413326,
                                 43.24654501, 42.90486371, 42.49658531, 42.04703367, 41.7375249, 41.38161411,
                                 40.97767545, 40.57799496, 40.09976839, 39.5777659, 39.13603924, 38.67509947,
                                 38.28799685, 37.8727953, 37.41314805, 36.98071036, 36.62073293, 36.27739177,
                                 35.97220738])
    data_list.append(data_pptnet)
    # PPT-Net with L2 Norm
    data_pptnet_w_l2norm = RecallPrecisionTopN()
    data_pptnet_w_l2norm.name = 'PPT-Net w/ L2 Norm'
    data_pptnet_w_l2norm.fig_color = 'green'
    data_pptnet_w_l2norm.fig_marker = 'x'
    data_pptnet_w_l2norm.data = np.array([50.88188135, 50.80171032, 50.63246036, 50.32068413, 49.98396579, 49.66150009,
                                          49.14866, 48.76402993, 48.44111883, 48.24692678, 47.84509985, 47.35435596,
                                          46.92677712, 46.45338627, 46.04667736, 45.50040086, 45.13786273, 44.73840489,
                                          44.37536921, 43.8909674, 43.4552442, 43.05670278, 42.63704599, 42.22340994,
                                          41.77872795])
    data_list.append(data_pptnet_w_l2norm)
    # PatchAugNet(ours)
    data_patchaugnet = RecallPrecisionTopN()
    data_patchaugnet.name = 'PatchAugNet(ours)'
    data_patchaugnet.fig_color = 'red'
    data_patchaugnet.fig_marker = 'd'
    data_patchaugnet.data = np.array([60.34206307, 60.12827365, 59.9501158, 59.23303046, 58.78140032, 58.23980046,
                                      57.83767275, 57.3757349, 56.83829206, 56.30144308, 55.76988485, 55.13539996,
                                      54.61497348, 54.08490494, 53.51505434, 52.93626403, 52.32181595, 51.72219253,
                                      51.22788264, 50.75093533, 50.21760709, 49.60157427, 49.08326168, 48.60146089,
                                      48.12185997])
    data_list.append(data_patchaugnet)
    save_filepath = '/home/ericxhzou/place recognition precision on Data B.svg'
    draw_line_chart(data_list, title='Evaluation result on Data B', xlabel='N-Number of top database candidates',
                    ylabel='Precision @N(%)', save_filepath=save_filepath, yrange=[35, 65], ystep=5)


# draw rerank result: recall@top1~25 curve data B
def draw_rerank_recall_top1_25_data_B():
    data_list = []
    # Init(PatchAugNet)
    data_init = RecallPrecisionTopN()
    data_init.name = 'Init(PatchAugNet)'
    data_init.fig_color = 'blue'
    data_init.fig_marker = 'v'
    data_init.data = np.array([60.34206307, 65.04543025, 67.82469268, 69.58845537, 71.61945484, 72.74184928,
                               73.59700695, 74.77284874, 75.5211117, 76.48316408, 76.91074292, 77.49866382,
                               77.92624265, 78.19347942, 78.62105826, 78.88829503, 79.04863709, 79.52966328,
                               79.95724212, 80.27792624, 80.81239979, 81.07963656, 81.29342598, 81.66755746,
                               82.04168894])
    data_list.append(data_init)
    # Rerank-MS
    data_rerank_ms = RecallPrecisionTopN()
    data_rerank_ms.name = 'Rerank-MS'
    data_rerank_ms.fig_color = 'green'
    data_rerank_ms.fig_marker = 's'
    data_rerank_ms.data = np.array([56.33351149, 61.67824693, 64.77819348, 67.55745591, 69.16087654, 70.97808658,
                                    72.3677178, 73.27632282, 74.71940139, 75.73490112, 76.37626937, 77.07108498,
                                    77.76590059, 78.67450561, 79.36932122, 79.69000534, 80.17103153, 80.49171566,
                                    80.81239979, 81.07963656, 81.34687333, 81.6141101, 81.82789952, 81.98824158,
                                    82.04168894])
    data_list.append(data_rerank_ms)
    # Rerank-MS-Top25
    data_rerank_ms_topk = RecallPrecisionTopN()
    data_rerank_ms_topk.name = 'Rerank-MS-Top25'
    data_rerank_ms_topk.fig_color = 'green'
    data_rerank_ms_topk.fig_marker = 'x'
    data_rerank_ms_topk.data = np.array([59.11277392, 64.61785142, 67.93158739, 69.9091395, 71.67290219, 72.63495457,
                                         74.07803314, 75.94869054, 76.42971673, 76.96419027, 77.71245323, 78.19347942,
                                         78.67450561, 79.1555318, 79.47621593, 79.95724212, 80.27792624, 80.65205772,
                                         80.81239979, 81.07963656, 81.18653127, 81.50721539, 81.6141101, 81.88134687,
                                         82.04168894])
    data_list.append(data_rerank_ms_topk)
    # Rerank-MS-Top25-Dist
    data_rerank_ms_topk_dist = RecallPrecisionTopN()
    data_rerank_ms_topk_dist.name = 'Rerank-MS-Top25-Dist'
    data_rerank_ms_topk_dist.fig_color = 'red'
    data_rerank_ms_topk_dist.fig_marker = 'd'
    data_rerank_ms_topk_dist.data = np.array([81.82789952, 82.04168894, 82.04168894, 82.04168894, 82.04168894, 82.04168894,
                                              82.04168894, 82.04168894, 82.04168894, 82.04168894, 82.04168894, 82.04168894,
                                              82.04168894, 82.04168894, 82.04168894, 82.04168894, 82.04168894, 82.04168894,
                                              82.04168894, 82.04168894, 82.04168894, 82.04168894, 82.04168894, 82.04168894,
                                              82.04168894])
    data_list.append(data_rerank_ms_topk_dist)
    # Rerank-MS-Top50
    data_rerank_ms_top50 = RecallPrecisionTopN()
    data_rerank_ms_top50.name = 'Rerank-MS-Top50'
    data_rerank_ms_top50.fig_color = 'green'
    data_rerank_ms_top50.fig_marker = '+'
    data_rerank_ms_top50.data = np.array([53.71459113, 60.181721, 62.90753608, 65.63335115, 67.71779797, 69.9091395,
                                          71.03153394, 72.31427044, 73.27632282, 74.39871726, 75.68145377, 76.32282202,
                                          76.75040086, 77.60555852, 77.92624265, 78.40726884, 78.78140032, 79.52966328,
                                          80.06413683, 80.49171566, 80.91929449, 81.18653127, 81.56066275, 81.82789952,
                                          82.30892571])
    data_list.append(data_rerank_ms_top50)
    # Rerank-MS-Top50-Dist
    data_rerank_ms_top50_dist = RecallPrecisionTopN()
    data_rerank_ms_top50_dist.name = 'Rerank-MS-Top50-Dist'
    data_rerank_ms_top50_dist.fig_color = 'red'
    data_rerank_ms_top50_dist.fig_marker = '*'
    data_rerank_ms_top50_dist.data = np.array([86.26402993, 86.42437199, 86.47781935, 86.5312667, 86.5312667, 86.5312667,
                                               86.5312667, 86.5312667, 86.5312667, 86.5312667, 86.5312667, 86.5312667,
                                               86.5312667, 86.5312667, 86.5312667, 86.5312667, 86.5312667, 86.5312667,
                                               86.5312667, 86.5312667, 86.5312667, 86.5312667, 86.5312667, 86.5312667,
                                               86.5312667])
    data_list.append(data_rerank_ms_top50_dist)
    save_filepath = '/home/ericxhzou/rerank recall on Data B.svg'
    draw_line_chart(data_list, title='Evaluation result on Data B', xlabel='N-Number of top database candidates',
                    ylabel='Recall @N(%)', save_filepath=save_filepath, yrange=[50, 90], ystep=5)


# draw rerank result: precision@top1~25 curve data B
def draw_rerank_precision_top1_25_data_B():
    data_list = []
    # Init(PatchAugNet)
    data_init = RecallPrecisionTopN()
    data_init.name = 'Init(PatchAugNet)'
    data_init.fig_color = 'blue'
    data_init.fig_marker = 'v'
    data_init.data = np.array([60.34206307, 60.12827365, 59.9501158, 59.23303046, 58.78140032, 58.23980046,
                               57.83767275, 57.3757349, 56.83829206, 56.30144308, 55.76988485, 55.13539996,
                               54.61497348, 54.08490494, 53.51505434, 52.93626403, 52.32181595, 51.72219253,
                               51.22788264, 50.75093533, 50.21760709, 49.60157427, 49.08326168, 48.60146089,
                               48.12185997])
    data_list.append(data_init)
    # Rerank-MS
    data_rerank_ms = RecallPrecisionTopN()
    data_rerank_ms.name = 'Rerank-MS'
    data_rerank_ms.fig_color = 'green'
    data_rerank_ms.fig_marker = 's'
    data_rerank_ms.data = np.array([56.33351149, 57.0283271, 57.45590593, 57.56280064, 57.43452699, 57.10849813,
                                    56.83744369, 56.42704436, 56.2028624, 55.97006948, 55.67270784, 55.22447889,
                                    54.76298154, 54.32160037, 53.9070016, 53.35716195, 52.88144119, 52.48233268,
                                    51.9311373, 51.37626937, 50.77753175, 50.17734804, 49.56893547, 48.8820595,
                                    48.11758418])
    data_list.append(data_rerank_ms)
    # Rerank-MS-Top25
    data_rerank_ms_topk = RecallPrecisionTopN()
    data_rerank_ms_topk.name = 'Rerank-MS-Top25'
    data_rerank_ms_topk.fig_color = 'green'
    data_rerank_ms_topk.fig_marker = 'x'
    data_rerank_ms_topk.data = np.array([59.11277392, 59.19294495, 59.18403706, 59.15285943, 58.72795297, 58.29324782,
                                         57.86057876, 57.46258685, 57.09365164, 56.63281668, 56.20718138, 55.57634064,
                                         54.95621428, 54.38268306, 53.79654374, 53.23356494, 52.648788, 52.14086347,
                                         51.64983544, 51.16515232, 50.6400957, 50.04372965, 49.4178886, 48.79743453,
                                         48.11758418])
    data_list.append(data_rerank_ms_topk)
    # Rerank-MS-Top25-Dist
    data_rerank_ms_topk_dist = RecallPrecisionTopN()
    data_rerank_ms_topk_dist.name = 'Rerank-MS-Top25-Dist'
    data_rerank_ms_topk_dist.fig_color = 'red'
    data_rerank_ms_topk_dist.fig_marker = 'd'
    data_rerank_ms_topk_dist.data = np.array([81.82789952, 80.19775521, 78.42508462, 77.07108498, 75.73490112, 74.36308569,
                                              73.09307475, 71.72634955, 70.46736742, 69.13949759, 67.84412808, 66.55531801,
                                              65.21399498, 63.80087043, 62.28754677, 60.78968466, 59.2385324, 57.72017341,
                                              56.23786886, 54.79422769, 53.41681301, 52.03099947, 50.71456789, 49.39649029,
                                              48.11758418])
    data_list.append(data_rerank_ms_topk_dist)
    # Rerank-MS-Top50
    data_rerank_ms_top50 = RecallPrecisionTopN()
    data_rerank_ms_top50.name = 'Rerank-MS-Top50'
    data_rerank_ms_top50.fig_color = 'green'
    data_rerank_ms_top50.fig_marker = '+'
    data_rerank_ms_top50.data = np.array([53.71459113, 54.00855158, 53.66114377, 53.59433458, 53.50080171, 53.53643328,
                                          53.46262503, 53.18011758, 52.84161767, 52.66702298, 52.41241922, 52.03990736,
                                          51.84393373, 51.56142628, 51.22038126, 50.85515767, 50.4637344, 50.1128333,
                                          49.73979577, 49.37466595, 49.04176529, 48.64681016, 48.29317036, 47.83538215,
                                          47.43773383])
    data_list.append(data_rerank_ms_top50)
    # Rerank-MS-Top50-Dist
    data_rerank_ms_top50_dist = RecallPrecisionTopN()
    data_rerank_ms_top50_dist.name = 'Rerank-MS-Top50-Dist'
    data_rerank_ms_top50_dist.fig_color = 'red'
    data_rerank_ms_top50_dist.fig_marker = '*'
    data_rerank_ms_top50_dist.data = np.array([86.26402993, 84.58043827, 83.2531623, 82.13522181, 81.04756815, 80.08195261,
                                               79.29296786, 78.32041689, 77.35019894, 76.34420096, 75.35105194, 74.36308569,
                                               73.31332484, 72.26082309, 71.18474969, 70.11290754, 68.97538278, 67.81281549,
                                               66.61509466, 65.42490647, 64.29207707, 63.14319032, 62.02216903, 60.90994121,
                                               59.8161411])
    data_list.append(data_rerank_ms_top50_dist)
    save_filepath = '/home/ericxhzou/rerank precision on Data B.svg'
    draw_line_chart(data_list, title='Evaluation result on Data B', xlabel='N-Number of top database candidates',
                    ylabel='Precision @N(%)', save_filepath=save_filepath, yrange=[45, 90], ystep=5)


# draw place recognition result: recall@top1~25 curve data oxford robot car
def draw_pr_recall_top1_25_data_oxford():
    data_list = []
    # PointNetVLAD
    data_pointnetvlad = RecallPrecisionTopN()
    data_pointnetvlad.name = 'PointNetVLAD'
    data_pointnetvlad.fig_color = 'blue'
    data_pointnetvlad.fig_marker = 'v'
    data_pointnetvlad.data = np.array([52.93683337, 60.06589687, 64.19709941, 66.95357798, 69.37836879, 71.30786494,
                                       72.91697777, 74.16592682, 75.27896153, 76.26519752, 77.13148821, 78.03735056,
                                       78.81021729, 79.50317657, 80.20806879, 80.82549182, 81.427387, 81.96764716,
                                       82.51093906, 82.99700888, 83.51473927, 83.93938106, 84.33437637, 84.73991069,
                                       85.15043913])
    data_list.append(data_pointnetvlad)
    # PPT-Net: original version
    data_pptnet = RecallPrecisionTopN()
    data_pptnet.name = 'PPT-Net'
    data_pptnet.fig_color = 'green'
    data_pptnet.fig_marker = 's'
    data_pptnet.data = np.array([55.72751476, 63.91275896, 68.37450691, 71.46653847, 73.66131674, 75.43677553,
                                 76.89335085, 78.25930648, 79.38488443, 80.40820953, 81.26614031, 82.00446275,
                                 82.68946702, 83.32594854, 83.91613368, 84.51128707, 85.05947257, 85.52374353,
                                 85.98790536, 86.42615253, 86.89045545, 87.25891359, 87.62775329, 87.98843178,
                                 88.34560823])
    data_list.append(data_pptnet)
    # PPT-Net with L2 Norm
    data_pptnet_w_l2norm = RecallPrecisionTopN()
    data_pptnet_w_l2norm.name = 'PPT-Net w/ L2 Norm'
    data_pptnet_w_l2norm.fig_color = 'green'
    data_pptnet_w_l2norm.fig_marker = 'x'
    data_pptnet_w_l2norm.data = np.array([62.17497597, 70.42821061, 74.85053322, 77.64278325, 79.64194327, 81.21107955,
                                          82.54333987, 83.67252861, 84.614989, 85.47669316, 86.1993298, 86.86210174,
                                          87.46021736, 87.98490958, 88.48564375, 88.91461131, 89.3549708, 89.70476652,
                                          90.02349748, 90.31201304, 90.61925366, 90.90898528, 91.1912091, 91.4707256,
                                          91.73927027])
    data_list.append(data_pptnet_w_l2norm)
    # Minkloc3D v2 without data augmentation (DA)
    data_minkloc3dv2_wo_DA = RecallPrecisionTopN()
    data_minkloc3dv2_wo_DA.name = 'Minkloc3D v2 w/o DA'
    data_minkloc3dv2_wo_DA.fig_color = 'chocolate'
    data_minkloc3dv2_wo_DA.fig_marker = '^'
    data_minkloc3dv2_wo_DA.data = np.array([51.52148355, 60.79210571, 65.7770681, 69.04727968, 71.59268472, 73.54375431,
                                            75.16394094, 76.58078662, 77.88189625, 78.95046776, 79.91668996, 80.76972743,
                                            81.58294542, 82.29022219, 82.98848911, 83.57635257, 84.1562382, 84.6669645,
                                            85.1468956, 85.6145648, 86.04418271, 86.433871, 86.81486502, 87.14421482,
                                            87.51645134])
    data_list.append(data_minkloc3dv2_wo_DA)
    # PatchAugNet(ours)
    data_patchaugnet = RecallPrecisionTopN()
    data_patchaugnet.name = 'PatchAugNet(ours)'
    data_patchaugnet.fig_color = 'red'
    data_patchaugnet.fig_marker = 'd'
    data_patchaugnet.data = np.array([66.26022609, 74.04031774, 77.84566526, 80.26742516, 82.1353671, 83.57325124,
                                      84.78997831, 85.71907097, 86.54313421, 87.28007203, 87.92625949, 88.54039605,
                                      89.05478348, 89.51170141, 89.91797785, 90.3493201, 90.7402002, 91.05451967,
                                      91.39615608, 91.66882308, 91.95906601, 92.23529016, 92.47803091, 92.70984527,
                                      92.93206345])
    data_list.append(data_patchaugnet)
    save_filepath = '/home/ericxhzou/place recognition average recall on Data Oxford.svg'
    draw_line_chart(data_list, title='Evaluation result on Data Oxford', xlabel='N-Number of top database candidates',
                    ylabel='Recall @N(%)', save_filepath=save_filepath, yrange=[50, 100], ystep=5)


# draw place recognition result: recall@top1~25 curve data university
def draw_pr_recall_top1_25_data_university():
    data_list = []
    # PointNetVLAD
    data_pointnetvlad = RecallPrecisionTopN()
    data_pointnetvlad.name = 'PointNetVLAD'
    data_pointnetvlad.fig_color = 'blue'
    data_pointnetvlad.fig_marker = 'v'
    data_pointnetvlad.data = np.array([65.57525808, 72.34265734, 76.66167166, 78.66133866, 80.46786547, 82.20695971,
                                       83.81868132, 85.49367299, 86.33283383, 87.36513487, 88.2034632, 88.91275391,
                                       89.62204462, 90.00999001, 90.78255078, 91.68581419, 92.2027972, 92.84715285,
                                       93.23343323, 93.68464868, 94.00682651, 94.2007992, 94.32900433, 94.52214452,
                                       94.97419247])
    data_list.append(data_pointnetvlad)
    # PPT-Net: original version
    data_pptnet = RecallPrecisionTopN()
    data_pptnet.name = 'PPT-Net'
    data_pptnet.fig_color = 'green'
    data_pptnet.fig_marker = 's'
    data_pptnet.data = np.array([61.78404928, 70.02747253, 73.12104562, 75.56943057, 76.92057942, 78.47319347,
                                 79.76023976, 81.30785881, 81.82400932, 82.53246753, 83.43406593, 84.20829171,
                                 84.85347985, 85.23976024, 85.56360306, 86.01398601, 86.65917416, 87.17449217,
                                 87.56160506, 87.75474525, 88.01448551, 88.27422577, 88.66050616, 88.98434898,
                                 89.36979687])
    data_list.append(data_pptnet)
    # PPT-Net with L2 Norm
    data_pptnet_w_l2norm = RecallPrecisionTopN()
    data_pptnet_w_l2norm.name = 'PPT-Net w/ L2 Norm'
    data_pptnet_w_l2norm.fig_color = 'green'
    data_pptnet_w_l2norm.fig_marker = 'x'
    data_pptnet_w_l2norm.data = np.array([77.5008325, 83.37079587, 86.53096903, 88.33583084, 90.46203796, 91.75074925,
                                          92.45920746, 93.2958708, 93.81202131, 94.26240426, 94.71361971, 95.16233766,
                                          95.54945055, 95.80752581, 96.12887113, 96.38611389, 96.51431901, 96.57925408,
                                          96.77322677, 96.9022644, 96.96636697, 97.22527473, 97.41924742, 97.74225774,
                                          97.93539794])
    data_list.append(data_pptnet_w_l2norm)
    # Minkloc3D v2 without data augmentation (DA)
    data_minkloc3dv2_wo_DA = RecallPrecisionTopN()
    data_minkloc3dv2_wo_DA.name = 'Minkloc3D v2 w/o DA'
    data_minkloc3dv2_wo_DA.fig_color = 'chocolate'
    data_minkloc3dv2_wo_DA.fig_marker = '^'
    data_minkloc3dv2_wo_DA.data = np.array([70.54945055, 80.99317349, 84.5987346, 86.98551449, 88.01531802, 89.24075924,
                                            90.01415251, 90.78671329, 91.68997669, 91.94721945, 92.46420246, 92.78721279,
                                            93.17432567, 93.75374625, 94.07509158, 94.3981019, 94.65534466, 94.97835498,
                                            95.17149517, 95.49284049, 95.62104562, 95.94405594, 96.00899101, 96.07309357,
                                            96.33116883])
    data_list.append(data_minkloc3dv2_wo_DA)
    # PatchAugNet(ours)
    data_patchaugnet = RecallPrecisionTopN()
    data_patchaugnet.name = 'PatchAugNet(ours)'
    data_patchaugnet.fig_color = 'red'
    data_patchaugnet.fig_marker = 'd'
    data_patchaugnet.data = np.array([82.07042957, 87.61988012, 90.77339327, 92.25857476, 93.54811855, 94.25824176,
                                      94.64368964, 95.0957376, 95.54861805, 95.93656344, 96.06476856, 96.45271395,
                                      96.71078921, 96.9047619, 97.22693973, 97.42091242, 97.61571762, 97.74475524,
                                      97.80885781, 97.93706294, 98.0011655, 98.0011655, 98.12937063, 98.12937063,
                                      98.25757576])
    data_list.append(data_patchaugnet)
    save_filepath = '/home/ericxhzou/place recognition average recall on Data University.svg'
    draw_line_chart(data_list, title='Evaluation result on Data University', xlabel='N-Number of top database candidates',
                    ylabel='Recall @N(%)', save_filepath=save_filepath, yrange=[60, 100], ystep=5)


# draw place recognition result: recall@top1~25 curve data residential
def draw_pr_recall_top1_25_data_residential():
    data_list = []
    # PointNetVLAD
    data_pointnetvlad = RecallPrecisionTopN()
    data_pointnetvlad.name = 'PointNetVLAD'
    data_pointnetvlad.fig_color = 'blue'
    data_pointnetvlad.fig_marker = 'v'
    data_pointnetvlad.data = np.array([58.61441441, 67.41711712, 70.57207207, 72.98828829, 75.0018018, 76.60990991,
                                       78.15225225, 79.6981982, 80.44054054, 81.44864865, 82.65945946, 83.33063063,
                                       83.93333333, 84.80720721, 85.34594595, 85.95135135, 86.55495495, 87.09279279,
                                       87.63153153, 88.50720721, 88.77657658, 89.31351351, 89.58108108, 89.98288288,
                                       90.38648649])
    data_list.append(data_pointnetvlad)
    # PPT-Net: original version
    data_pptnet = RecallPrecisionTopN()
    data_pptnet.name = 'PPT-Net'
    data_pptnet.fig_color = 'green'
    data_pptnet.fig_marker = 's'
    data_pptnet.data = np.array([49.78018018, 57.84954955, 63.16486486, 66.25765766, 68.67567568, 70.29189189,
                                 71.7027027, 72.77837838, 73.45315315, 74.32792793, 74.86576577, 75.73873874,
                                 76.41081081, 76.88108108, 77.28468468, 78.15675676, 78.62612613, 79.0963964,
                                 79.5, 79.83603604, 80.17117117, 80.50630631, 80.77657658, 81.11261261,
                                 81.51621622])
    data_list.append(data_pptnet)
    # PPT-Net with L2 Norm
    data_pptnet_w_l2norm = RecallPrecisionTopN()
    data_pptnet_w_l2norm.name = 'PPT-Net w/ L2 Norm'
    data_pptnet_w_l2norm.fig_color = 'green'
    data_pptnet_w_l2norm.fig_marker = 'x'
    data_pptnet_w_l2norm.data = np.array([67.2, 76.68288288, 80.65045045, 83.3972973, 85.01081081, 86.42252252,
                                          87.63513514, 88.50630631, 89.31081081, 90.11711712, 91.05495495, 91.39009009,
                                          92.1981982, 92.6027027, 93.07297297, 93.67837838, 93.94864865, 94.21801802,
                                          94.82252252, 95.09279279, 95.42972973, 95.56396396, 95.6981982, 95.9009009,
                                          96.03423423])
    data_list.append(data_pptnet_w_l2norm)
    # Minkloc3D v2 without data augmentation (DA)
    data_minkloc3dv2_wo_DA = RecallPrecisionTopN()
    data_minkloc3dv2_wo_DA.name = 'Minkloc3D v2 w/o DA'
    data_minkloc3dv2_wo_DA.fig_color = 'chocolate'
    data_minkloc3dv2_wo_DA.fig_marker = '^'
    data_minkloc3dv2_wo_DA.data = np.array([59.42252252, 71.78288288, 76.82072072, 80.44594595, 82.86396396, 84.88378378,
                                            86.29459459, 88.37657658, 89.31801802, 90.59369369, 91.13243243, 91.73783784,
                                            92.07297297, 92.47477477, 93.21351351, 93.48198198, 93.68378378, 93.88648649,
                                            93.95405405, 94.28828829, 94.55675676, 94.55675676, 94.75765766, 94.96036036,
                                            95.36126126])
    data_list.append(data_minkloc3dv2_wo_DA)
    # PatchAugNet(ours)
    data_patchaugnet = RecallPrecisionTopN()
    data_patchaugnet.name = 'PatchAugNet(ours)'
    data_patchaugnet.fig_color = 'red'
    data_patchaugnet.fig_marker = 'd'
    data_patchaugnet.data = np.array([73.52792793, 82.86846847, 86.62972973, 88.24144144, 90.19189189, 91.53603604,
                                      92.14144144, 92.74594595, 93.35225225, 93.88828829, 94.56036036, 95.02882883,
                                      95.2990991, 95.56756757, 95.83693694, 96.17297297, 96.23963964, 96.30720721,
                                      96.50810811, 96.70990991, 96.91171171, 96.97837838, 97.04504505, 97.24684685,
                                      97.24684685])
    data_list.append(data_patchaugnet)
    save_filepath = '/home/ericxhzou/place recognition average recall on Data Residential.svg'
    draw_line_chart(data_list, title='Evaluation result on Data Residential', xlabel='N-Number of top database candidates',
                    ylabel='Recall @N(%)', save_filepath=save_filepath, yrange=[45, 100], ystep=5)


# draw place recognition result: recall@top1~25 curve data business
def draw_pr_recall_top1_25_data_business():
    data_list = []
    # PointNetVLAD
    data_pointnetvlad = RecallPrecisionTopN()
    data_pointnetvlad.name = 'PointNetVLAD'
    data_pointnetvlad.fig_color = 'blue'
    data_pointnetvlad.fig_marker = 'v'
    data_pointnetvlad.data = np.array([60.29056461, 67.24636543, 70.70276719, 73.80408385, 75.82052889, 77.38572022,
                                       78.94926822, 80.18224013, 80.93932626, 82.02241357, 83.08088924, 83.88760106,
                                       84.82133566, 85.42613752, 86.15708534, 86.56075659, 87.11480217, 87.66910412,
                                       88.02252284, 88.52580828, 89.00472567, 89.38301748, 89.88681307, 90.24023435,
                                       90.69504389])
    data_list.append(data_pointnetvlad)
    # PPT-Net: original version
    data_pptnet = RecallPrecisionTopN()
    data_pptnet.name = 'PPT-Net'
    data_pptnet.fig_color = 'green'
    data_pptnet.fig_marker = 's'
    data_pptnet.data = np.array([46.39857817, 53.69194625, 58.43385161, 61.28480294, 63.47841429, 65.84930478,
                                 67.31247078, 69.20113932, 70.31000337, 71.14145884, 72.12443457, 73.03253596,
                                 73.7129749, 74.41816134, 74.97208512, 75.70354441, 76.25860649, 76.71291353,
                                 77.1165899, 77.69690708, 78.15045016, 78.68039044, 79.10994874, 79.51362255,
                                 79.84191309])
    data_list.append(data_pptnet)
    # PPT-Net with L2 Norm
    data_pptnet_w_l2norm = RecallPrecisionTopN()
    data_pptnet_w_l2norm.name = 'PPT-Net w/ L2 Norm'
    data_pptnet_w_l2norm.fig_color = 'green'
    data_pptnet_w_l2norm.fig_marker = 'x'
    data_pptnet_w_l2norm.data = np.array([63.83594252, 71.7067167, 76.50077296, 79.07266843, 81.01551197, 82.9570916,
                                          84.42139972, 85.50348075, 86.36019257, 87.36791453, 87.97462762, 88.402158,
                                          89.1589929, 89.9656996, 90.77228197, 91.22696843, 91.73178052, 92.15969035,
                                          92.46184005, 92.86512805, 93.1677892, 93.77234621, 93.94898956, 94.12525732,
                                          94.50393113])
    data_list.append(data_pptnet_w_l2norm)
    # Minkloc3D v2 without data augmentation (DA)
    data_minkloc3dv2_wo_DA = RecallPrecisionTopN()
    data_minkloc3dv2_wo_DA.name = 'Minkloc3D v2 w/o DA'
    data_minkloc3dv2_wo_DA.fig_color = 'chocolate'
    data_minkloc3dv2_wo_DA.fig_marker = '^'
    data_minkloc3dv2_wo_DA.data = np.array([61.89479996, 72.1850699, 77.91158539, 81.14134123, 83.43659209, 85.25253973,
                                            86.4135535, 87.49690354, 88.45563303, 89.43911127, 90.2211974, 90.92600184,
                                            91.48055505, 92.08714124, 92.48967164, 93.12049388, 93.42340882, 93.92745182,
                                            94.17998221, 94.60801766, 94.8095341, 95.03668635, 95.23858479, 95.61598701,
                                            95.79187022])
    data_list.append(data_minkloc3dv2_wo_DA)
    # PatchAugNet(ours)
    data_patchaugnet = RecallPrecisionTopN()
    data_patchaugnet.name = 'PatchAugNet(ours)'
    data_patchaugnet.fig_color = 'red'
    data_patchaugnet.fig_marker = 'd'
    data_patchaugnet.data = np.array([75.57442947, 82.83225309, 86.03370689, 88.15384491, 89.74186994, 90.85047635,
                                      91.80794192, 92.74091508, 93.57224363, 94.35369781, 94.75609619, 95.0341342,
                                      95.48818616, 95.81597036, 96.01723557, 96.37077863, 96.62292832, 96.97583431,
                                      97.25324164, 97.35374542, 97.45462863, 97.60640273, 97.75779483, 97.78317554,
                                      97.88418821])
    data_list.append(data_patchaugnet)
    save_filepath = '/home/ericxhzou/place recognition average recall on Data Business.svg'
    draw_line_chart(data_list, title='Evaluation result on Data Business', xlabel='N-Number of top database candidates',
                    ylabel='Recall @N(%)', save_filepath=save_filepath, yrange=[45, 100], ystep=5)


# draw place recognition result: success / failure cases
def draw_pr_cases(case_path, num_query, num_top, show_fig=False):
    if not os.path.exists(case_path):
        print('Invalid case path: ', case_path)
    else:
        for i in range(num_query):
            # PatchAugNet success case-query
            pc_file = os.path.join(case_path, 'case{}-query.bin'.format(i))
            svg_file = os.path.join(case_path, 'case{}-query.svg'.format(i))
            draw_pc(pc_file, svg_file, show_fig)
            # PatchAugNet success case-top
            for j in range(num_top):
                pc_file = os.path.join(case_path, 'case{}-top{}.bin'.format(i, j))
                svg_file = os.path.join(case_path, 'case{}-top{}.svg'.format(i, j))
                draw_pc(pc_file, svg_file, show_fig)


# draw ablation result of patch feature augmentation: recall@top1~25 curve data A
def draw_ablation_pfa_recall_top1_25_data_A():
    data_list = []
    # PatchAugNet: w/o recon, w/o align
    data_patchaugnet_wo_recon_wo_align = RecallPrecisionTopN()
    data_patchaugnet_wo_recon_wo_align.name = 'w/o recon, w/o align'
    data_patchaugnet_wo_recon_wo_align.fig_color = 'blue'
    data_patchaugnet_wo_recon_wo_align.fig_marker = 'v'
    data_patchaugnet_wo_recon_wo_align.data = np.array(
        [67.84888204, 73.93986122, 77.02390131, 78.87432537, 80.49344641, 81.80416345,
         82.88357749, 83.3461835, 84.11719352, 84.88820355, 85.58211257, 86.27602159,
         86.7386276, 87.27833462, 87.66383963, 87.97224364, 88.35774865, 88.82035466,
         89.05165767, 89.12875867, 89.36006168, 89.36006168, 89.59136469, 89.8997687,
         90.0539707])
    data_list.append(data_patchaugnet_wo_recon_wo_align)
    # PatchAugNet: w recon, w/o align
    data_patchaugnet_w_recon_wo_align = RecallPrecisionTopN()
    data_patchaugnet_w_recon_wo_align.name = 'w/ recon, w/o align'
    data_patchaugnet_w_recon_wo_align.fig_color = 'green'
    data_patchaugnet_w_recon_wo_align.fig_marker = 'x'
    data_patchaugnet_w_recon_wo_align.data = np.array(
        [72.86044719, 78.56592136, 80.64764842, 81.80416345, 83.2690825, 84.42559753,
         85.50501157, 86.27602159, 86.27602159, 86.5073246, 86.89282961, 87.12413261,
         87.43253662, 87.89514264, 88.20354665, 88.58905166, 88.97455667, 89.28296068,
         89.8997687, 90.28527371, 90.51657672, 90.90208173, 91.21048574, 91.28758674,
         91.36468774])
    data_list.append(data_patchaugnet_w_recon_wo_align)
    # PatchAugNet: w/o recon, w align, w/o filter
    data_patchaugnet_wo_recon_w_align_wo_filter = RecallPrecisionTopN()
    data_patchaugnet_wo_recon_w_align_wo_filter.name = 'w/o recon, w/ align, w/o filter'
    data_patchaugnet_wo_recon_w_align_wo_filter.fig_color = 'chocolate'
    data_patchaugnet_wo_recon_w_align_wo_filter.fig_marker = 's'
    data_patchaugnet_wo_recon_w_align_wo_filter.data = np.array(
        [76.02158828, 79.10562837, 80.57054742, 82.42097147, 83.88589052, 84.50269854,
         85.42791056, 86.27602159, 87.12413261, 87.81804163, 88.20354665, 88.66615266,
         89.51426369, 89.66846569, 90.0539707, 90.36237471, 90.67077872, 91.05628373,
         91.36468774, 91.67309175, 91.90439476, 92.05859676, 92.13569776, 92.28989977,
         92.36700077])
    data_list.append(data_patchaugnet_wo_recon_w_align_wo_filter)
    # PatchAugNet: w/o recon, w align, w filter
    data_patchaugnet_wo_recon_w_align_w_filter = RecallPrecisionTopN()
    data_patchaugnet_wo_recon_w_align_w_filter.name = 'w/o recon, w/ align, w/ filter'
    data_patchaugnet_wo_recon_w_align_w_filter.fig_color = 'purple'
    data_patchaugnet_wo_recon_w_align_w_filter.fig_marker = 'P'
    data_patchaugnet_wo_recon_w_align_w_filter.data = np.array(
        [75.17347726, 78.48882035, 79.8766384, 80.80185042, 81.34155744, 82.80647648,
         83.50038551, 84.42559753, 85.27370856, 86.35312259, 86.5073246, 87.04703161,
         87.58673863, 88.51195066, 88.89745567, 89.20585968, 89.66846569, 90.0539707,
         90.20817271, 90.43947571, 90.90208173, 91.05628373, 91.21048574, 91.51888975,
         91.82729375])
    data_list.append(data_patchaugnet_wo_recon_w_align_w_filter)
    # PatchAugNet: w recon, w align, w/o filter
    data_patchaugnet_w_recon_w_align_wo_filter = RecallPrecisionTopN()
    data_patchaugnet_w_recon_w_align_wo_filter.name = 'w/ recon, w/ align, w/o filter'
    data_patchaugnet_w_recon_w_align_wo_filter.fig_color = 'gray'
    data_patchaugnet_w_recon_w_align_wo_filter.fig_marker = 'o'
    data_patchaugnet_w_recon_w_align_wo_filter.data = np.array(
        [76.02158828, 80.18504241, 82.49807247, 83.96299152, 84.81110254, 85.65921357,
         86.8157286, 87.27833462, 87.74094063, 88.04934464, 88.28064765, 88.43484965,
         88.82035466, 89.12875867, 89.43716268, 89.59136469, 89.9768697, 90.0539707,
         90.43947571, 90.82498072, 91.13338473, 91.51888975, 91.67309175, 92.13569776,
         92.21279877])
    data_list.append(data_patchaugnet_w_recon_w_align_wo_filter)
    # PatchAugNet(ours): w recon, w align, w filter
    data_patchaugnet_w_recon_w_align_w_filter = RecallPrecisionTopN()
    data_patchaugnet_w_recon_w_align_w_filter.name = 'w/ recon, w/ align, w/ filter'
    data_patchaugnet_w_recon_w_align_w_filter.fig_color = 'red'
    data_patchaugnet_w_recon_w_align_w_filter.fig_marker = 'd'
    data_patchaugnet_w_recon_w_align_w_filter.data = np.array(
        [76.40709329, 80.33924441, 82.03546646, 83.73168851, 84.81110254, 85.58211257,
         86.5073246, 87.43253662, 88.28064765, 88.74325366, 89.28296068, 89.59136469,
         89.8997687, 90.20817271, 90.36237471, 90.67077872, 90.97918273, 91.13338473,
         91.13338473, 91.44178874, 91.59599075, 91.75019275, 91.82729375, 92.05859676,
         92.13569776])
    data_list.append(data_patchaugnet_w_recon_w_align_w_filter)
    save_filepath = '/home/ericxhzou/ablation result-recall of patch feature augmentation on Data A.svg'
    draw_line_chart(data_list, title='Evaluation result on Data A', xlabel='N-Number of top database candidates',
                    ylabel='Recall @N(%)', save_filepath=save_filepath, yrange=[60, 100], ystep=5)


# draw ablation result of patch feature augmentation: recall@top1~25 curve data B
def draw_ablation_pfa_recall_top1_25_data_B():
    data_list = []
    # PatchAugNet: w/o recon, w/o align
    data_patchaugnet_wo_recon_wo_align = RecallPrecisionTopN()
    data_patchaugnet_wo_recon_wo_align.name = 'w/o recon, w/o align'
    data_patchaugnet_wo_recon_wo_align.fig_color = 'blue'
    data_patchaugnet_wo_recon_wo_align.fig_marker = 'v'
    data_patchaugnet_wo_recon_wo_align.data = np.array(
        [49.22501336, 54.40940673, 57.56280064, 60.23516836, 61.99893105, 63.65579904,
         65.6867985, 66.64885088, 67.98503474, 68.41261358, 69.3212186, 70.33671833,
         70.92463923, 71.61945484, 72.26082309, 72.74184928, 73.16942811, 73.91769107,
         74.34526991, 75.30732229, 75.94869054, 76.37626937, 76.75040086, 77.12453234,
         77.44521646])
    data_list.append(data_patchaugnet_wo_recon_wo_align)
    # PatchAugNet: w recon, w/o align
    data_patchaugnet_w_recon_wo_align = RecallPrecisionTopN()
    data_patchaugnet_w_recon_wo_align.name = 'w/ recon, w/o align'
    data_patchaugnet_w_recon_wo_align.fig_color = 'green'
    data_patchaugnet_w_recon_wo_align.fig_marker = 'x'
    data_patchaugnet_w_recon_wo_align.data = np.array(
        [53.76803848, 58.95243185, 62.37306253, 64.24371994, 66.48850882, 68.62640299,
         70.92463923, 72.3677178, 73.32977018, 74.34526991, 75.04008552, 75.89524319,
         76.85729556, 77.81934794, 78.46071619, 79.20897916, 79.90379476, 80.22447889,
         80.86584714, 81.18653127, 81.72100481, 82.04168894, 82.36237306, 82.62960983,
         82.84339925])
    data_list.append(data_patchaugnet_w_recon_wo_align)
    # PatchAugNet: w/o recon, w align, w/o filter
    data_patchaugnet_wo_recon_w_align_wo_filter = RecallPrecisionTopN()
    data_patchaugnet_wo_recon_w_align_wo_filter.name = 'w/o recon, w/ align, w/o filter'
    data_patchaugnet_wo_recon_w_align_wo_filter.fig_color = 'chocolate'
    data_patchaugnet_wo_recon_w_align_wo_filter.fig_marker = 's'
    data_patchaugnet_wo_recon_w_align_wo_filter.data = np.array(
        [53.87493319, 58.84553715, 61.46445751, 63.97648316, 65.52645644, 66.54195617,
         67.4505612, 68.51950828, 69.64190273, 70.33671833, 70.87119188, 71.40566542,
         72.15392838, 72.58150722, 73.11598076, 73.59700695, 74.07803314, 74.77284874,
         75.04008552, 75.46766435, 75.89524319, 76.37626937, 76.75040086, 77.17797969,
         77.60555852])
    data_list.append(data_patchaugnet_wo_recon_w_align_wo_filter)
    # PatchAugNet: w/o recon, w align, w filter
    data_patchaugnet_wo_recon_w_align_w_filter = RecallPrecisionTopN()
    data_patchaugnet_wo_recon_w_align_w_filter.name = 'w/o recon, w/ align, w/ filter'
    data_patchaugnet_wo_recon_w_align_w_filter.fig_color = 'purple'
    data_patchaugnet_wo_recon_w_align_w_filter.fig_marker = 'P'
    data_patchaugnet_wo_recon_w_align_w_filter.data = np.array(
        [56.97487974, 61.51790486, 63.86958846, 65.20577231, 67.55745591, 69.00053447,
         70.17637627, 71.29877071, 72.20737573, 73.16942811, 73.81079637, 74.8262961,
         75.25387493, 76.00213789, 76.85729556, 77.23142704, 77.60555852, 77.8727953,
         78.30037413, 78.67450561, 78.67450561, 78.99518974, 79.42276857, 79.79690005,
         80.06413683])
    data_list.append(data_patchaugnet_wo_recon_w_align_w_filter)
    # PatchAugNet: w recon, w align, w/o filter
    data_patchaugnet_w_recon_w_align_wo_filter = RecallPrecisionTopN()
    data_patchaugnet_w_recon_w_align_wo_filter.name = 'w/ recon, w/ align, w/o filter'
    data_patchaugnet_w_recon_w_align_wo_filter.fig_color = 'gray'
    data_patchaugnet_w_recon_w_align_wo_filter.fig_marker = 'o'
    data_patchaugnet_w_recon_w_align_wo_filter.data = np.array(
        [58.52485302, 63.60235168, 67.18332443, 68.94708712, 70.44361304, 71.88669161,
         73.00908605, 73.86424372, 74.87974345, 76.05558525, 76.48316408, 77.44521646,
         77.81934794, 78.46071619, 78.94174238, 79.31587386, 79.69000534, 80.11758418,
         80.54516301, 80.97274185, 81.23997862, 81.72100481, 81.93479423, 82.14858365,
         82.30892571])
    data_list.append(data_patchaugnet_w_recon_w_align_wo_filter)
    # PatchAugNet(ours): w recon, w align, w filter
    data_patchaugnet_w_recon_w_align_w_filter = RecallPrecisionTopN()
    data_patchaugnet_w_recon_w_align_w_filter.name = 'w/ recon, w/ align, w/ filter'
    data_patchaugnet_w_recon_w_align_w_filter.fig_color = 'red'
    data_patchaugnet_w_recon_w_align_w_filter.fig_marker = 'd'
    data_patchaugnet_w_recon_w_align_w_filter.data = np.array(
        [60.34206307, 65.04543025, 67.82469268, 69.58845537, 71.61945484, 72.74184928,
         73.59700695, 74.77284874, 75.5211117, 76.48316408, 76.91074292, 77.49866382,
         77.92624265, 78.19347942, 78.62105826, 78.88829503, 79.04863709, 79.52966328,
         79.95724212, 80.27792624, 80.81239979, 81.07963656, 81.29342598, 81.66755746,
         82.04168894])
    data_list.append(data_patchaugnet_w_recon_w_align_w_filter)
    save_filepath = '/home/ericxhzou/ablation result-recall of patch feature augmentation on Data B.svg'
    draw_line_chart(data_list, title='Evaluation result on Data B', xlabel='N-Number of top database candidates',
                    ylabel='Recall @N(%)', save_filepath=save_filepath, yrange=[45, 85], ystep=5)


# draw reconstructed point clouds
def draw_recon_pcs(recon_dir, show_fig=False):
    for i in range(1, 8):
        draw_pc(os.path.join(recon_dir, 'origin{}.npy'.format(i)), os.path.join(recon_dir, 'origin{}.svg'.format(i)), show_fig=show_fig)
        draw_pc(os.path.join(recon_dir, 'recon{}.npy'.format(i)), os.path.join(recon_dir, 'recon{}.svg'.format(i)), show_fig=show_fig)


# draw patch pairs
def draw_patch_pairs(patch_pair_dir, pt_size=3, show_fig=False):
    for i in range(0, 100):
        draw_pc(os.path.join(patch_pair_dir, 'i_{}.bin'.format(i)), os.path.join(patch_pair_dir, 'i_{}.svg'.format(i)), pt_size=pt_size, show_fig=show_fig)
        draw_pc(os.path.join(patch_pair_dir, 'j_{}.bin'.format(i)), os.path.join(patch_pair_dir, 'j_{}.svg'.format(i)), pt_size=pt_size, show_fig=show_fig)


# draw ablation result of feature aggregation: recall@top1~25 curve data A
def draw_ablation_fa_recall_top1_25_data_A():
    data_list = []
    # PatchAugNet-no-APFA (use Max Pooling)
    data_patchaugnet_no_apfa = RecallPrecisionTopN()
    data_patchaugnet_no_apfa.name = 'Max Pooling'
    data_patchaugnet_no_apfa.fig_color = 'blue'
    data_patchaugnet_no_apfa.fig_marker = 'v'
    data_patchaugnet_no_apfa.data = np.array([73.55435621, 78.33461835, 80.64764842, 82.80647648, 83.96299152, 85.19660756,
                                              86.12181958, 86.8157286, 87.27833462, 88.28064765, 89.12875867, 89.43716268,
                                              89.74556669, 89.82266769, 90.0539707, 90.36237471, 90.82498072, 91.36468774,
                                              91.44178874, 91.59599075, 91.67309175, 91.98149576, 92.05859676, 92.13569776,
                                              92.28989977])
    data_list.append(data_patchaugnet_no_apfa)
    # PatchAugNet-APFA1 (use APFA separately)
    data_patchaugnet_apfa1 = RecallPrecisionTopN()
    data_patchaugnet_apfa1.name = 'APFA1'
    data_patchaugnet_apfa1.fig_color = 'green'
    data_patchaugnet_apfa1.fig_marker = 'x'
    data_patchaugnet_apfa1.data = np.array([58.59676176, 64.14803392, 68.00308404, 69.8535081, 71.70393215, 72.93754819,
                                            74.17116423, 75.48188126, 76.6383963, 78.41171935, 79.18272938, 79.79953739,
                                            80.64764842, 81.80416345, 83.03777949, 83.65458751, 84.50269854, 84.88820355,
                                            85.58211257, 85.89051658, 86.43022359, 87.43253662, 87.66383963, 87.81804163,
                                            88.28064765])
    data_list.append(data_patchaugnet_apfa1)
    # PatchAugNet-APFA2 (ours, use APFA only once)
    data_patchaugnet_apfa2 = RecallPrecisionTopN()
    data_patchaugnet_apfa2.name = 'APFA2'
    data_patchaugnet_apfa2.fig_color = 'red'
    data_patchaugnet_apfa2.fig_marker = 'd'
    data_patchaugnet_apfa2.data = np.array([76.40709329, 80.33924441, 82.03546646, 83.73168851, 84.81110254, 85.58211257,
                                            86.5073246, 87.43253662, 88.28064765, 88.74325366, 89.28296068, 89.59136469,
                                            89.8997687, 90.20817271, 90.36237471, 90.67077872, 90.97918273, 91.13338473,
                                            91.13338473, 91.44178874, 91.59599075, 91.75019275, 91.82729375, 92.05859676,
                                            92.13569776])
    data_list.append(data_patchaugnet_apfa2)
    save_filepath = '/home/ericxhzou/ablation result-recall of feature aggregation on Data A.svg'
    draw_line_chart(data_list, title='Evaluation result on Data A', xlabel='N-Number of top database candidates',
                    ylabel='Recall @N(%)', save_filepath=save_filepath, yrange=[50, 100], ystep=5)


# draw ablation result of feature aggregation: recall@top1~25 curve data B
def draw_ablation_fa_recall_top1_25_data_B():
    data_list = []
    # PatchAugNet-no-APFA (use Max Pooling)
    data_patchaugnet_no_apfa = RecallPrecisionTopN()
    data_patchaugnet_no_apfa.name = 'Max Pooling'
    data_patchaugnet_no_apfa.fig_color = 'blue'
    data_patchaugnet_no_apfa.fig_marker = 'v'
    data_patchaugnet_no_apfa.data = np.array([45.10956708, 50.82843399, 54.5697488, 56.97487974, 59.00587921, 61.19722074,
                                              62.90753608, 64.61785142, 65.74024586, 66.8626403, 68.1453768, 68.94708712,
                                              69.37466595, 69.9091395, 70.33671833, 70.92463923, 71.56600748, 72.04703367,
                                              72.52805986, 72.90219134, 73.49011224, 73.6504543, 73.91769107, 74.39871726,
                                              74.71940139])
    data_list.append(data_patchaugnet_no_apfa)
    # PatchAugNet-APFA1 (use APFA separately)
    data_patchaugnet_apfa1 = RecallPrecisionTopN()
    data_patchaugnet_apfa1.name = 'APFA1'
    data_patchaugnet_apfa1.fig_color = 'green'
    data_patchaugnet_apfa1.fig_marker = 'x'
    data_patchaugnet_apfa1.data = np.array([40.29930518, 47.62159273, 51.84393373, 54.83698557, 57.13522181, 59.21966863,
                                            60.66274719, 62.42650989, 63.70924639, 64.72474613, 65.36611438, 66.48850882,
                                            67.4505612, 68.51950828, 69.21432389, 69.69535008, 70.06948156, 70.6039551,
                                            70.92463923, 71.56600748, 72.10048103, 72.68840192, 73.0625334, 73.43666489,
                                            74.18492785])
    data_list.append(data_patchaugnet_apfa1)
    # PatchAugNet-APFA2 (ours, use APFA only once)
    data_patchaugnet_apfa2 = RecallPrecisionTopN()
    data_patchaugnet_apfa2.name = 'APFA2'
    data_patchaugnet_apfa2.fig_color = 'red'
    data_patchaugnet_apfa2.fig_marker = 'd'
    data_patchaugnet_apfa2.data = np.array([60.34206307, 65.04543025, 67.82469268, 69.58845537, 71.61945484, 72.74184928,
                                            73.59700695, 74.77284874, 75.5211117, 76.48316408, 76.91074292, 77.49866382,
                                            77.92624265, 78.19347942, 78.62105826, 78.88829503, 79.04863709, 79.52966328,
                                            79.95724212, 80.27792624, 80.81239979, 81.07963656, 81.29342598, 81.66755746,
                                            82.04168894])
    data_list.append(data_patchaugnet_apfa2)
    save_filepath = '/home/ericxhzou/ablation result-recall of feature aggregation on Data B.svg'
    draw_line_chart(data_list, title='Evaluation result on Data B', xlabel='N-Number of top database candidates',
                    ylabel='Recall @N(%)', save_filepath=save_filepath, yrange=[35, 85], ystep=5)


def draw_demo_pic(mls_traj_file, pls_traj_file, result_file, demo_out_dir, demo_title, show_fig=False):
    if not show_fig:
        matplotlib.use('Agg')
    import csv
    import math
    history_query_xs = []
    history_query_ys = []
    history_query_states = []

    # 从文本文件中读取MLS轨迹数据
    mls_data = []
    if os.path.splitext(os.path.basename(mls_traj_file))[1] == ".csv":
        pls_hgt = 100.0
        with open(mls_traj_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                mls_data.append([float(row[1]), float(row[2]), 0.0])
    else:
        pls_hgt = 1.0
        with open(mls_traj_file, 'r') as f:
            lines = f.readlines()
            lines.pop(0)
            count = 0
            for line in lines:
                line_strs = line.split(' ')
                if count % 50 == 0:
                    mls_data.append([float(line_strs[8]), float(line_strs[7]), 0.0])
                count = count + 1
    mls_data = np.array(mls_data)
    mls_x, mls_y, mls_z = mls_data[:, 1], mls_data[:, 0], mls_data[:, 2]

    # 从文本文件中读取PLS轨迹数据
    pls_data = []
    with open(pls_traj_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            pls_data.append([float(row[1]), float(row[2]), pls_hgt])
    pls_data = np.array(pls_data)
    pls_x, pls_y, pls_z = pls_data[:, 1], pls_data[:, 0], pls_data[:, 2]

    # get query files and found files
    with open(result_file, 'r') as f:
        lines = f.readlines()
        lines.pop(0)
    num_lines = len(lines)
    num_units = num_lines // 7
    item_dict = {
            'state': [],
            'name': [],
            'query_file': [],
            'query_x': [],
            'query_y': [],
            'found_file': [],
            'found_x': [],
            'found_y': []
        }
    for i in range(num_units):
        unit_lines = lines[i*7 : (i+1)*7]
        line0_strs = unit_lines[0].split(' ')
        query_submap_file = line0_strs[2]
        query_x = float(line0_strs[6])
        query_y = float(line0_strs[8])
        found_submap_file = None
        found_submap_state = False
        found_x = found_y = None
        for j in range(2, 7):
            unit_lines_j_strs = unit_lines[j].split(' ')
            found_submap_file = unit_lines_j_strs[1]
            found_submap_state = unit_lines_j_strs[3]
            found_x = float(unit_lines_j_strs[5])
            found_y = float(unit_lines_j_strs[7])
            if found_submap_state == 'True':
                found_submap_state = True
                break
            else:
                found_submap_state = False

        #if found_submap_state is False:
        #    continue

        dist = math.sqrt((query_x - found_x)**2 + (query_y - found_y)**2)
        if dist > 30.0 and found_submap_state:
            continue

        item_dict['state'].append(found_submap_state)
        item_dict['name'].append(float(os.path.splitext(os.path.basename(query_submap_file))[0]))
        item_dict['query_file'].append(query_submap_file)
        item_dict['query_x'].append(query_x)
        item_dict['query_y'].append(query_y)
        item_dict['found_file'].append(found_submap_file)
        item_dict['found_x'].append(found_x)
        item_dict['found_y'].append(found_y)

    item_pd = pd.DataFrame(item_dict)
    #item_pd = item_pd.sort_values(by=['name'])
    item_pd = item_pd.sample(frac=1.0)
    count = 0
    for index, row in item_pd.iterrows():
        if count == 250:
            break
        # 创建画布和子图
        fig = plt.figure(figsize=(16, 9))
        gs = fig.add_gridspec(2, 3)

        # 绘制MLS&PLS轨迹，添加标题
        axs_left = fig.add_subplot(gs[:, :2], projection='3d')
        axs_left.scatter(mls_x, mls_y, mls_z, s=2, color='gray', zorder=1)
        axs_left.scatter(pls_x, pls_y, pls_z, s=2, color='black', zorder=1)
        # axs_left.set_title('MLS and PLS Trajectory')
        if not os.path.splitext(os.path.basename(mls_traj_file))[1] == ".csv":
            axs_left.view_init(elev=60.0, azim=-90)
        axs_left.axis('off')

        # 连接加粗的轨迹点
        line_color = 'limegreen'
        if not row['state']:
            line_color = 'red'
        history_query_xs.append(row['query_x'])
        history_query_ys.append(row['query_y'])
        history_query_states.append(row['state'])
        for j in range(len(history_query_xs)):
            if not history_query_states[j]:
                axs_left.scatter(history_query_xs[j], history_query_ys[j], pls_hgt, marker='o', s=75, color='red', zorder=3)
            else:
                axs_left.scatter(history_query_xs[j], history_query_ys[j], pls_hgt, marker='o', s=75, color='limegreen',zorder=3)
        axs_left.scatter(row['found_x'], row['found_y'], 0.0, marker='o', s=25, color='black', zorder=3)
        axs_left.plot([row['query_x'], row['found_x']], [row['query_y'], row['found_y']], [pls_hgt, 0.0], linewidth=4, color=line_color, zorder=2)

        # 绘制找到的query子图点云，添加标题
        query_pc = load_pc_file(row['query_file'], use_np_load=True)
        query_pc = normalize_point_cloud(query_pc)
        axs11 = fig.add_subplot(gs[0, 2], projection='3d')
        axs11.scatter(query_pc[:, 0], query_pc[:, 1], query_pc[:, 2], s=3, c=query_pc[:, 2], cmap='rainbow')
        axs11.set_title('Query Submap: {}'.format(row['name']))
        axs11.axis()

        # 绘制找到的reference子图点云，添加标题
        ref_pc = load_pc_file(row['found_file'], use_np_load=True)
        ref_pc = normalize_point_cloud(ref_pc)
        axs01 = fig.add_subplot(gs[1, 2], projection='3d')
        axs01.scatter(ref_pc[:, 0], ref_pc[:, 1], ref_pc[:, 2], s=3, c=ref_pc[:, 2], cmap='rainbow')
        axs01.set_title('Matched Submap: {}'.format(os.path.splitext(os.path.basename(row['found_file']))[0]))
        axs01.axis()

        # 添加整个图的标题
        fig.suptitle(demo_title)
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9, wspace=0.1, hspace=0.2)

        # 保存图片
        out_file = os.path.join(demo_out_dir, str(count) + ".png")
        plt.savefig(out_file, dpi=200, bbox_inches='tight')
        plt.close('all')
        count = count + 1


if __name__ == '__main__':
    # ----------test Draw point pairs
    # src_pc = np.random.randint(-30, 30, (100, 3))
    # src_kpt = np.random.randint(-30, 30, (10, 3))
    # tgt_pc = np.random.randint(-30, 30, (100, 3))
    # tgt_kpt = np.random.randint(-30, 30, (10, 3))
    # pps_state = np.random.randint(0, 2, (10, 1))
    # draw_pc_pps(src_pc, src_kpt, tgt_pc, tgt_kpt, pps_state)

    # ----------test Draw tsne
    # features = np.random.randint(0, 100, (25, 32))
    # labels = list(np.random.randint(0, 2, (25, 1)))
    # pc_idxs = list(range(25))
    # pc_files = ['/home/ericxhzou/Code/benchmark_datasets/whu_campus_origin/mls_submap_along_traj/pointcloud_30m_2m_clean/1637217995817382.bin'] * 25
    # draw_features_with_tsne(features, labels, pc_idxs, pc_files, '/home/ericxhzou/Code/ppt-net-plus/exp/reranknet/tsne.png')

    # ----------Draw reconstructed pc
    # draw_recon_pcs('/home/ericxhzou/Data/recon_clouds', show_fig=False)
    # ----------Draw patch pairs
    # draw_patch_pairs('/media/ericxhzou/eric_1T/2-kl_div_result', pt_size=30, show_fig=False)
    # ----------Draw disappeared bad cases
    # draw_pcs_in_dir('/home/ericxhzou/Code/ppt-net-plus/exp/pptnet/events/2023-03-16T14-15-14-model-epoch=24/cmp_res_no_pfaa_maxpool/disappeared_badcase_ref')
    # draw_pcs_in_dir('/home/ericxhzou/Code/ppt-net-plus/exp/pptnet/events/2023-03-16T14-15-14-model-epoch=24/cmp_res_no_pfaa_maxpool/disappeared_badcase_cmp')

    # ----------Draw demo pictures
    # mls_traj_file = '/home/ericxhzou/Code/benchmark_datasets/wh_hankou_origin/map_submap_along_traj/pointcloud_30m_2m_clean.csv'
    # pls_traj_file = '/home/ericxhzou/Code/benchmark_datasets/wh_hankou_origin/helmet_submap/pointcloud_30m_2m_clean.csv'
    # result_file = '/home/ericxhzou/Code/ppt-net-plus/exp/pptnet/events/2023-05-16T19-58-42/result_dataA.txt'
    # demo_out_dir = '/home/ericxhzou/Code/ppt-net-plus/exp/pptnet/events/2023-05-16T19-58-42/png'
    # demo_title = 'Place recognition results on Data A, PLS (query) to MLS (map) using PatchAugNet'
    # draw_demo_pic(mls_traj_file, pls_traj_file, result_file, demo_out_dir, demo_title)
    #
    # mls_traj_file = '/home/ericxhzou/Code/ppt-net-plus/exp/pptnet/events/2023-05-16T20-00-40/@@2021-11-18-063511_Scanner1.PosL'
    # pls_traj_file = '/home/ericxhzou/Code/benchmark_datasets/whu_campus_origin/helmet_submap/pointcloud_30m_2m_clean.csv'
    # result_file = '/home/ericxhzou/Code/ppt-net-plus/exp/pptnet/events/2023-05-16T20-00-40/result_dataB.txt'
    # demo_out_dir = '/home/ericxhzou/Code/ppt-net-plus/exp/pptnet/events/2023-05-16T20-00-40/png'
    # demo_title = 'Place recognition results on Data B, PLS (query) to MLS (map) using PatchAugNet'
    # draw_demo_pic(mls_traj_file, pls_traj_file, result_file, demo_out_dir, demo_title)

    # ----------Draw place recognition results
    # # draw place recognition result: recall@top1~25 curve data A
    # draw_pr_recall_top1_25_data_A()
    # draw_pr_precision_top1_25_data_A()
    # draw_rerank_recall_top1_25_data_A()
    # draw_rerank_precision_top1_25_data_A()
    # # draw place recognition result: recall@top1~25 curve data B
    # draw_pr_recall_top1_25_data_B()
    # draw_pr_precision_top1_25_data_B()
    # draw_rerank_recall_top1_25_data_B()
    # draw_rerank_precision_top1_25_data_B()
    # draw place recognition result: recall@top1_25 curve data Oxford / 3-Inhouse
    draw_pr_recall_top1_25_data_oxford()
    draw_pr_recall_top1_25_data_university()
    draw_pr_recall_top1_25_data_residential()
    draw_pr_recall_top1_25_data_business()
    # draw place recognition result: success / failure cases
    #draw_pr_cases('/home/ericxhzou/Code/ppt-net-plus/exp/pptnet/events/2023-03-16T14-15-14-model-epoch=24/query_results_bin', num_query=474, num_top=5, show_fig=False)

    # ----------Draw ablation study results
    # # draw ablation result of patch feature augmentation: recall@top1~25 curve data A
    # draw_ablation_pfa_recall_top1_25_data_A()
    # # draw ablation result of patch feature augmentation: recall@top1~25 curve data B
    # draw_ablation_pfa_recall_top1_25_data_B()
    # # draw ablation result of feature aggregation: recall@top1~25 curve data A
    # draw_ablation_fa_recall_top1_25_data_A()
    # # draw ablation result of feature aggregation: recall@top1~25 curve data B
    # draw_ablation_fa_recall_top1_25_data_B()
