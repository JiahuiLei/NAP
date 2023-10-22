# Debug use, not used for evaluation
# check the sampled pcl npz by viz them
import os, sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join("..", os.path.dirname(os.path.abspath(__file__))))

from render_helper import render
import networkx as nx
from matplotlib import cm
from transforms3d.axangles import axangle2mat
from matplotlib import pyplot as plt
import imageio
import numpy as np
from tqdm import tqdm


def viz_npz(fn, dst, max_viz=5):
    data = np.load(fn, allow_pickle=True)
    pcl_list, T_list = data["pcl"], data["pose"]
    viz_list = []
    for pcl, Ts in zip(pcl_list, T_list):
        _rgbs = []
        for T in Ts:
            T_inv = np.linalg.inv(T)
            pcl = np.matmul(T_inv[:3, :3], pcl.T).T + T_inv[:3, 3]
            # canonicalize the pcl
            rgb = render(
                pcl_list=[pcl],
                pcl_radius_list=[0.03],
                cam_dist=3.0,
                cam_angle_pitch=np.pi / 4,
                cam_angle_yaw=np.pi / 4,
                shape=(200,200)
            )
            
            _rgbs.append(rgb)
        _rgbs = np.concatenate(_rgbs, axis=1,)
        viz_list.append(_rgbs)
        if len(viz_list) == max_viz:
            break
    ret = np.concatenate(viz_list, axis=0)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    imageio.imsave(dst, ret)
    return


# name = "K_8_cate_all_gt"
# name = "K_8_cate_all_b2.3.1_5455"
# name = "K_8_cate_all_b2.2_5455"
# name = "K_8_cate_all_b1.0_9091"
# name = "K_8_cate_all_b1.0_9091_retrieval"
# name = "K_8_cate_all_b2.0_5455"
# name = "K_8_cate_all_v5.1.5_5455"
name = "K_8_cate_all_v5.1.6_5455"

src = f"../log/test/PCL/{name}"
dst = f"../log/test/PCL_viz/{name}"
for fn in tqdm(os.listdir(src)):
    if fn.endswith(".npz"):
        viz_npz(os.path.join(src, fn), os.path.join(dst, fn)+".png")
# viz_npz("../log/test/PCL/K_8_cate_all_gt/45594.npz")
