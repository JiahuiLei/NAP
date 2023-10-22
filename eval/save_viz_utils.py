# helpers for Save the generated V, E
import sys, os, os.path as osp

sys.path.append(osp.dirname(os.getcwd()))
from matplotlib.axes._axes import _log as matplotlib_axes_logger

matplotlib_axes_logger.setLevel("ERROR")
import yaml, logging, imageio, torch, os
import os.path as osp
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import networkx as nx
from copy import deepcopy
import trimesh

device = torch.device("cuda:0")
from object_utils.arti_graph_utils_v3 import get_G_from_VE
from object_utils.arti_viz_utils import append_mesh_to_G, viz_G_topology, viz_G
import pickle
from multiprocessing import Pool

# import multiprocessing


def extract_recon_mesh_for_nodes(G, extract_fn):
    for v, v_data in G.nodes(data=True):
        if "additional" not in v_data:
            continue
        z = v_data["additional"][None, :]
        mesh = extract_fn(torch.from_numpy(z).cuda())
        mesh_centroid = mesh.bounds.mean(0)
        mesh.apply_translation(-mesh_centroid)
        bbox = v_data["bbox"].copy()
        scale = 2.0 * np.linalg.norm(bbox) / np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])
        mesh.apply_scale(scale)
        nx.set_node_attributes(G, {v: {"mesh": mesh}})
    return G


def find_nn_database_mesh_and_update_G(G, database, mesh_names, mesh_dir):
    for v, v_data in G.nodes(data=True):
        if "additional" not in v_data:
            continue
        z = v_data["additional"][None, :]
        # find nn
        _d, _ind = database.kneighbors(z, return_distance=True)
        # print(_ind)
        _ind = int(_ind.squeeze(0))
        fn = osp.join(mesh_dir, mesh_names[int(_ind)] + ".off")
        gt_mesh = trimesh.load(fn, force="mesh", process=False)  # ! debug
        mesh_centroid = gt_mesh.bounds.mean(0)
        gt_mesh.apply_translation(-mesh_centroid)
        bbox = v_data["bbox"].copy()
        scale = 2.0 * np.linalg.norm(bbox) / np.linalg.norm(gt_mesh.bounds[1] - gt_mesh.bounds[0])
        gt_mesh.apply_scale(scale)
        nx.set_node_attributes(G, {v: {"mesh": gt_mesh}})
    return G


def _save_viz_thread(p):
    G, save_dir, name = p
    # print(p)
    viz_dir = save_dir + "_viz"
    os.makedirs(viz_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    # viz_topo = viz_G_topology(G)
    # imageio.imwrite(osp.join(viz_dir, f"{name}.png"), viz_topo)
    # print("start rendering...")
    viz_list = viz_G(G, cam_dist=3.0, viz_frame_N=5, shape=(128, 128))
    imageio.mimsave(osp.join(viz_dir, f"{name}.gif"), viz_list, fps=10)
    # print("end rendering...")
    fn = osp.join(save_dir, f"{name}.pkl")
    pickle.dump(G, open(fn, "wb"))
    data = pickle.load(open(fn, "rb"))
    return


def save(
    gen_V,
    gen_E,
    mesh_extraction_fn=None,
    dst=None,
    dst_retrieval=None,
    database=None,
    mesh_names=None,
    gt_mesh_dir=None,
):
    N = gen_V.shape[0]

    save_gen_flag = dst is not None
    save_retrieval_flag = dst_retrieval is not None
    assert save_gen_flag or save_retrieval_flag
    if save_gen_flag:
        os.makedirs(dst, exist_ok=True)
    if save_retrieval_flag:
        os.makedirs(dst_retrieval, exist_ok=True)

    # G_p_list, G_r_p_list = [], []
    for i in tqdm(range(N)):
        v, e = gen_V[i].cpu().numpy(), gen_E[i].cpu().numpy()
        if save_gen_flag:
            G = get_G_from_VE(v, e)
            G = extract_recon_mesh_for_nodes(G, mesh_extraction_fn)
            # G_p_list.append((G, dst, f"gen_{i}"))
            fn = osp.join(dst, f"gen_{i}.pkl")
            pickle.dump(G, open(fn, "wb"))
            pickle.load(open(fn, "rb"))  # check

        if save_retrieval_flag:
            if save_gen_flag:
                G_r = deepcopy(G)
            else:
                G_r = get_G_from_VE(v, e)
            G_r = find_nn_database_mesh_and_update_G(G_r, database, mesh_names, gt_mesh_dir)
            # G_r_p_list.append((G_r, dst_retrieval, f"gen_retrieval_{i}"))
            fn = osp.join(dst_retrieval, f"gen_retrieval_{i}.pkl")
            pickle.dump(G_r, open(fn, "wb"))
            pickle.load(open(fn, "rb"))  # check


def _viz_thread(p):
    G, fn, n_frames = p
    viz_list = viz_G(G, cam_dist=3.0, viz_frame_N=n_frames, shape=(256, 256))
    print(f"finished render {fn}")
    imageio.mimsave(fn, viz_list, fps=10)
    print(f"finished save {fn}")
    return


def viz_dir(src, dst, n_threads=16, max_viz=100, n_frames=5):
    os.makedirs(dst, exist_ok=True)
    fn_list = [f for f in os.listdir(src) if f.endswith(".pkl")]
    p_list = []
    for fn in fn_list:
        G = pickle.load(open(osp.join(src, fn), "rb"))
        viz_fn = osp.join(dst, fn[:-4] + ".gif")
        p_list.append((G, viz_fn, n_frames))
    if len(p_list) > max_viz:
        step = len(p_list) // max_viz
        p_list = p_list[::step]
    print(f"start rendering {len(p_list)} files...")
    with Pool(n_threads) as p:
        p.map(_viz_thread, p_list)
    return


def save_and_viz(
    gen_V,
    gen_E,
    mesh_extraction_fn=None,
    dst=None,
    dst_retrieval=None,
    database=None,
    mesh_names=None,
    gt_mesh_dir=None,
    n_threads=16,
):
    N = gen_V.shape[0]

    save_gen_flag = dst is not None
    save_retrieval_flag = dst_retrieval is not None
    assert save_gen_flag or save_retrieval_flag

    G_p_list, G_r_p_list = [], []
    for i in tqdm(range(N)):
        v, e = gen_V[i].cpu().numpy(), gen_E[i].cpu().numpy()
        if save_gen_flag:
            G = get_G_from_VE(v, e)
            G = extract_recon_mesh_for_nodes(G, mesh_extraction_fn)
            G_p_list.append((G, dst, f"gen_{i}"))

        if save_retrieval_flag:
            if save_gen_flag:
                G_r = deepcopy(G)
            else:
                G_r = get_G_from_VE(v, e)
            G_r = find_nn_database_mesh_and_update_G(G_r, database, mesh_names, gt_mesh_dir)
            G_r_p_list.append((G_r, dst_retrieval, f"gen_retrieval_{i}"))

    if save_gen_flag:
        print(f"save gen {len(G_p_list)}")
        with Pool(n_threads) as p:
            p.map(_save_viz_thread, G_p_list)
    if save_retrieval_flag:
        print(f"save retrieval {len(G_r_p_list)}")
        with Pool(n_threads) as p:
            p.map(_save_viz_thread, G_r_p_list)
