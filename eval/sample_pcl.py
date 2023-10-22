# sample saved pkl Graphs
import pickle, trimesh
import networkx as nx
import numpy as np
from transforms3d.axangles import axangle2mat
from tqdm import tqdm
import logging

logger = logging.getLogger("trimesh")
logger.setLevel(logging.ERROR)


def sample(pkl_fn, dst_fn, N_states=100, N_PCL=10000):
    G = pickle.load(open(pkl_fn, "rb"))
    mesh_list, pose_list = forward_G(G, N_frame=N_states)
    pcl_list = [sample_tmesh(mesh, N_PCL) for mesh in mesh_list]
    # # debug
    # for sid in range(N_states):
    #     for pid, pose in enumerate(pose_list[sid]):
    #         T_inv = np.linalg.inv(pose)
    #         pcl = np.concatenate([pcl_list[sid].copy(), np.ones((pcl_list[sid].shape[0], 1))], 1).T
    #         pcl = T_inv @ pcl
    #         pcl = pcl.T[:, :3]
    #         np.savetxt(f"./debug/{sid}_{pid}.txt", fmt="%.6f", X=pcl)

    pose_list = np.stack(pose_list, 0)  # N_states, N_parts, 4,4
    pcl_list = np.stack(pcl_list, 0)  # N_states, N_pcl, 3
    np.savez_compressed(dst_fn, pcl=pcl_list, pose=pose_list)
    return


def sample_tmesh(tmesh, pre_sample_n):
    pcl, _ = trimesh.sample.sample_surface_even(tmesh, pre_sample_n * 2)
    while len(pcl) < pre_sample_n:
        _pcl, _ = trimesh.sample.sample_surface_even(tmesh, pre_sample_n * 2)
        pcl = np.concatenate([pcl, _pcl])
    pcl = pcl[:pre_sample_n]
    pcl = np.asarray(pcl, dtype=np.float16)
    return pcl


def screw_to_T(theta, d, l, m):
    assert abs(np.linalg.norm(l) - 1.0) < 1e-4
    R = axangle2mat(l, theta)
    t = (np.eye(3) - R) @ (np.cross(l, m)) + l * d
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def forward_G(G, N_frame=100, mesh_key="mesh"):
    ret = []
    assert len(G.nodes) >= 2 and nx.is_tree(G)  # now only support tree viz
    # * now G is connected and acyclic
    vid = [n for n in G.nodes]
    v_bbox = np.stack([d["bbox"] for nid, d in G.nodes(data=True)], 0)
    v_volume = v_bbox.prod(axis=-1) * 8
    root_vid = vid[v_volume.argmax()]
    POSE, MESH = [], []
    # * sample a set of possible angle range for each joint
    for step in tqdm(range(N_frame)):
        node_traverse_list = [n for n in nx.dfs_preorder_nodes(G, root_vid)]
        T_rl_list = [np.eye(4)]  # p_root = T_rl @ p_link
        # * prepare the node pos
        for i in range(len(node_traverse_list) - 1):
            cid = node_traverse_list[i + 1]
            for e, e_data in G.edges.items():
                if cid in e:
                    # determine the direction by ensure the other end is a predessor in the traversal list
                    other_end = e[0] if e[1] == cid else e[1]
                    if node_traverse_list.index(other_end) > i:
                        continue
                    else:
                        pid = other_end

                    # T1: e_T_src_j1, T2: e_T_j2_dst
                    e_data = G.edges[e]
                    _T0 = e_data["T_src_dst"]
                    plucker = e_data["plucker"]
                    l, m = plucker[:3], plucker[3:]
                    plim, rlim = e_data["plim"], e_data["rlim"]

                    # ! random sample
                    # theta = np.linspace(*rlim, N_frame)[step]
                    # d = np.linspace(*plim, N_frame)[step]
                    theta = np.random.uniform(*rlim)
                    d = np.random.uniform(*plim)

                    _T1 = screw_to_T(theta, d, l, m)
                    T_src_dst = _T1 @ _T0
                    if pid == e_data["src"]:  # parent is src
                        T_parent_child = T_src_dst
                    else:  # parent is dst
                        T_parent_child = np.linalg.inv(T_src_dst)
                        # T_parent_child = T_src_dst
                    T_root_child = T_rl_list[node_traverse_list.index(pid)] @ T_parent_child
                    T_rl_list.append(T_root_child)
                    break
        assert len(T_rl_list) == len(node_traverse_list)

        # * prepare the bbox
        mesh_list, mesh_color_list = [], []
        for nid, T in zip(node_traverse_list, T_rl_list):
            assert mesh_key in G.nodes[nid].keys()
            mesh = G.nodes[nid][mesh_key].copy()
            mesh.apply_transform(T.copy())
            mesh_list.append(mesh)
        mesh_list = trimesh.util.concatenate(mesh_list)
        MESH.append(mesh_list)
        POSE.append(np.stack(T_rl_list, 0))

    return MESH, POSE


def thread(p):
    pkl_fn, dst_fn, N_states, N_PCL = p
    sample(pkl_fn, dst_fn, N_states, N_PCL)
    return


if __name__ == "__main__":
    import os
    from multiprocessing import Pool
    import argparse
    from random import shuffle

    arg_parser = argparse.ArgumentParser(description="Run")
    arg_parser.add_argument(
        "--src",
        default="../log/test/G/K_8_cate_all_gt",
    )
    arg_parser.add_argument(
        "--dst",
        default="../log/test/PCL/K_8_cate_all_gt",
    )
    arg_parser.add_argument("--n_states", default=50, type=int)
    arg_parser.add_argument("--n_pcl", default=5000, type=int)
    arg_parser.add_argument("--n_thread", default=16, type=int)
    args = arg_parser.parse_args()

    SRC = args.src
    DST = args.dst # + f"_N_{args.n_states}_PCL_{args.n_pcl}"
    N_states = args.n_states
    N_pcl = args.n_pcl
    os.makedirs(DST, exist_ok=True)
    p_list = []
    for fn in os.listdir(SRC):
        if fn.endswith(".pkl"):
            pkl_fn = os.path.join(SRC, fn)
            dst_fn = os.path.join(DST, fn[:-4] + ".npz")
            p_list.append((pkl_fn, dst_fn, N_states, N_pcl))
    # sample("../log/test/K_8_cate_all_gt/103593.pkl", "./debug.npz", N_states=100, N_PCL=10000)
    shuffle(p_list)
    with Pool(args.n_thread) as p:
        p.map(thread, p_list)
