# v2 is
# ! by default, the list of edges represent the upper triangle, i.e. row i, col j, then i < j

import torch
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
import numpy as np
import logging
import networkx as nx
from matplotlib import cm
from transforms3d.axangles import axangle2mat
import imageio
import torch.nn.functional as F


def map_upper_triangle_to_list(i, j, K):
    assert i < j, "not upper triangle"
    e_list_ind = i * (2 * K - i - 1) // 2 + j - i - 1
    return e_list_ind


def plucker_need_flip(plucker):
    # if z < 0, or if z=0 and y<0 or y,z = 0 and x<0, then flip
    assert plucker.ndim == 1
    l = plucker[:3]
    if l[2] < 0:
        return True
    elif l[2] == 0:
        if l[1] < 0:
            return True
        elif l[1] == 0:
            assert l[0] != 0, "don't check infinity line"
            if l[0] < 0:
                return True
    return False


def compact_pack(V, E, K=25, permute=True):
    if len(V) > K:
        print(f"Warning, extend {K} to {len(V)}")
        K = len(V)
    num_v = len(V)
    n_empty = K - num_v

    # Nodes
    v_mask = np.zeros(K, dtype=np.bool)
    v_mask[: len(V)] = True
    if permute:
        v_map = np.random.permutation(num_v).tolist()  # stores the original id
    else:
        v_map = np.arange(num_v).tolist()
    v_bbox = [V[i]["bbox_L"] for i in v_map] + [np.zeros(3)] * n_empty
    v_bbox = torch.from_numpy(np.stack(v_bbox, axis=0)).float()
    # p_global = T_gl @ p_local
    v_t_gl = [V[i]["abs_center"] for i in v_map] + [np.zeros(3)] * n_empty
    v_t_gl = torch.from_numpy(np.stack(v_t_gl, axis=0)).float()
    # ! Now assume partnet-M all init part R = I
    v_r_gl = torch.zeros(K, 3).float()
    ret_v = torch.cat([torch.from_numpy(v_mask[..., None]), v_bbox, v_r_gl, v_t_gl], -1)

    # Edges
    total_edges = int(K * (K - 1) / 2)  # include invalid
    e_plucker = torch.zeros((total_edges, 6), dtype=torch.float32)
    e_lim = torch.zeros((total_edges, 4), dtype=torch.float32)
    e_type = torch.zeros((total_edges), dtype=torch.long)  # [0,1,2] [empty, ij, ji]
    for e in E:
        # ! by default, the list of edges represent the upper triangle, i.e. row i, col j, then i < j
        _src_ind, _dst_ind = e["e0"]["src_ind"], e["e0"]["dst_ind"]
        src_ind, dst_ind = v_map.index(_src_ind), v_map.index(_dst_ind)
        plucker = e["e0"]["plucker"]
        # transform the plucker to global frame
        _r_global = v_r_gl[src_ind]
        _t_global = v_t_gl[src_ind]
        plucker_global = torch.from_numpy(plucker.copy()).float()
        _R_global = axis_angle_to_matrix(_r_global)
        _lg = _R_global @ plucker_global[:3]
        _mg = _R_global @ plucker_global[3:] + torch.cross(_t_global, _lg)
        plucker_global = torch.cat([_lg, _mg], 0)
        flip = plucker_need_flip(plucker_global)
        if flip:  # orient the global plucker to hemisphere
            plucker_global = -plucker_global

        if src_ind > dst_ind:  # i = dst, j = src
            i, j = dst_ind, src_ind
            flip = not flip  # when reverse the src and dst, the plucker should multiply by -1.0
        elif src_ind < dst_ind:
            i, j = src_ind, dst_ind
        else:
            raise ValueError("src_ind == dst_ind")
        e_list_ind = map_upper_triangle_to_list(i, j, K)

        if flip:  # 2 is flip plucker
            e_type[e_list_ind] = 2
        else:  # 1 is not flip plucker
            e_type[e_list_ind] = 1

        e_lim[e_list_ind, :2] = torch.Tensor(e["r_limits"])
        e_lim[e_list_ind, 2:] = torch.Tensor(e["p_limits"])

        e_plucker[e_list_ind] = plucker_global

    e_type = F.one_hot(e_type, num_classes=3).float()

    ret_e = torch.cat([e_type, e_plucker, e_lim], dim=1)
    # v: [mask_occ(1), bbox(3), r_gl(3), t_gl(3) | additional codes in the future]
    # e: [type(3), plucker(6), rlim(2), plim(2)]
    return ret_v, ret_e, v_map


def get_G_from_VE(V, E):
    # v: [mask_occ(1), bbox(3), r_gl(3), t_gl(3) | additional codes in the future]
    # e: [type(3), plucker(6), rlim(2), plim(2)]
    # ! warning, here occ v mask must after sigmoid;
    if isinstance(V, torch.Tensor):
        V = V.cpu().numpy()
    if isinstance(E, torch.Tensor):
        E = E.cpu().numpy()
    v_mask = V[:, 0] > 0.5
    K = len(v_mask)
    assert len(E) == int(K * (K - 1) / 2), f"len(E)={len(E)}, K={K}"
    v = V[v_mask]
    n_v = len(v)
    assert v_mask[:n_v].all(), f"only support first N true v mask for now"
    G = nx.Graph()
    if n_v >= 2:
        node_color_list = cm.hsv(np.linspace(0, 1, n_v + 1))[:-1]
        # Fill in VTX
        for vid in range(n_v):
            G.add_node(vid)
            _r = torch.as_tensor(v[vid][4:7])
            R = axis_angle_to_matrix(_r).cpu().numpy()
            v_attr = {
                "vid": vid,
                "bbox": v[vid][1:4],
                "R": R.copy(),
                "t": v[vid][7:10].copy(),
                "color": node_color_list[vid],
            }
            if v.shape[1] > 10:
                v_attr["additional"] = v[vid][10:]
            nx.set_node_attributes(G, {vid: v_attr})
        # Fill in EDGE
        for _i in range(n_v):
            for _j in range(n_v):
                if _i >= _j:
                    continue
                # src = i, dst = j
                ind = map_upper_triangle_to_list(_i, _j, K)
                # e: [type(3), plucker(6), rlim(2), plim(2)]
                e_type = E[ind, :3].argmax()
                if e_type == 0:
                    continue
                e_plucker = E[ind, 3:9]
                if e_type == 2:  # flip
                    e_plucker = -e_plucker
                e_rlim, e_plim = E[ind, 9:11], E[ind, 11:13]
                T_gi, T_gj = np.eye(4), np.eye(4)
                T_gi[:3, :3] = G.nodes[_i]["R"]
                T_gi[:3, 3] = G.nodes[_i]["t"]
                T_gj[:3, :3] = G.nodes[_j]["R"]
                T_gj[:3, 3] = G.nodes[_j]["t"]
                T_ig = np.linalg.inv(T_gi).copy()
                T_ij = T_ig.copy() @ T_gj.copy()  # T0
                local_plucker = e_plucker.copy()
                li = T_ig[:3, :3] @ local_plucker[:3]
                mi = T_ig[:3, :3] @ local_plucker[3:] + np.cross(T_ig[:3, 3], li)
                local_plucker = np.concatenate([li, mi])
                G.add_edge(_i, _j)
                nx.set_edge_attributes(
                    G,
                    {
                        (_i, _j): {
                            "src": _i,
                            "dst": _j,
                            "T_src_dst": T_ij,
                            "plucker": local_plucker,
                            "plim": e_plim,
                            "rlim": e_rlim,
                        }
                    },
                )
    return G


if __name__ == "__main__":
    from arti_viz_utils import viz_G_topology, viz_G, append_mesh_to_G
    import trimesh
    import os.path as osp
    import os
    from multiprocessing import Pool
    
    # from partnet_urdf import PartNetMobilityURDF
    # urdf_fn = "/home/ray/datasets/partnet-mobility-v0/102506/mobility.urdf"
    # urdf: PartNetMobilityURDF = PartNetMobilityURDF(urdf_path=urdf_fn)
    # V_list, E_list = urdf.export_structure()

    # mesh_list, transformed_mesh_list, pcl_list = [], [], []
    # for v in V_list:
    #     _v, _f = v["agg_mesh"]
    #     c = v["abs_center"].copy()
    #     mesh_list.append(trimesh.Trimesh(vertices=_v, faces=_f))
    #     m = trimesh.Trimesh(vertices=_v + c, faces=_f)
    #     transformed_mesh_list.append(m)
    #     pcl_list.append(v["pcl"] + c)
    # pcl = np.concatenate(pcl_list, -0)
    # np.savetxt("./debug.txt", pcl, fmt="%6f")
    # transformed_mesh = trimesh.util.concatenate(transformed_mesh_list)
    # transformed_mesh.export("./debug.obj")
    
    def check(p):
        fn, dst = p
        name = osp.basename(fn)
        os.makedirs(dst, exist_ok=True)
        # from partnet_urdf import PartNetMobilityURDF
        # urdf_fn = "/home/ray/datasets/partnet-mobility-v0/102506/mobility.urdf"
        # urdf: PartNetMobilityURDF = PartNetMobilityURDF(urdf_path=urdf_fn)
        # V_list, E_list = urdf.export_structure()
        data = np.load(fn, allow_pickle=True)
        V_list, E_list = data["V"].tolist(), data["E"].tolist()
        V, E, v_map = compact_pack(V_list, E_list, permute=True)
        G = get_G_from_VE(V.cpu().numpy(), E.cpu().numpy())
        
        mesh_list = []
        for i in v_map:
            _v, _f = V_list[i]["agg_mesh"]
            c = V_list[i]["abs_center"].copy()
            mesh_list.append(trimesh.Trimesh(vertices=_v, faces=_f))
        G = append_mesh_to_G(G, mesh_list)
        viz_topo = viz_G_topology(G)
        imageio.imwrite(osp.join(dst, f"{name}_topo_{v_map}.png"), viz_topo)
        viz_list = viz_G(G, cam_dist=3.0, viz_frame_N=12)
        imageio.mimsave(osp.join(dst, f"{name}_viz_{v_map}.gif"), viz_list, fps=10)
        return
    
    # fn = "/home/ray/datasets/flatwhite/partnet_mobility_graph_v4/102506.npz"
    # # fn = "/home/ray/datasets/flatwhite/partnet_mobility_graph_v3/103242.npz"
    # check((fn, "./debug"))
    
    SRC = "/home/ray/datasets/flatwhite/partnet_mobility_graph_v4/"
    DST = "/home/ray/datasets/flatwhite/partnet_mobility_graph_v4_check_viz/"
    
    p_list= []
    for fn in os.listdir(SRC):
        if not fn.endswith(".npz"):
            continue
        p_list.append((osp.join(SRC, fn), DST))
    
    with Pool(12) as p:
        p.map(check, p_list)
    
