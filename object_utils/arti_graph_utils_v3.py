# v2 is
# ! by default, the list of edges represent the upper triangle, i.e. row i, col j, then i < j
# ! v3, from v2, this version support totally permuted v, and the v_mask is not always first oness

import torch
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
import numpy as np
import logging
import networkx as nx
from matplotlib import cm
from transforms3d.axangles import axangle2mat
import imageio
import torch.nn.functional as F
from copy import deepcopy


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
        # in the origin index, first num_v are object
        v_map = np.random.permutation(K).tolist()  # stores the original id
    else:
        v_map = np.arange(K).tolist()
    v_mask = [v_mask[i] for i in v_map]

    _v_bbox = [v["bbox_L"] for v in V] + [np.zeros(3)] * n_empty
    v_bbox = [_v_bbox[i] for i in v_map]
    v_bbox = torch.from_numpy(np.stack(v_bbox, axis=0)).float()
    # p_global = T_gl @ p_local
    _v_t_gl = [v["abs_center"] for v in V] + [np.zeros(3)] * n_empty
    v_t_gl = [_v_t_gl[i] for i in v_map]
    v_t_gl = torch.from_numpy(np.stack(v_t_gl, axis=0)).float()
    # ! Now assume partnet-M all init part R = I
    v_r_gl = torch.zeros(K, 3).float()
    ret_v = torch.cat([torch.LongTensor(v_mask)[..., None], v_bbox, v_r_gl, v_t_gl], -1)

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
        
        # # debug
        # print(e["r_limits"], e["p_limits"])
        
        # assert e["r_limits"][0] <= e["r_limits"][1]
        # assert e["p_limits"][0] <= e["p_limits"][1]

        e_plucker[e_list_ind] = plucker_global

    e_type = F.one_hot(e_type, num_classes=3).float()

    ret_e = torch.cat([e_type, e_plucker, e_lim], dim=1)
    # v: [mask_occ(1), bbox(3), r_gl(3), t_gl(3) | additional codes in the future]
    # e: [type(3), plucker(6), rlim(2), plim(2)]
    return ret_v, ret_e, v_map


def uppertri_list_to_mtx(trilist):
    N = len(trilist)
    K = int(np.ceil(np.sqrt(2 * N)))
    r = torch.arange(K)
    mask = r[:, None] < r
    mtx = torch.zeros(K, K, trilist.shape[1], dtype=trilist.dtype)
    mtx[mask] = trilist
    return mtx


def complete_to_autoregressive(V, E, random_walk=True):
    # after calling compact_pack, V, and E may have random order
    # convert the format to autoregressive
    # v: [mask_occ(1), bbox(3), r_gl(3), t_gl(3) | additional codes in the future]
    # e: [type(3), plucker(6), rlim(2), plim(2)]
    assert torch.is_tensor(V) and torch.is_tensor(E)
    # remove the Chirality
    # remove the compact Edge list, only have |V|-1 elements, not |V|*(|V|-1)/2
    # the returned V,E must be first compact, i.e all valid nodes are in the front
    ret_E = torch.zeros(len(V) - 1, 12, dtype=E.dtype, device=E.device)

    v_mask = V[:, 0] > 0.5
    E_mtx = uppertri_list_to_mtx(E)  # K,K,13, upper
    original_vid = torch.arange(len(V))[v_mask]  # V,10
    rest_vid = torch.arange(len(V))[~v_mask].numpy().tolist()

    K = int(v_mask.sum())  # K is the actual map length
    E_mtx = E_mtx[v_mask][:, v_mask]
    E_mtx = E_mtx + E_mtx.permute(1, 0, 2)

    E_fg = E_mtx[..., 1:3].max(-1).values > 0.5
    E_chirality = E_mtx[..., 1:3].argmax(-1)  # 0 is stay, 1 is flip

    # random walk in the graph
    v_ind_list = [np.random.randint(0, K)] if random_walk else [0]  # random select a root
    e_list = []
    for k in range(K - 1):
        # * find a parent id in the prefix list
        prefix_ind = deepcopy(v_ind_list)
        if random_walk:  # shuffle the prefix, random grow the tree
            np.random.shuffle(prefix_ind)
        for p_ind in prefix_ind:  # find the first non-zero row
            row = E_fg[p_ind]
            if row.sum() > 0.0:
                break
        assert row.sum() > 0.0

        c_ind_list = torch.where(row > 0.0)[0]
        if len(c_ind_list) == 1:
            c_ind = c_ind_list[0]
        else:
            if random_walk:
                c_ind = c_ind_list[np.random.randint(0, len(c_ind_list))]
            else:
                c_ind = c_ind_list[0]
        c_ind = int(c_ind)
        # parent is i, row;  child is j, col
        assert p_ind != c_ind

        flip = p_ind > c_ind
        if E_chirality[p_ind, c_ind] == 1:
            flip = not flip  # flip once

        # because E_mtx is already symmetric (not handling flip yet in E_mtx)
        plucker, lim = E_mtx[p_ind, c_ind, 3:9], E_mtx[p_ind, c_ind, 9:]
        if flip:
            plucker = -plucker  # ! Note, here the plucker have to
        e = torch.cat([torch.Tensor([v_ind_list.index(p_ind)]), plucker, lim])
        # e: [parent_ind(in output V) |plucker(6), rlim(2), plim(2)] len 11
        e_list.append(e)
        v_ind_list.append(c_ind)
        E_fg[p_ind, c_ind] = 0.0
        E_fg[c_ind, p_ind] = 0.0
    e_list = torch.stack(e_list, dim=0)

    # map _v_map to original index
    ret_E[: len(e_list), 1:] = e_list
    ret_E[
        : len(e_list), 0
    ] = 1.0  # [pad mask |parent_ind(in output V) |plucker(6), rlim(2), plim(2)] len 12

    v_map = [int(original_vid[i]) for i in v_ind_list]  # the output ind of the input V
    v_map = v_map + rest_vid  # ret -1 for invalid nodes
    # V should also permute to the prefix order
    ret_V = torch.stack([V[i] for i in v_map], 0)
    return ret_V, ret_E, v_map


def autoregressive_to_complete(V, E):
    # the V is not changed, stay the same order, but E from compact |V|-1 list ot |V|*(|V|-1)/2 list

    # v: [mask_occ(1), bbox(3), r_gl(3), t_gl(3) | additional codes in the future]
    # e: [e_mask, parent_id, plucker(6), rlim(2), plim(2)]

    v_mask = V[:, 0] > 0.5
    e_mask = E[:, 0] > 0.5
    nv = v_mask.sum()
    assert v_mask[:nv].all(), print(v_mask, nv)
    assert e_mask[: nv - 1].all(), print(e_mask, nv - 1)

    ret_e = torch.zeros(len(V) * (len(V) - 1) // 2, 13, dtype=E.dtype, device=E.device)
    ret_e[..., 0] = 1.0
    for i in range(nv - 1):
        e = E[i]
        parent_id, child_id = int(e[1]), i + 1  # the list is aligned with the nodes

        plucker, lim = e[2:8], e[8:]
        flip = plucker_need_flip(plucker)
        if flip:  # orient the global plucker to hemisphere
            plucker = -plucker

        if parent_id > child_id:
            src, dst = child_id, parent_id
            flip = not flip
        else:
            src, dst = parent_id, child_id
        assert src < dst

        trilist_ind = map_upper_triangle_to_list(src, dst, len(V))
        ret_e[trilist_ind, 0] = 0.0
        if flip:  # 2 is flip
            ret_e[trilist_ind, 2] = 1.0
        else:
            ret_e[trilist_ind, 1] = 1.0
        ret_e[trilist_ind, 3:] = torch.cat([plucker, lim])

    return V, ret_e


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
    # assert n_v >= 2, f"n_v={n_v}"
    original_vid = np.arange(K)[v_mask].tolist()
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
        for _i in range(K):
            for _j in range(K):
                if _i >= _j:
                    continue
                # src = i, dst = j
                ind = map_upper_triangle_to_list(_i, _j, K)
                # e: [type(3), plucker(6), rlim(2), plim(2)]
                e_type = E[ind, :3].argmax()
                if e_type == 0:
                    continue
                assert _i in original_vid and _j in original_vid, "invalid edge detected!"
                src_i, dst_j = original_vid.index(_i), original_vid.index(_j)
                e_plucker = E[ind, 3:9]
                if e_type == 2:  # flip
                    e_plucker = -e_plucker
                e_rlim, e_plim = E[ind, 9:11], E[ind, 11:13]
                T_gi, T_gj = np.eye(4), np.eye(4)
                T_gi[:3, :3] = G.nodes[src_i]["R"]
                T_gi[:3, 3] = G.nodes[src_i]["t"]
                T_gj[:3, :3] = G.nodes[dst_j]["R"]
                T_gj[:3, 3] = G.nodes[dst_j]["t"]
                T_ig = np.linalg.inv(T_gi).copy()
                T_ij = T_ig.copy() @ T_gj.copy()  # T0
                local_plucker = e_plucker.copy()
                li = T_ig[:3, :3] @ local_plucker[:3]
                mi = T_ig[:3, :3] @ local_plucker[3:] + np.cross(T_ig[:3, 3], li)
                local_plucker = np.concatenate([li, mi])
                G.add_edge(src_i, dst_j)
                nx.set_edge_attributes(
                    G,
                    {
                        (src_i, dst_j): {
                            "src": src_i,
                            "dst": dst_j,
                            "T_src_dst": T_ij.copy(),
                            "plucker": local_plucker.copy(),
                            "plim": e_plim.copy(),
                            "rlim": e_rlim.copy(),
                            # additional info
                            "global_plucker": e_plucker.copy(),  # this along with t0, R0 may be used for compute parameter space distance for evaluation
                            "T_ig": T_ig.copy(),
                            "T_gj": T_gj.copy(),
                        }
                    },
                )
    return G


def compact_pack_fixed_chirality(V, E, K=25, permute=True):
    # for rebuttal use
    if len(V) > K:
        print(f"Warning, extend {K} to {len(V)}")
        K = len(V)
    num_v = len(V)
    n_empty = K - num_v

    # Nodes
    v_mask = np.zeros(K, dtype=np.bool)
    v_mask[: len(V)] = True
    if permute:
        # in the origin index, first num_v are object
        v_map = np.random.permutation(K).tolist()  # stores the original id
    else:
        v_map = np.arange(K).tolist()
    v_mask = [v_mask[i] for i in v_map]

    _v_bbox = [v["bbox_L"] for v in V] + [np.zeros(3)] * n_empty
    v_bbox = [_v_bbox[i] for i in v_map]
    v_bbox = torch.from_numpy(np.stack(v_bbox, axis=0)).float()
    # p_global = T_gl @ p_local
    _v_t_gl = [v["abs_center"] for v in V] + [np.zeros(3)] * n_empty
    v_t_gl = [_v_t_gl[i] for i in v_map]
    v_t_gl = torch.from_numpy(np.stack(v_t_gl, axis=0)).float()
    # ! Now assume partnet-M all init part R = I
    v_r_gl = torch.zeros(K, 3).float()
    ret_v = torch.cat([torch.LongTensor(v_mask)[..., None], v_bbox, v_r_gl, v_t_gl], -1)

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
        
        # ! force to be not fliped!
        if flip:
            plucker_global = -plucker_global
        e_type[e_list_ind] = 1
        
        # if flip:  # 2 is flip plucker
        #     e_type[e_list_ind] = 2
        # else:  # 1 is not flip plucker
        #     e_type[e_list_ind] = 1

        e_lim[e_list_ind, :2] = torch.Tensor(e["r_limits"])
        e_lim[e_list_ind, 2:] = torch.Tensor(e["p_limits"])
        
        # # debug
        # print(e["r_limits"], e["p_limits"])
        
        # assert e["r_limits"][0] <= e["r_limits"][1]
        # assert e["p_limits"][0] <= e["p_limits"][1]

        e_plucker[e_list_ind] = plucker_global

    e_type = F.one_hot(e_type, num_classes=3).float()

    ret_e = torch.cat([e_type, e_plucker, e_lim], dim=1)
    # v: [mask_occ(1), bbox(3), r_gl(3), t_gl(3) | additional codes in the future]
    # e: [type(3), plucker(6), rlim(2), plim(2)]
    return ret_v, ret_e, v_map




if __name__ == "__main__":
    from arti_viz_utils import viz_G_topology, viz_G, append_mesh_to_G
    import trimesh
    import os.path as osp
    import os
    from multiprocessing import Pool

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
        # G = get_G_from_VE(V.cpu().numpy(), E.cpu().numpy())

        ag_V, ag_E, ag_v_map = complete_to_autoregressive(V, E, random_walk=False)
        check_V, check_E = autoregressive_to_complete(ag_V, ag_E)
        full_v_map = [v_map[ag_v_map[i]] for i in range(len(ag_v_map))]

        G = get_G_from_VE(check_V.cpu().numpy(), check_E.cpu().numpy())

        mesh_list = []
        # for v, v_origin_id in zip(V, v_map):
        for v, v_origin_id in zip(check_V, full_v_map):
            if v[0] == 0:
                continue
            _v, _f = V_list[v_origin_id]["agg_mesh"]
            c = V_list[v_origin_id]["abs_center"].copy()
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
    DST = "/home/ray/datasets/flatwhite/partnet_mobility_graph_v4_check_viz3_autoregressive_nonrandom/"

    p_list = []
    for fn in os.listdir(SRC):
        if not fn.endswith(".npz"):
            continue
        p_list.append((osp.join(SRC, fn), DST))
        # check(p_list[-1])

    with Pool(12) as p:
        p.map(check, p_list)
