import torch
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
import numpy as np
import logging
import networkx as nx
from matplotlib import cm
from transforms3d.axangles import axangle2mat
import imageio


def compact_pack(V, E, K=25, mode="e0", return_center=False):
    if len(V) > K:
        print(f"Warning, extend {K} to {len(V)}")
        K = len(V)
    n_empty = K - len(V)
    v_mask, e_mask = np.zeros(K, dtype=np.bool), np.zeros(K - 1, dtype=np.bool)
    v_mask[: len(V)] = True
    e_mask[: len(V) - 1] = True

    v_bbox = [v["bbox_L"] for v in V] + [np.zeros(3)] * n_empty
    v_bbox = np.stack(v_bbox, axis=0)
    ret_v = torch.cat([torch.from_numpy(v_mask).unsqueeze(1), torch.from_numpy(v_bbox)], dim=1)

    e_src = [e[mode]["src_ind"] for e in E] + [0] * n_empty
    e_dst = [e[mode]["dst_ind"] for e in E] + [0] * n_empty
    e_plucker = [e[mode]["plucker"] for e in E] + [np.zeros(6)] * n_empty
    e_T0 = [e[mode]["T0"] for e in E] + [np.eye(4)] * n_empty
    e_rlim = [e["r_limits"] for e in E] + [np.zeros(2)] * n_empty
    e_plim = [e["p_limits"] for e in E] + [np.zeros(2)] * n_empty

    e_src = torch.from_numpy(np.stack(e_src, axis=0))
    e_dst = torch.from_numpy(np.stack(e_dst, axis=0))
    e_plucker = torch.from_numpy(np.stack(e_plucker, axis=0))
    e_rlim = torch.from_numpy(np.stack(e_rlim, axis=0))
    e_plim = torch.from_numpy(np.stack(e_plim, axis=0))

    e_T0 = torch.from_numpy(np.stack(e_T0, axis=0))
    e_r0 = matrix_to_axis_angle(e_T0[:, :3, :3])
    e_t0 = e_T0[:, :3, 3]

    e_mask = torch.from_numpy(e_mask).unsqueeze(1)
    ret_e = torch.cat(
        [e_mask, e_src.unsqueeze(1), e_dst.unsqueeze(1), e_r0, e_t0, e_plucker, e_rlim, e_plim],
        dim=1,
    )
    # v: [mask_occ, bbox, | additional codes in the future]
    # e: [mask_occ, src, dst, r0, t0, plucker, rlim, plim]
    if return_center:
        v_abs_center = [v["abs_center"] for v in V] + [np.zeros(3)] * n_empty
        v_abs_center = torch.from_numpy(np.stack(v_abs_center, axis=0))
        return ret_v, ret_e, v_abs_center
    else:
        return ret_v, ret_e


def get_G_from_VE(V, E):
    # v: [mask_occ, bbox, | additional codes in the future]
    # e: [mask_occ, src, dst, r0, t0, plucker, rlim, plim]
    v_mask, e_mask = V[:, 0] > 0.5, E[:, 0] > 0.5
    K = len(v_mask)
    v_original_index = torch.arange(len(V))[v_mask].cpu().numpy().tolist()
    e = E[e_mask].detach().cpu().numpy()
    v = V[v_mask].detach().cpu().numpy()
    n_v, n_e = len(v), len(e)
    if not n_v == n_e + 1:
        logging.warning("Viz: the graph is not a tree!")
    G = nx.Graph()
    if n_v >= 2:
        node_color_list = cm.hsv(np.linspace(0, 1, n_v + 1))[:-1]
        # Fill in VTX
        for vid in range(n_v):
            G.add_node(vid)
            v_attr = {"bbox": v[vid][1:4], "color": node_color_list[vid]}
            if v.shape[1] > 4:
                v_attr["additional"] = v[vid][4:]
            nx.set_node_attributes(G, {vid: v_attr})
        # Fill in EDGE
        e_src, e_dst = e[:, 1], e[:, 2]
        e_r, e_t = e[:, 3:6], e[:, 6:9]
        e_plucker = e[:, 9:15]
        e_rlim, e_plim = e[:, 15:17], e[:, 17:19]
        for eid in range(n_e):
            o_src, o_dst = int(e_src[eid]), int(e_dst[eid])
            src = v_original_index.index(o_src)
            dst = v_original_index.index(o_dst)
            G.add_edge(src, dst)
            if np.allclose(e_r[eid], 0):
                R = np.eye(3)
            else:
                R = axangle2mat(
                    e_r[eid] / (np.linalg.norm(e_r[eid]) + 1e-6), np.linalg.norm(e_r[eid])
                )
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = e_t[eid]
            nx.set_edge_attributes(
                G,
                {
                    (src, dst): {
                        "src": src,
                        "dst": dst,
                        "T_src_dst": T,
                        "plucker": e_plucker[eid],
                        "plim": e_plim[eid],
                        "rlim": e_rlim[eid],
                    }
                },
            )
    return G

if __name__ == "__main__":
    from arti_viz_utils import viz_G_topology, viz_G, append_mesh_to_G
    import trimesh
    
    # fn = "/home/ray/datasets/flatwhite/partnet_mobility_graph_v3/102506.npz"
    # # fn = "/home/ray/datasets/flatwhite/partnet_mobility_graph_v3/103242.npz"
    # data = np.load(fn, allow_pickle=True)

    # V_list, E_list = data["V"].tolist(), data["E"].tolist()
    
    from arti_viz_utils import viz_G_topology, viz_G, append_mesh_to_G
    import trimesh
    
    from partnet_urdf import PartNetMobilityURDF
    
    urdf_fn = "/home/ray/datasets/partnet-mobility-v0/102506/mobility.urdf"
    urdf: PartNetMobilityURDF = PartNetMobilityURDF(urdf_path=urdf_fn)
    V_list, E_list = urdf.export_structure()
    
    mesh_list, transformed_mesh_list = [], []
    for v in V_list:
        _v, _f = v["agg_mesh"]
        c = v["abs_center"]
        mesh_list.append(trimesh.Trimesh(vertices=_v, faces=_f))
        m = trimesh.Trimesh(vertices=_v+c, faces=_f)
        transformed_mesh_list.append(m)
    transformed_mesh = trimesh.util.concatenate(transformed_mesh_list)
    transformed_mesh.export("./debug.obj")
        
    V, E = compact_pack(V_list, E_list)

    G = get_G_from_VE(V, E)
    
    rgb = viz_G_topology(G)
    
    G = append_mesh_to_G(G, mesh_list)
    imageio.imwrite("./debug.png", rgb)
    viz_list = viz_G(G, cam_dist=3.0, viz_frame_N=10)
    imageio.mimsave("./debug.gif", viz_list + viz_list[::-1], fps=10)
    imageio.imsave("./debug0.png", viz_list[0])
    
    print()