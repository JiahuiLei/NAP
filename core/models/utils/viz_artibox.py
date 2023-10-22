import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__)))

from pyrender_helper_new import render
import numpy as np
import networkx as nx
import imageio
from transforms3d.axangles import axangle2mat
from matplotlib import cm
from tqdm import tqdm
import torch
import logging
import matplotlib.pyplot as plt

BBOX_CORNER = np.array(
    [
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, -1.0],
    ]
)


def make_graph(V, E, edge_onehot=True):
    if V.shape[1] == 7:
        shapecode_flag = False
    elif V.shape[1] > 9:
        shapecode_flag = True
    else:
        raise RuntimeError("Invalid V shape")
    v_mask, e_mask = V[:, 0] > 0.5, E[:, 0] > 0.5
    K = len(v_mask)
    v_original_index = torch.arange(len(V))[v_mask].cpu().numpy().tolist()
    e = E[e_mask].detach().cpu().numpy()
    v = V[v_mask].detach().cpu().numpy()
    n_v, n_e = len(v), len(e)
    if not n_v == n_e + 1:
        logging.warning("Viz: the graph is not a tree!")

    # viz the topology
    G = nx.Graph()
    revolute_edges, prismatic_edges, invalid_edges = [], [], []
    if n_v >= 2:
        # unpack
        # e: [alpha :1, T1 1:13, T2 13:25, limits 25:27, axis 27:30, type 30:31, edge: 31:]
        if edge_onehot:
            e_src = np.argmax(e[:, 31 : 31 + K], axis=-1)
            e_dst = np.argmax(e[:, 31 + K : 31 + 2 * K], axis=-1)
        else:
            e_src, e_dst = e[:, 31], e[:, 32]
        e_T1 = e[:, 1:13].reshape((-1, 3, 4))
        e_T2 = e[:, 13:25].reshape((-1, 3, 4))
        e_lim, e_axis, e_type = e[:, 25:27], e[:, 27:30], e[:, 30:31]
        # binarize
        e_type = (e_type > 0.5).astype(np.int32)

        node_color_list = cm.hsv(np.linspace(0, 1, n_v + 1))[:-1]
        # Fill in VTX
        for vid in range(n_v):
            G.add_node(vid)
            if shapecode_flag:
                dummy_flag = v[vid, 1] < 0.5  # already after sigmoid
                v_data = v[vid, 2:]
                shapecode = v_data[6:]
            else:
                dummy_flag = np.allclose(v[vid, 4:], np.zeros_like(v[vid, 4:]))
                v_data = v[vid, 1:]
                shape_code = None
            nx.set_node_attributes(
                G,
                {
                    vid: {
                        "center": v_data[:3],
                        "bbox": v_data[3:6],
                        "color": node_color_list[vid],
                        "dummy": dummy_flag,
                        "shapecode": shapecode,
                    }
                },
            )
        # Fill in EDGE
        for eid in range(n_e):
            o_src, o_dst = int(e_src[eid]), int(e_dst[eid])
            invalid_flag = False
            if o_src not in v_original_index:
                o_src = v_original_index[0]
                invalid_flag = True
            if o_dst not in v_original_index:
                o_dst = v_original_index[1]
                invalid_flag = True
            src = v_original_index.index(o_src)
            dst = v_original_index.index(o_dst)
            G.add_edge(src, dst)
            nx.set_edge_attributes(
                G,
                {
                    (src, dst): {
                        "src": src,
                        "dst": dst,
                        "T1": e_T1[eid],
                        "T2": e_T2[eid],
                        "lim": e_lim[eid],
                        "axis": e_axis[eid],
                        "type": e_type[eid],
                        "type_name": "prismatic" if e_type[eid] > 0.5 else "revolute",
                        "invalid": invalid_flag,
                    }
                },
            )
    return G


def viz_articulated_graph(
    G,
    r_edge_color="tab:red",
    p_edge_color="tab:blue",
    invalid_edge_color="tab:gray",
    node_size=800,
    title="",
    fig_size=(3, 3),
    dpi=100,
):
    fig = plt.figure(figsize=fig_size, dpi=dpi)
    pos = nx.kamada_kawai_layout(G)
    options = {"edgecolors": "tab:gray", "node_size": node_size, "alpha": 0.9}
    for n, n_data in G.nodes(data=True):
        if n_data["dummy"]:
            nx.draw_networkx_nodes(G, pos, nodelist=[n], node_color="tab:gray", **options)
        else:
            nx.draw_networkx_nodes(G, pos, nodelist=[n], node_color=G.nodes[n]["color"], **options)

    nx.draw_networkx_edges(G, pos, width=2.0, alpha=1.0)
    invalid_edge_cnt = 0
    for e0, e1, e_data in G.edges(data=True):
        if e_data["invalid"]:
            edge_color = invalid_edge_color
            invalid_edge_cnt += 1
        elif e_data["type_name"] == "prismatic":
            edge_color = p_edge_color
        else:
            edge_color = r_edge_color
        nx.draw_networkx_edges(
            G, pos, edgelist=[(e0, e1)], width=8, alpha=0.7, edge_color=edge_color
        )

    plt.title(f"{title}|V|={len(G.nodes)},|E|={len(G.edges)},Invalid-E={invalid_edge_cnt}")
    ax = plt.gca()
    fig.tight_layout(pad=1.0)
    rgb = plot_to_image(fig)
    plt.close(fig)
    return rgb


def viz_articulated_box(
    G,
    bbox_thickness=0.03,
    bbox_alpha=0.6,
    viz_frame_N=16,
    cam_dist=4.0,
    pitch=np.pi / 4.0,
    yaw=np.pi / 4.0,
    shape=(480, 480),
    light_intensity=1.0,
    light_vertical_angle=-np.pi / 3.0,
    cat_dim=1,
):
    # ! Now plot a 3D viz_color
    ret = []
    if len(G.nodes) >= 2 and nx.is_tree(G):  # now only support tree viz
        # * now G is connected and acyclic
        # find the root
        vid = [n for n in G.nodes]
        v_bbox = np.stack([d["bbox"] for nid, d in G.nodes(data=True)], 0)
        v_volume = v_bbox.prod(axis=-1) * 8
        root_vid = vid[v_volume.argmax()]

        # * sample a set of possible angle range for each joint
        # for step in tqdm(range(viz_frame_N)):
        for step in range(viz_frame_N):
            node_traverse_list = [n for n in nx.dfs_preorder_nodes(G, root_vid)]
            T_rl_list = [np.eye(4)]  # p_root = T_rl @ p_link
            # * prepare the node pos
            for i in range(len(node_traverse_list) - 1):
                # ! find the parent!
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
                        T_src_j1, T_j2_dst = G.edges[e]["T1"], G.edges[e]["T2"]
                        T_src_j1 = np.concatenate([T_src_j1, np.array([[0, 0, 0, 1]])], 0)
                        T_j2_dst = np.concatenate([T_j2_dst, np.array([[0, 0, 0, 1]])], 0)
                        axis, lim, type = (
                            G.edges[e]["axis"],
                            G.edges[e]["lim"],
                            G.edges[e]["type"],
                        )
                        # ! note, now use a liner
                        joint_param = np.linspace(*lim, viz_frame_N)[step]
                        if type == 0:  # revolute
                            R_j1_j2 = axangle2mat(axis, joint_param)
                            T_j1_j2 = np.zeros((3, 4))
                            T_j1_j2[:3, :3] = R_j1_j2
                        elif type == 1:  # prismatic
                            T_j1_j2 = np.zeros((3, 4))
                            T_j1_j2[:3, :3] = np.eye(3)
                            T_j1_j2[:3, 3] = axis * joint_param
                        else:
                            raise NotImplementedError()
                        T_j1_j2 = np.concatenate([T_j1_j2, np.array([[0, 0, 0, 1]])], 0)
                        T_src_dst = T_src_j1 @ T_j1_j2 @ T_j2_dst
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
            bbox_edge_start_list, bbox_edge_dir_list = [], []
            bbox_corner_list, bbox_color_list = [], []
            pcl_color_list = []
            # bbox_colors = cm.hsv(np.linspace(0, 1, len(node_traverse_list) + 1))[:-1]
            mesh_list, mesh_color_list = [], []
            for nid, T in zip(node_traverse_list, T_rl_list):
                bbox, center = G.nodes[nid]["bbox"], G.nodes[nid]["center"]
                color = G.nodes[nid]["color"]
                pcl_color_list.append(color)

                if "mesh" in G.nodes[nid].keys():
                    mesh = G.nodes[nid]["mesh"].copy()
                    mesh.apply_transform(T.copy())
                    mesh_list.append(mesh)
                    mesh_color_list.append([c * 0.5 for c in color[:3]] + [0.7])
                bbox_corner = BBOX_CORNER * bbox
                bbox_corner = bbox_corner + center[None, :]
                bbox_corner = bbox_corner @ T[:3, :3].T + T[:3, 3]
                bbox_corner_list.append(bbox_corner)
                bbox_edge_start_ind = [0, 0, 2, 1, 3, 3, 5, 6, 0, 1, 4, 2]
                bbox_edge_end_ind = [1, 2, 4, 4, 6, 5, 7, 7, 3, 6, 7, 5]
                bbox_start = bbox_corner[bbox_edge_start_ind]
                bbox_end = bbox_corner[bbox_edge_end_ind]
                bbox_edge_start_list.append(bbox_start)
                bbox_edge_dir_list.append(bbox_end - bbox_start)
                bbox_color = color
                bbox_color[-1] = bbox_alpha
                bbox_color_list.append(np.tile(bbox_color[None, :], [12, 1]))

            if len(bbox_corner_list) > 0:
                bbox_color_list = np.concatenate(bbox_color_list, 0)
                bbox_edge_start_list = np.concatenate(bbox_edge_start_list, 0)
                bbox_edge_dir_list = np.concatenate(bbox_edge_dir_list, 0)
                arrow_tuples = (bbox_edge_start_list, bbox_edge_dir_list)
            else:
                bbox_color_list, bbox_edge_start_list, bbox_edge_dir_list = None, None, None
                arrow_tuples = None

            rgb0 = render(
                mesh_list=mesh_list,
                mesh_color_list=mesh_color_list,
                # pcl_list=bbox_corner_list,
                # pcl_color_list=pcl_color_list,
                # pcl_radius_list=[bbox_thickness * 2.0] * len(bbox_corner_list),
                # # arrows
                # arrow_head=False,
                # arrow_tuples=arrow_tuples,
                # arrow_colors=bbox_color_list,
                # arrow_radius=bbox_thickness,
                cam_dist=cam_dist,
                cam_angle_pitch=pitch,
                cam_angle_yaw=yaw,
                shape=shape,
                light_intensity=light_intensity,
                light_vertical_angle=light_vertical_angle,
            )
            rgb1 = render(
                pcl_list=bbox_corner_list,
                pcl_color_list=pcl_color_list,
                pcl_radius_list=[bbox_thickness * 2.0] * len(bbox_corner_list),
                # arrows
                arrow_head=False,
                arrow_tuples=arrow_tuples,
                arrow_colors=bbox_color_list,
                arrow_radius=bbox_thickness,
                cam_dist=cam_dist,
                cam_angle_pitch=pitch,
                cam_angle_yaw=yaw,
                shape=shape,
                light_intensity=light_intensity,
                light_vertical_angle=light_vertical_angle,
            )
            # # debug
            # imageio.imsave("./dbg.png", rgb)
            rgb = np.concatenate([rgb0, rgb1], cat_dim)
            ret.append(rgb)
    else:
        dummy = np.ones((shape[0], shape[1] * 2, 3), dtype=np.uint8) * 127
        ret = [dummy] * viz_frame_N
    # imageio.mimsave("./debug/dbg.gif", ret, fps=10)
    ret = ret + ret[::-1]
    return ret


def plot_to_image(fig):
    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image_from_plot


if __name__ == "__main__":
    import pickle

    with open("./G.pkl", "rb") as f:
        G = pickle.load(f)
    viz_articulated_box(G)

    print()
