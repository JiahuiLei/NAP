import os, sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from render_helper import render
import logging
import networkx as nx
from matplotlib import cm
from transforms3d.axangles import axangle2mat
from matplotlib import pyplot as plt
import imageio
import numpy as np


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


def append_mesh_to_G(G, mesh_list, key="mesh"):
    for v, v_data in G.nodes(data=True):
        if mesh_list[v] is None:
            continue
        nx.set_node_attributes(G, {v: {key: mesh_list[v]}})
    return G


def viz_G_topology(
    G,
    r_edge_color="tab:red",
    p_edge_color="tab:blue",
    hybrid_edge_color="tab:orange",
    node_size=800,
    title="",
    fig_size=(3, 3),
    dpi=100,
    r_range_th=3e-3,
    p_range_th=3e-3,
    show_border=True,
):
    fig = plt.figure(figsize=fig_size, dpi=dpi)
    pos = nx.kamada_kawai_layout(G)
    options = {"edgecolors": "tab:gray", "node_size": node_size, "alpha": 0.9}
    for n, n_data in G.nodes(data=True):
        nx.draw_networkx_nodes(G, pos, nodelist=[n], node_color=G.nodes[n]["color"], **options)
    nx.draw_networkx_edges(G, pos, width=2.0, alpha=1.0)
    for e0, e1, e_data in G.edges(data=True):
        r_lim, p_lim = e_data["rlim"], e_data["plim"]
        r_range = abs(r_lim[1] - r_lim[0])
        p_range = abs(p_lim[1] - p_lim[0])
        if r_range > r_range_th and p_range > p_range_th:
            edge_color = hybrid_edge_color
        elif r_range > r_range_th:
            edge_color = r_edge_color
        else:
            edge_color = p_edge_color
        nx.draw_networkx_edges(
            G, pos, edgelist=[(e0, e1)], width=8, alpha=0.7, edge_color=edge_color
        )

    plt.title(f"{title}|V|={len(G.nodes)},|E|={len(G.edges)}")
    if not show_border:
        plt.axis("off")
    ax = plt.gca()
    fig.tight_layout(pad=1.0)
    rgb = plot_to_image(fig)
    plt.close(fig)
    return rgb


def screw_to_T(theta, d, l, m):
    assert abs(np.linalg.norm(l) - 1.0) < 1e-4
    R = axangle2mat(l, theta)
    t = (np.eye(3) - R) @ (np.cross(l, m)) + l * d
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def viz_G(
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
    moving_mask=None,
    mesh_key="mesh",
    viz_box=True,
    render_flags=0,
):
    ret = []
    if len(G.nodes) >= 2 and nx.is_tree(G):  # now only support tree viz
        # * now G is connected and acyclic
        # find the root
        vid = [n for n in G.nodes]
        v_bbox = np.stack([d["bbox"] for nid, d in G.nodes(data=True)], 0)
        v_volume = v_bbox.prod(axis=-1) * 8
        root_vid = vid[v_volume.argmax()]

        # * sample a set of possible angle range for each joint
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
                        e_data = G.edges[e]
                        _T0 = e_data["T_src_dst"]
                        plucker = e_data["plucker"]
                        l, m = plucker[:3], plucker[3:]
                        plim, rlim = e_data["plim"], e_data["rlim"]
                        if moving_mask is None or moving_mask[e]:
                            theta = np.linspace(*rlim, viz_frame_N)[step]
                            d = np.linspace(*plim, viz_frame_N)[step]
                        else:  # don't move
                            theta = np.linspace(*rlim, viz_frame_N)[0]
                            d = np.linspace(*plim, viz_frame_N)[0]
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
            bbox_edge_start_list, bbox_edge_dir_list = [], []
            bbox_corner_list, bbox_color_list = [], []
            pcl_color_list = []
            # bbox_colors = cm.hsv(np.linspace(0, 1, len(node_traverse_list) + 1))[:-1]
            mesh_list, mesh_color_list = [], []
            for nid, T in zip(node_traverse_list, T_rl_list):
                bbox = G.nodes[nid]["bbox"]
                color = G.nodes[nid]["color"]
                pcl_color_list.append(color)

                if mesh_key in G.nodes[nid].keys():
                    mesh = G.nodes[nid][mesh_key].copy()
                    mesh.apply_transform(T.copy())
                    mesh_list.append(mesh)
                    mesh_color_list.append([c * 0.5 for c in color[:3]] + [0.7])
                bbox_corner = BBOX_CORNER * bbox
                bbox_corner = bbox_corner
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
                cam_dist=cam_dist,
                cam_angle_pitch=pitch,
                cam_angle_yaw=yaw,
                shape=shape,
                light_intensity=light_intensity,
                light_vertical_angle=light_vertical_angle,
                render_flags=render_flags,
            )
            if viz_box:
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
                    render_flags=render_flags,
                )
                # # debug
                # imageio.imsave("./dbg.png", rgb)
                rgb = np.concatenate([rgb0, rgb1], cat_dim)
            else:
                rgb = rgb0
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
