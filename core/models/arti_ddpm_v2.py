# Naive DDPM on whole graph
# From v1, v2 let the number of nodes also be sampled, not predefined as input

from .model_base import ModelBase
import torch
import copy
import trimesh
from torch import nn

import torch.nn.functional as F

import time
import logging
from .utils.misc import cfg_with_default, count_param
import numpy as np

import matplotlib.pyplot as plt
import gc
from matplotlib import cm
from tqdm import tqdm
import matplotlib.pyplot as plt
import imageio
from .utils.viz_artibox import *
from core.lib.implicit_func.onet_decoder import DecoderCBatchNorm, Decoder, DecoderSIREN
from torch import distributions as dist
from .utils.occnet_utils import get_generator as get_mc_extractor
from sympy.combinatorics.prufer import Prufer
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from core.models.utils.mst import Graph as MSTGraph
from scipy.sparse.csgraph import minimum_spanning_tree


from object_utils.arti_graph_utils_v3 import get_G_from_VE
from object_utils.arti_viz_utils import *

from core.lib.lib_arti.articualted_denoiser_v5 import (
    GraphDenoiseConvV5,
    GraphDenoiseConvV51,
    GraphDenoiseConvV52,
)
from core.lib.lib_arti.articualted_denoiser_v6 import GraphDenoiseConvV60
from core.lib.lib_arti.articualted_denoiser_v6_2 import GraphDenoiseConvV62
from core.lib.lib_arti.articualted_denoiser_naive import NaiveDenoiser

LARGE = 1e6
EPS = 1e-6


class Model(ModelBase):
    def __init__(self, cfg):
        network = ArticualtedDDPM(cfg)
        super().__init__(cfg, network)
        self.output_specs = {
            "metric": ["batch_loss", "loss_e", "loss_v"]
            + [
                "loss_v_occ",
                "loss_v_bbox",
                "loss_v_center",
                "loss_v_shape",
                "loss_e_type",
                "loss_e_plucker",
                "loss_e_lim",
            ],
            "image": ["viz_compare", "gen_viz", "viz_gt"],
            "video": ["gen_viz_vid"],  # ["viz_gt_vid", "gen_viz_vid"],
            "mesh": ["input"],
            "hist": ["loss_i", "loss_e_i", "loss_v_i"],
            "xls": [],
        }
        self.viz_one = cfg["logging"]["viz_one_per_batch"]
        self.iou_threshold = cfg["evaluation"]["iou_threshold"]
        self.viz_dpi = cfg_with_default(cfg, ["logging", "viz_dpi"], 200)
        self.viz_frame_N = cfg_with_default(cfg, ["logging", "viz_frame"], 10)
        self.mesh_extractor = get_mc_extractor(cfg)

        # prepare for generation
        self.N_gen = cfg_with_default(cfg, ["generation", "N_gen"], 10)
        self.K = cfg["dataset"]["max_K"]
        return

    def generate_mesh(self, embedding):
        net = self.network.module if self.__dataparallel_flag__ else self.network
        net.network_dict["sdf_decoder"].eval()
        mesh = self.mesh_extractor.generate_from_latent(c=embedding, F=net.decode_sdf)
        if mesh.vertices.shape[0] == 0:
            mesh = trimesh.primitives.Sphere(radius=1e-6)
            logging.warning("Mesh extraction fail, replace by a tiny place holder")
        return mesh

    def extract_mesh_for_G(self, G, gt_mesh_list=None):
        logging.info("Extract mesh for G ...")
        # for v, v_data in tqdm(G.nodes(data=True)):
        for v, v_data in G.nodes(data=True):
            if "additional" not in v_data:
                continue
            z = v_data["additional"][None, :]
            mesh = self.generate_mesh(torch.from_numpy(z).cuda())
            mesh_centroid = mesh.bounds.mean(0)
            mesh.apply_translation(-mesh_centroid)
            bbox = v_data["bbox"].copy()
            scale = 2.0 * np.linalg.norm(bbox) / np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])
            mesh.apply_scale(scale)
            # v_data["mesh"] = mesh
            nx.set_node_attributes(G, {v: {"mesh": mesh}})
            if gt_mesh_list is not None:
                gt_mesh = gt_mesh_list[v].copy()
                mesh_centroid = gt_mesh.bounds.mean(0)
                gt_mesh.apply_translation(-mesh_centroid)
                scale = (
                    2.0
                    * np.linalg.norm(bbox)
                    / np.linalg.norm(gt_mesh.bounds[1] - gt_mesh.bounds[0])
                )
                gt_mesh.apply_scale(scale)
                nx.set_node_attributes(G, {v: {"gt_mesh": gt_mesh}})
        return G

    def generate_object(self, V_scale, E_scale):
        net = self.network.module if self.__dataparallel_flag__ else self.network
        noise_V = torch.randn(self.N_gen, self.K, 1 + 6 + net.shapecode_dim).cuda()
        noise_E = torch.randn(self.N_gen, (self.K * (self.K - 1)) // 2, 13).cuda()
        if net.use_hard_v_mask:
            random_n = torch.randint(low=2, high=self.K + 1, size=(self.N_gen,))
            V_mask = torch.arange(self.K)[None, :] < random_n[:, None]
            V_mask = V_mask.float().to(noise_V.device)
            noise_V[..., 0] = V_mask  # set the noisy first channel to gt v_mask
        else:
            V_mask = None
        V, E = net.generate(noise_V, noise_E, V_scale, E_scale, V_mask=V_mask)
        return V, E

    def _postprocess_after_optim(self, batch):
        if batch["viz_flag"]:
            is_training = self.network.training
            self.network.eval()
            net = self.network.module if self.__dataparallel_flag__ else self.network

            viz_gt_list, viz_pred_list = [], []
            viz_gt_vid_list, viz_pred_vid_list = [], []

            # only viz one gt
            rgb_gt, rgb_gt_list = self.viz_graph(batch["V_gt"][0], batch["E_gt"][0], title="gt")
            viz_gt_list.append(rgb_gt_list[0].transpose(2, 0, 1))
            viz_gt_vid_list.append(np.stack(rgb_gt_list, 0).transpose(0, 3, 1, 2))

            # viz gen
            gen_V, gen_E = self.generate_object(batch["V_scale"], batch["E_scale"])
            B = len(gen_V)
            if self.viz_one:
                _iter = np.random.permutation(B)
            else:
                _iter = range(B)
            for bid in _iter:
                bid = int(bid)
                rgb_pred, rgb_pred_list = self.viz_graph(gen_V[bid], gen_E[bid], title="pred")
                viz_pred_list.append(rgb_pred_list[0].transpose(2, 0, 1))
                viz_pred_vid_list.append(np.stack(rgb_pred_list, 0).transpose(0, 3, 1, 2))
                if self.viz_one:
                    break
            if len(viz_gt_list) > 0:
                batch["viz_gt"] = torch.Tensor(np.stack(viz_gt_list, axis=0))  # B,3,H,W
                batch["viz_gt_vid"] = torch.Tensor(np.stack(viz_gt_vid_list, axis=0))  # B,3,H,W
                # batch["viz_vid"] = torch.cat([batch["viz_gt_vid"], batch["viz_pred_vid"]], 3)
            if len(viz_pred_list) > 0:
                batch["gen_viz"] = torch.Tensor(np.stack(viz_pred_list, axis=0))  # B,3,H,W
                batch["gen_viz_vid"] = torch.Tensor(np.stack(viz_pred_vid_list, axis=0))  # B,3,H,W
            if is_training:
                self.network.train()
        return batch

    def viz_graph(
        self,
        V,
        E,
        fig_size=(3, 3),
        title="",
        horizon=True,
        cam_dist=4.0,
        n_frames=None,
        moving_eid=None,
        gt_mesh_list=None,
    ):
        cat_dim = 1 if horizon else 0
        G = get_G_from_VE(V, E)
        G = self.extract_mesh_for_G(G, gt_mesh_list)
        viz_graph = viz_G_topology(G, title=title, show_border=False)
        render_shape = (self.viz_dpi * fig_size[0], self.viz_dpi * fig_size[1])
        if n_frames is None:
            n_frames = self.viz_frame_N
        if moving_eid is not None:
            assert isinstance(moving_eid, int)
            moving_mask = {}
            for cnt, e in enumerate(G.edges):
                if cnt == moving_eid:
                    moving_mask[e] = True
                else:
                    moving_mask[e] = False
        else:
            moving_mask = None
        render_list = viz_G(
            G,
            shape=render_shape,
            cam_dist=cam_dist,
            viz_frame_N=n_frames,
            cat_dim=cat_dim,
            moving_mask=moving_mask,
        )
        render_list = [np.concatenate([viz_graph, gif], axis=cat_dim) for gif in render_list]
        if gt_mesh_list is not None:
            gt_render_list = viz_G(
                G,
                shape=render_shape,
                cam_dist=cam_dist,
                viz_frame_N=n_frames,
                cat_dim=cat_dim,
                moving_mask=moving_mask,
                mesh_key="gt_mesh",
                viz_box=False,
                render_flags=1024,  # skip back cull
            )
            render_list = [
                np.concatenate([f1, f2], axis=cat_dim)
                for f1, f2 in zip(render_list, gt_render_list)
            ]
        # imageio.mimsave("./debug/dbg.gif", rgb_list, fps=10)
        return viz_graph, render_list


class ArticualtedDDPM(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = copy.deepcopy(cfg)

        self.max_K = cfg["dataset"]["max_K"]

        self.plucker_mode = cfg_with_default(cfg, ["model", "plucker_mode"], "6d")
        assert self.plucker_mode in ["6d"], NotImplementedError()

        self.network_dict = torch.nn.ModuleDict()
        self.init_sdf_decoder(cfg)
        self.init_diffusion(cfg)
        denoiser_type = cfg_with_default(cfg, ["model", "denoiser_type"], "v5")
        denoiser_class = {
            "v5": GraphDenoiseConvV5,
            "v5.1": GraphDenoiseConvV51,
            "v5.2": GraphDenoiseConvV52,
            "v60": GraphDenoiseConvV60,"v62": GraphDenoiseConvV62,
            "naive": NaiveDenoiser,
        }[denoiser_type]
        self.network_dict["denoiser"] = denoiser_class(
            K=self.max_K,
            M=self.M,
            **cfg["model"]["denoiser"],
        )
        # loss weight
        t_cfg = cfg["training"]
        self.use_saperate_loss = cfg_with_default(t_cfg, ["use_saperate_loss"], False)
        count_param(self.network_dict)

        self.use_hard_v_mask = cfg_with_default(cfg, ["model", "use_hard_v_mask"], False)
        if self.use_hard_v_mask:
            logging.warning("use_hard_v_mask is True")
        
        # rebuttal, 2023.8.6, for check if we force chirality = +1, will the performance be worse?
        self.force_chirality = cfg_with_default(cfg, ["model", "force_chirality"], False)
        if self.force_chirality:
            logging.warning("force_chirality is True")
        return

    def init_diffusion(self, cfg):
        d_cfg = cfg["model"]["diffusion_config"]
        self.M = d_cfg["M"]
        self.scheduling = cfg_with_default(d_cfg, ["scheduling"], "linear")
        if self.scheduling == "cosine":
            betas = betas_for_alpha_bar(
                self.M,
                lambda t: np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2,
            )
            betas = torch.from_numpy(betas).float()
        elif self.scheduling == "linear":
            # following the old version
            self.beta_min, self.beta_max = d_cfg["beta_min"], d_cfg["beta_max"]
            betas = torch.linspace(self.beta_min, self.beta_max, self.M)
        else:
            raise NotImplementedError()
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", 1.0 - betas)
        self.register_buffer(
            "alpha_bars",
            torch.tensor([torch.prod(self.alphas[: i + 1]) for i in range(len(self.alphas))]),
        )

        self.N_t_training = cfg_with_default(cfg, ["training", "N_t_training"], 1)
        return

    def init_sdf_decoder(self, cfg):
        _cfg = cfg["model"]["part_shape_prior"]
        self.N_pe = cfg_with_default(_cfg, ["N_pe"], 0)
        if self.N_pe > 0:
            self.freq = 2 ** torch.Tensor([i for i in range(self.N_pe)])
        self.sdf2occ_factor = cfg_with_default(_cfg, ["sdf2occ_factor"], -1.0)
        sdf_decoder_type = cfg_with_default(_cfg, ["sdf_decoder_type"], "decoder")
        sdf_decoder_class = {
            "decoder": Decoder,
            "cbatchnorm": DecoderCBatchNorm,
            "decoder_siren": DecoderSIREN,
        }[sdf_decoder_type]
        self.network_dict["sdf_decoder"] = sdf_decoder_class(**_cfg["sdf_decoder"])
        self.shapecode_dim = _cfg["sdf_decoder"]["c_dim"]
        ckpt_fn = _cfg["pretrained_shapeprior_path"]
        ckpt = torch.load(ckpt_fn, map_location="cpu")["model_state_dict"]
        sdf_decoder_weights = {}
        for k, v in ckpt.items():
            if "decoder" in k:
                new_k = ".".join(k.split(".")[2:])
                sdf_decoder_weights[new_k] = v
        self.network_dict["sdf_decoder"].load_state_dict(sdf_decoder_weights, strict=True)
        logging.info("loaded sdf decoder weights from %s", ckpt_fn)
        return

    @torch.no_grad()
    def extract_tree_from_mst_mtx(self, mst_mtx, v_mask):
        device = mst_mtx.device
        B = mst_mtx.shape[0]
        binary_mask_list = []
        for bid in range(B):
            _v_mask = v_mask[bid] > 0
            sub_mtx = -mst_mtx[bid, _v_mask][:, _v_mask]  # minus
            sub_mtx = sub_mtx - sub_mtx.min() + 1.0
            nv = sub_mtx.shape[0]
            sub_mtx = sub_mtx.cpu() * (1.0 - torch.eye(nv))
            Tcsr = minimum_spanning_tree(sub_mtx).toarray()
            Tcsr = (Tcsr > 1e-8).astype(np.float32)
            assert Tcsr.sum() == nv - 1
            full_mtx = np.zeros((self.max_K * self.max_K))
            _mtx_mask = _v_mask[:, None] * _v_mask[None, :]
            full_mtx[_mtx_mask.cpu().numpy().reshape(-1)] = Tcsr.reshape(-1)
            full_mtx = full_mtx.reshape(self.max_K, self.max_K)
            binary_mask_list.append(torch.from_numpy(full_mtx))
        binary_mask_list = torch.stack(binary_mask_list, 0).to(device)
        # sometimes, the Tcsr will set lower triangle to 1
        binary_mask_list = binary_mask_list + binary_mask_list.permute(0, 2, 1)
        binary_mask_list = binary_mask_list > 0
        return binary_mask_list.float()

    def project_to_plucker(self, e_attr_pred):
        # ! warning, for now will always normalize, nomatter whether the edge is valid, i.e. when the edge should have a line at infinity
        # e: [type(3), plucker(6), rlim(2), plim(2)]
        l_pred, m_pred = e_attr_pred[..., 3:6], e_attr_pred[..., 6:9]
        l_pred = F.normalize(l_pred, dim=-1)
        _inner = (l_pred * m_pred).sum(-1, keepdim=True)
        m_pred = m_pred - l_pred * _inner
        e_attr_pred = torch.cat([e_attr_pred[..., :3], l_pred, m_pred, e_attr_pred[..., 9:]], -1)
        return e_attr_pred

    def make_sure_two_fg_nodes(self, v_mask):
        # v_mask: B,K
        top2ind = torch.topk(v_mask, 2, dim=-1)[1]
        v_mask = torch.scatter(
            input=v_mask, dim=1, index=top2ind, src=torch.ones_like(top2ind).float()
        )
        v_mask = v_mask > 0.5
        assert v_mask.sum(-1).min() >= 2
        return v_mask.float()

    @torch.no_grad()
    def masked_generate(
        self,
        noise_V,
        noise_E,
        V_scale,
        E_scale,
        V_known=None,
        V_update_mask=None,
        E_known=None,
        E_update_mask=None,
        E_update_projection_fn=None,
    ):
        # v_mask is used as padded batch, the number of nodes is decided during sampling
        # noise_V: B,K,1+6+C_shapecode; noise_E: B, K(K-1)/2,13; v_mask: B,K,1
        # shapecode_std: C_shapecode
        # update mask: 1.0 is updatable, 0.0 should be fixed condition

        # shape check
        B, K, _ = noise_V.shape
        assert K == self.max_K, "TODO: support larger K for inference"
        assert noise_E.shape[0] == B and noise_E.shape[1] == K * (K - 1) // 2

        if V_known is not None:
            assert V_known.shape == noise_V.shape
            assert V_update_mask is not None
            assert len(V_update_mask.unique()) <= 2
        else:
            V_known = torch.zeros_like(noise_V)
            V_update_mask = torch.ones_like(noise_V)

        if E_known is not None:
            assert E_known.shape == noise_E.shape
            assert E_update_mask is not None
            assert len(E_update_mask.unique()) <= 2
        else:
            E_known = torch.zeros_like(noise_E)
            E_update_mask = torch.ones_like(noise_E)

        E, V = noise_E.clone(), noise_V.clone()
        logging.info(f"Diffusion Generation with {self.M} steps...")
        for idx, t in tqdm(enumerate(list(range(self.M))[::-1])):
            t_pad = torch.ones(E.shape[0]).cuda().squeeze() * t
            t_pad = t_pad.long()
            # TODO: to be safe, here better to mask also mask
            eta_V, eta_E = self.network_dict["denoiser"](V, E, t_pad)
            if t_pad.ndim == 0:
                t_pad = t_pad[None]
            alpha_t = self.alphas[t_pad][:, None, None]
            alpha_t_bar = self.alpha_bars[t_pad][:, None, None]
            # Partially denoising the image

            V_c = alpha_t_bar.sqrt() * V_known + (1 - alpha_t_bar).sqrt() * noise_V
            V_u = (1 / alpha_t.sqrt()) * (V - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_V)
            V = V_u * V_update_mask + V_c * (1 - V_update_mask)

            E_c = alpha_t_bar.sqrt() * E_known + (1 - alpha_t_bar).sqrt() * noise_E
            E_u = (1 / alpha_t.sqrt()) * (E - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_E)
            E = E_u * E_update_mask + E_c * (1 - E_update_mask)
            if E_update_projection_fn is not None:
                # ! This is for querying local part-joint pair, the joint wants to stay in the local frame of the parent part
                E = E_update_projection_fn(t, V, E, alpha_t_bar)

            # print(abs(eta_V).max(), abs(eta_E).max())
            if t > 0:
                zV = torch.randn_like(V)
                z_E = torch.randn_like(E)
                # Option 1: sigma_t squared = beta_t
                beta_t = self.betas[t_pad]
                sigma_t = beta_t.sqrt()[:, None, None]
                V = V + (sigma_t * zV) * V_update_mask
                E = E + (sigma_t * z_E) * E_update_mask
            else:
                if V_known is not None:
                    V = V * V_update_mask + V_known * (1 - V_update_mask)
                if E_known is not None:
                    E = E * E_update_mask + E_known * (1 - E_update_mask)

        # scale back
        # V[:, :, 7:] = V[:, :, 7:] * shapecode_std[None, None, :]
        # ! now assume the R0 = I, append this back to the V
        V = torch.cat([V[..., :4], torch.zeros_like(V[..., 1:4]), V[..., 4:]], -1)
        V = V * V_scale[None, None, :]
        E = E * E_scale[None, None, :]
        confirmed_v_mask = self.make_sure_two_fg_nodes(V[..., 0])
        V[..., 0] = confirmed_v_mask
        # extract the graph
        E_type = E[..., :3]
        E_value = (
            E_type[..., 1:].max(-1).values - E_type[..., 0]
        )  # how 1,2 edge is larger than the prob to be the empty edge
        E_mtx = self.network_dict["denoiser"].scatter_trilist_to_mtx(E_value[..., None])
        E_mtx = E_mtx.squeeze(-1)
        E_mtx = self.extract_tree_from_mst_mtx(E_mtx, confirmed_v_mask)
        gather_ind = self.network_dict["denoiser"].tri_ind_to_full_ind.clone()
        E_mtx = E_mtx.reshape(B, -1)
        E_fg = torch.gather(E_mtx, 1, gather_ind[None, :].expand(B, -1).to(E_mtx.device))
        # E_fg.sum(-1)
        E_fg = E_fg > 0
        final_E_type = torch.zeros_like(E_type[..., 0]).long()
        if self.force_chirality:
            raise NotImplementedError("Not debugged, should continue here")
            final_E_type[E_fg] = torch.ones_like(E_type[..., 1:].argmax(-1)[E_fg] + 1)
        else:
            final_E_type[E_fg] = E_type[..., 1:].argmax(-1)[E_fg] + 1
        final_E_type = F.one_hot(final_E_type, num_classes=3).float()

        # project the plucker to valid plucker
        E = self.project_to_plucker(E)
        # pack output
        ret_E = torch.cat([final_E_type, E[..., 3:]], -1)
        return V, ret_E

    @torch.no_grad()
    def generate(self, noise_V, noise_E, V_scale, E_scale, V_mask=None, V_known=None):
        # v_mask is used as padded batch, the number of nodes is decided during sampling
        # noise_V: B,K,1+6+C_shapecode; noise_E: B, K(K-1)/2,13; v_mask: B,K,1
        # shapecode_std: C_shapecode

        # shape check
        B, K, _ = noise_V.shape
        assert K == self.max_K, "TODO: support larger K for inference"
        assert noise_E.shape[0] == B and noise_E.shape[1] == K * (K - 1) // 2

        v_condition_flag = False
        if V_known is not None:
            v_condition_flag = True
            logging.info("Experimental: conditional on V")
            assert (
                V_known.shape == noise_V.shape
            ), f"V_known shape {V_known.shape} != noise {noise_V.shape}"

        E, V = noise_E.clone(), noise_V.clone()
        if V_mask is not None:
            assert (noise_V[..., 0] == V_mask).all()
        logging.info(f"Diffusion Generation with {self.M} steps...")
        for idx, t in tqdm(enumerate(list(range(self.M))[::-1])):
            t_pad = torch.ones(E.shape[0]).cuda().squeeze() * t
            t_pad = t_pad.long()
            # TODO: to be safe, here better to mask also mask
            eta_V, eta_E = self.network_dict["denoiser"](V, E, t_pad, V_mask=V_mask)
            if t_pad.ndim == 0:
                t_pad = t_pad[None]
            alpha_t = self.alphas[t_pad][:, None, None]
            alpha_t_bar = self.alpha_bars[t_pad][:, None, None]
            # Partially denoising the image
            if v_condition_flag:
                # if t == 0:
                #     print(alpha_t_bar) # Not ideally 1.0, but 0.9990
                V = alpha_t_bar.sqrt() * V_known + (1 - alpha_t_bar).sqrt() * noise_V
            else:
                V = (1 / alpha_t.sqrt()) * (V - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_V)
            E = (1 / alpha_t.sqrt()) * (E - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_E)
            # print(abs(eta_V).max(), abs(eta_E).max())
            if t > 0:
                zV = torch.randn_like(V)
                z_E = torch.randn_like(E)
                # Option 1: sigma_t squared = beta_t
                beta_t = self.betas[t_pad]
                sigma_t = beta_t.sqrt()[:, None, None]
                if v_condition_flag:
                    V = V
                else:
                    V = V + sigma_t * zV
                E = E + sigma_t * z_E
            else:
                if v_condition_flag:
                    V = V_known
            if self.use_hard_v_mask:
                V[..., 0] = V_mask
        # scale back
        # V[:, :, 7:] = V[:, :, 7:] * shapecode_std[None, None, :]
        # ! now assume the R0 = I, append this back to the V
        V = torch.cat([V[..., :4], torch.zeros_like(V[..., 1:4]), V[..., 4:]], -1)
        V = V * V_scale[None, None, :]
        E = E * E_scale[None, None, :]
        confirmed_v_mask = self.make_sure_two_fg_nodes(V[..., 0])
        V[..., 0] = confirmed_v_mask
        # extract the graph
        E_type = E[..., :3]
        E_value = (
            E_type[..., 1:].max(-1).values - E_type[..., 0]
        )  # how 1,2 edge is larger than the prob to be the empty edge
        E_mtx = self.network_dict["denoiser"].scatter_trilist_to_mtx(E_value[..., None])
        E_mtx = E_mtx.squeeze(-1)
        E_mtx = self.extract_tree_from_mst_mtx(E_mtx, confirmed_v_mask)
        gather_ind = self.network_dict["denoiser"].tri_ind_to_full_ind.clone()
        E_mtx = E_mtx.reshape(B, -1)
        E_fg = torch.gather(E_mtx, 1, gather_ind[None, :].expand(B, -1).to(E_mtx.device))
        # E_fg.sum(-1)
        E_fg = E_fg > 0
        final_E_type = torch.zeros_like(E_type[..., 0]).long()
        if self.force_chirality:
            final_E_type[E_fg] = torch.ones_like(E_type[..., 1:].argmax(-1)[E_fg] + 1)
        else:
            final_E_type[E_fg] = E_type[..., 1:].argmax(-1)[E_fg] + 1
        final_E_type = F.one_hot(final_E_type, num_classes=3).float()

        # project the plucker to valid plucker
        E = self.project_to_plucker(E)
        # pack output
        ret_E = torch.cat([final_E_type, E[..., 3:]], -1)
        return V, ret_E

    @torch.no_grad()
    def generate_full_trace(self, noise_V, noise_E, V_scale, E_scale, V_mask=None, V_known=None):
        # v_mask is used as padded batch, the number of nodes is decided during sampling
        # noise_V: B,K,1+6+C_shapecode; noise_E: B, K(K-1)/2,13; v_mask: B,K,1
        # shapecode_std: C_shapecode

        # shape check
        B, K, _ = noise_V.shape
        assert K == self.max_K, "TODO: support larger K for inference"
        assert noise_E.shape[0] == B and noise_E.shape[1] == K * (K - 1) // 2

        v_condition_flag = False
        if V_known is not None:
            v_condition_flag = True
            logging.info("Experimental: conditional on V")
            assert (
                V_known.shape == noise_V.shape
            ), f"V_known shape {V_known.shape} != noise {noise_V.shape}"

        E, V = noise_E.clone(), noise_V.clone()
        if V_mask is not None:
            assert (noise_V[..., 0] == V_mask).all()
        logging.info(f"Diffusion Generation with {self.M} steps...")

        # V_trace = [V.cpu().clone()]
        # E_trace = [E.cpu().clone()]
        V_trace, E_trace = [], []

        for idx, t in tqdm(enumerate(list(range(self.M))[::-1])):
            t_pad = torch.ones(E.shape[0]).cuda().squeeze() * t
            t_pad = t_pad.long()
            # TODO: to be safe, here better to mask also mask
            eta_V, eta_E = self.network_dict["denoiser"](V, E, t_pad, V_mask=V_mask)
            if t_pad.ndim == 0:
                t_pad = t_pad[None]
            alpha_t = self.alphas[t_pad][:, None, None]
            alpha_t_bar = self.alpha_bars[t_pad][:, None, None]
            # Partially denoising the image
            if v_condition_flag:
                # if t == 0:
                #     print(alpha_t_bar) # Not ideally 1.0, but 0.9990
                V = alpha_t_bar.sqrt() * V_known + (1 - alpha_t_bar).sqrt() * noise_V
            else:
                V = (1 / alpha_t.sqrt()) * (V - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_V)
            E = (1 / alpha_t.sqrt()) * (E - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_E)
            # print(abs(eta_V).max(), abs(eta_E).max())
            if t > 0:
                zV = torch.randn_like(V)
                z_E = torch.randn_like(E)
                # Option 1: sigma_t squared = beta_t
                beta_t = self.betas[t_pad]
                sigma_t = beta_t.sqrt()[:, None, None]
                if v_condition_flag:
                    V = V
                else:
                    V = V + sigma_t * zV
                E = E + sigma_t * z_E
            else:
                if v_condition_flag:
                    V = V_known
            if self.use_hard_v_mask:
                V[..., 0] = V_mask

            V_trace.append(V.cpu().clone())
            E_trace.append(E.cpu().clone())
        # scale back
        # V[:, :, 7:] = V[:, :, 7:] * shapecode_std[None, None, :]
        # ! now assume the R0 = I, append this back to the V
        ret_V_trace, ret_E_trace = [], []
        for V, E in tqdm(zip(V_trace, E_trace)):
            V, E = V.cuda(), E.cuda()
            V = torch.cat([V[..., :4], torch.zeros_like(V[..., 1:4]), V[..., 4:]], -1)
            V = V * V_scale[None, None, :]
            E = E * E_scale[None, None, :]
            confirmed_v_mask = self.make_sure_two_fg_nodes(V[..., 0])
            V[..., 0] = confirmed_v_mask
            # extract the graph
            E_type = E[..., :3]
            E_value = (
                E_type[..., 1:].max(-1).values - E_type[..., 0]
            )  # how 1,2 edge is larger than the prob to be the empty edge
            E_mtx = self.network_dict["denoiser"].scatter_trilist_to_mtx(E_value[..., None])
            E_mtx = E_mtx.squeeze(-1)
            E_mtx = self.extract_tree_from_mst_mtx(E_mtx, confirmed_v_mask)
            gather_ind = self.network_dict["denoiser"].tri_ind_to_full_ind.clone()
            E_mtx = E_mtx.reshape(B, -1)
            E_fg = torch.gather(E_mtx, 1, gather_ind[None, :].expand(B, -1).to(E_mtx.device))
            # E_fg.sum(-1)
            E_fg = E_fg > 0
            final_E_type = torch.zeros_like(E_type[..., 0]).long()
            final_E_type[E_fg] = E_type[..., 1:].argmax(-1)[E_fg] + 1
            final_E_type = F.one_hot(final_E_type, num_classes=3).float()

            # project the plucker to valid plucker
            E = self.project_to_plucker(E)
            # pack output
            ret_E = torch.cat([final_E_type, E[..., 3:]], -1)
            ret_V_trace.append(V)
            ret_E_trace.append(ret_E)
        return ret_V_trace, ret_E_trace

    def convert_plucker_from_6d(self, E):
        raise NotImplementedError()
        return E

    def forward(self, input_pack, viz_flag):
        output = {}
        output["viz_flag"] = viz_flag
        phase, epoch = input_pack["phase"], input_pack["epoch"]

        V_gt, E_gt = input_pack["V"], input_pack["E"]
        # v: [mask_occ(1), bbox(3), r_gl(3), t_gl(3) | additional codes in the future]
        # e: [type(3), plucker(6), rlim(2), plim(2)]
        # ! remove the node rotation0 for now, assuem all I
        V_gt = torch.cat([V_gt[..., :4], V_gt[..., 7:]], -1)
        # Now v: [mask_occ(1), bbox(3), t_gl(3) | additional codes in the future]
        V_scale, E_scale = input_pack["V_scale"][0], input_pack["E_scale"][0]
        # C, loaded pre-computed shapecode already normalized
        B, K, _ = V_gt.shape

        # can supervise multiple t step for one object in the batch
        t = np.random.randint(0, self.M, (B * self.N_t_training))
        V0 = V_gt[:, None, ...].expand(-1, self.N_t_training, -1, -1)  # B,T,K,Fv
        E0 = E_gt[:, None, ...].expand(-1, self.N_t_training, -1, -1)  # B,T,K,Fe
        V0 = V0.reshape(B * self.N_t_training, K, -1)  # B*T,K,Fv
        E0 = E0.reshape(B * self.N_t_training, K * (K - 1) // 2, -1)  # B*T,|E|,Fe

        # Forward
        eta_V = torch.randn_like(V0)
        eta_E = torch.randn_like(E0)
        a_bar = self.alpha_bars[t]
        noisy_V = a_bar.sqrt()[:, None, None] * V0 + (1 - a_bar).sqrt()[:, None, None] * eta_V
        noisy_E = a_bar.sqrt()[:, None, None] * E0 + (1 - a_bar).sqrt()[:, None, None] * eta_E

        if self.use_hard_v_mask:
            V_mask_gt = V0[..., 0]
            # also always set the noisy mask to gt
            noisy_V[..., 0] = V_mask_gt
        else:
            V_mask_gt = None

        # backward
        eta_V_hat, eta_E_hat = self.network_dict["denoiser"](noisy_V, noisy_E, t, V_mask=V_mask_gt)
        if self.use_hard_v_mask:
            eta_V_hat[..., 0] = eta_V[..., 0]

        ########################################################################
        # output
        error_v, error_e = (eta_V - eta_V_hat) ** 2, (eta_E - eta_E_hat) ** 2

        if self.use_hard_v_mask:
            # prepare valid E mask
            E_mask_gt = self.network_dict["denoiser"].get_E_mask_gt(V_mask_gt)
            loss_v_i = (error_v * V_mask_gt[..., None]).sum(1) / V_mask_gt[..., None].sum(1)
            loss_e_i = (error_e * E_mask_gt[..., None]).sum(1) / E_mask_gt[..., None].sum(1)
        else:
            loss_v_i = error_v.mean(1)
            loss_e_i = error_e.mean(1)
        loss_v_occ_i = loss_v_i[:, 0]
        loss_v_bbox_i = loss_v_i[:, 1:4]
        loss_v_center_i = loss_v_i[:, 4:7]
        loss_v_shape_i = loss_v_i[:, 7:]
        if self.use_saperate_loss:
            loss_v_i = (
                loss_v_occ_i
                + loss_v_bbox_i.mean(-1)
                + loss_v_center_i.mean(-1)
                + loss_v_shape_i.mean(-1)
            ) / 4
        else:
            loss_v_i = loss_v_i.mean(-1)  # ! warning, here equally weight all channels
        loss_e_type_i = loss_e_i[:, :3]
        loss_e_plucker_i = loss_e_i[:, 3:9]
        loss_e_lim_i = loss_e_i[:, 9:]
        if self.use_saperate_loss:
            loss_e_i = loss_e_type_i.mean(-1) + loss_e_plucker_i.mean(-1) + loss_e_lim_i.mean(-1)
        else:
            loss_e_i = loss_e_i.mean(-1)  # ! warning, here equally weight all channels

        output["batch_loss"] = loss_v_i.mean() + loss_e_i.mean()
        output["loss_v"] = loss_v_i.mean().detach()
        output["loss_e"] = loss_e_i.mean().detach()
        output["loss_v_i"] = loss_v_i.detach()
        output["loss_e_i"] = loss_e_i.detach()
        output["V_scale"] = V_scale.detach()
        output["E_scale"] = E_scale.detach()

        output["loss_v_occ"] = loss_v_occ_i.mean().detach()
        output["loss_v_bbox"] = loss_v_bbox_i.mean().detach()
        output["loss_v_center"] = loss_v_center_i.mean().detach()
        output["loss_v_shape"] = loss_v_shape_i.mean().detach()
        output["loss_e_type"] = loss_e_type_i.mean().detach()
        output["loss_e_plucker"] = loss_e_plucker_i.mean().detach()
        output["loss_e_lim"] = loss_e_lim_i.mean().detach()

        if viz_flag:
            output["V_gt"] = input_pack["V"].clone() * V_scale[None, None, :]
            output["E_gt"] = input_pack["E"].clone() * E_scale[None, None, :]
            # v: [mask_occ(1), bbox(3), r_gl(3), t_gl(3) | additional codes in the future]

        return output

    def decode_sdf(self, query, z_none, c, return_sdf=False):
        if self.N_pe > 0:
            # Do pe
            B, N, _ = query.shape
            w = self.freq.to(query.device)
            pe = w[None, None, None, ...] * query[..., None]
            pe = pe.reshape(B, N, -1)
            query = torch.cat([query, torch.cos(pe), torch.sin(pe)], -1)
        sdf = self.network_dict["sdf_decoder"](query, None, c)
        if return_sdf:
            return sdf
        else:
            return dist.Bernoulli(logits=self.sdf2occ_factor * sdf)


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)
