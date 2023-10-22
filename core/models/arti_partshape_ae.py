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
from core.models.utils.probpose_utils import *
from matplotlib import cm
from tqdm import tqdm
from core.lib.implicit_func.onet_decoder import DecoderCBatchNorm, Decoder, DecoderSIREN
from core.lib.point_encoder.pointnet import ResnetPointnet
from torch import distributions as dist
import networkx as nx
import matplotlib.pyplot as plt
import imageio
from core.models.utils.oflow_eval.evaluator import MeshEvaluator
from core.models.utils.oflow_common import eval_iou
from .utils.occnet_utils import get_generator as get_mc_extractor

LARGE = 1e6


class Model(ModelBase):
    def __init__(self, cfg):
        network = PartAE(cfg)
        super().__init__(cfg, network)

        self.num_z = cfg_with_default(cfg, ["generation", "num_z"], 1)

        self.output_specs = {
            "metric": ["batch_loss", "loss_sdf", "loss_reg"] + ["loss_sdf_near", "loss_sdf_far"],
            "image": [],
            "mesh": ["mesh", "gt_pcl"],
            "hist": [],
            "xls": [],
        }
        self.viz_one = cfg["logging"]["viz_one_per_batch"]
        self.iou_threshold = cfg["evaluation"]["iou_threshold"]
        self.viz_dpi = cfg_with_default(cfg, ["logging", "viz_dpi"], 200)
        self.mesh_extractor = get_mc_extractor(cfg)
        return

    def generate_mesh(self, embedding):
        net = self.network.module if self.__dataparallel_flag__ else self.network
        mesh = self.mesh_extractor.generate_from_latent(c=embedding, F=net.decode)
        if mesh.vertices.shape[0] == 0:
            mesh = trimesh.primitives.Box(extents=(1.0, 1.0, 1.0))
            logging.warning("Mesh extraction fail, replace by a place holder")
        return mesh

    def _postprocess_after_optim(self, batch):
        if batch["viz_flag"]:
            n_batch = batch["z"].shape[0]
            with torch.no_grad():
                batch["mesh"], batch["gt_pcl"] = [], []
                for bid in range(n_batch):
                    mesh = self.generate_mesh(embedding=batch["z"][bid : bid + 1])
                    batch["mesh"].append(mesh)
                    batch["gt_pcl"].append(batch["model_input"]["pointcloud"][bid])
                    if self.viz_one:
                        break
            batch["gt_pcl"] = torch.stack(batch["gt_pcl"], 0)
        return batch


class PartAE(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = copy.deepcopy(cfg)

        self.z_dim = cfg["model"]["decoder"]["c_dim"]
        self.decoder_type = cfg_with_default(cfg, ["model", "decoder_type"], "decoder")
        decoder_class = {
            "decoder": Decoder,
            "cbatchnorm": DecoderCBatchNorm,
            "decoder_siren": DecoderSIREN,
        }[self.decoder_type]
        self.network_dict = torch.nn.ModuleDict(
            {
                "encoder": ResnetPointnet(**cfg["model"]["encoder"]),
                "decoder": decoder_class(**cfg["model"]["decoder"]),
            }
        )

        # loss weight
        m_cfg = cfg["model"]
        self.sdf2occ_factor = cfg_with_default(m_cfg, ["sdf2occ_factor"], -1.0)
        count_param(self.network_dict)

        self.w_sdf = cfg_with_default(cfg, ["training", "w_sdf"], 1.0)
        self.w_reg = cfg_with_default(cfg, ["training", "w_reg"], 0.001)

        self.near_th = cfg_with_default(cfg, ["training", "near_th"], -1)
        if self.near_th > 0:
            self.w_near = cfg_with_default(cfg, ["training", "w_near"], 1.0)
            self.w_far = cfg_with_default(cfg, ["training", "w_far"], 1.0)

        self.N_pe = cfg_with_default(m_cfg, ["N_pe"], 0)
        if self.N_pe > 0:
            self.freq = 2 ** torch.Tensor([i for i in range(self.N_pe)])

        return

    def decode(self, query, z_none, c, return_sdf=False):
        if self.N_pe > 0:
            # Do pe
            B, N, _ = query.shape
            w = self.freq.to(query.device)
            pe = w[None, None, None, ...] * query[..., None]
            pe = pe.reshape(B, N, -1)
            query = torch.cat([query, torch.cos(pe), torch.sin(pe)], -1)
        sdf = self.network_dict["decoder"](query, None, c)
        if return_sdf:
            return sdf
        else:
            return dist.Bernoulli(logits=self.sdf2occ_factor * sdf)
        
    def encode(self, pcl):
        return self.network_dict["encoder"](pcl)
    def forward(self, input_pack, viz_flag):
        output = {}
        output["viz_flag"] = viz_flag
        phase, epoch = input_pack["phase"], input_pack["epoch"]

        query = torch.cat([input_pack["points.uni"], input_pack["points.ns"]], 1)
        sdf_gt = torch.cat([input_pack["points.uni.value"], input_pack["points.ns.value"]], 1)

        pcl = input_pack["pointcloud"]  # B,N,3

        # z = self.network_dict["encoder"](pcl)
        z = self.encode(pcl)

        # Decode
        sdf_hat = self.decode(query, None, z, return_sdf=True)
        loss_sdf = abs(sdf_hat - sdf_gt)
        if self.near_th > 0:
            near_mask = sdf_gt < self.near_th
            near_loss = loss_sdf[near_mask].sum() / (near_mask.sum() + 1e-7)
            far_loss = loss_sdf[~near_mask].sum() / ((~near_mask).sum() + 1e-7)
            loss_sdf = self.w_near * near_loss + self.w_far * far_loss
            output["loss_sdf_near"] = near_loss.detach()
            output["loss_sdf_far"] = far_loss.detach()
        else:
            loss_sdf = loss_sdf.mean()
        loss_reg = (z**2).mean()
        output["batch_loss"] = self.w_sdf * loss_sdf + self.w_reg * loss_reg
        output["loss_sdf"] = loss_sdf.detach()
        output["loss_reg"] = loss_reg.detach()
        output["z"] = z.detach()
        return output
