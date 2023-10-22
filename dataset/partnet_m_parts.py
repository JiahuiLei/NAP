# Load partnet motion part

from random import random
from torch.utils.data import Dataset
import logging
import json
import os
import os.path as osp
import numpy as np
from os.path import join
import torch
from core.models.utils.misc import cfg_with_default
from tqdm import tqdm
import json


class Dataset(Dataset):
    def __init__(self, cfg, mode) -> None:
        super().__init__()

        d_cfg = cfg["dataset"]
        self.mode = mode.lower()
        self.dataset_proportion = d_cfg["dataset_proportion"][cfg["modes"].index(self.mode)]
        self.data_root = join(cfg["root"], d_cfg["data_root"])

        with open(d_cfg["split_path"], "r") as f:
            self.meshid = json.load(f)[self.mode]

        # check the validity of the npz files
        self.valid_mesh_id = []
        for mesh_id in self.meshid:
            if osp.isdir(osp.join(self.data_root, mesh_id)):
                self.valid_mesh_id.append(mesh_id)
            else:
                logging.warning(f"Skip missing {mesh_id}")

        proportion_step = int(1.0 / self.dataset_proportion)
        self.valid_mesh_id = self.valid_mesh_id[::proportion_step]
        logging.info(
            f"Dataset {mode} with {self.dataset_proportion * 100}% data, dataset len is {len(self)}"
        )

        self.chunk_cfg = d_cfg["chunk"]
        self.n_uniform = d_cfg["n_uniform"]
        self.n_nearsurface = d_cfg["n_nearsurface"]
        # pcl is for viz purpose
        self.n_pcl = cfg_with_default(d_cfg, "n_pcl", 1024)

        return

    def __len__(self):
        return len(self.valid_mesh_id)

    def __getitem__(self, index):
        ret = {"index": index}
        mesh_id = self.valid_mesh_id[index]
        mesh_dir = osp.join(self.data_root, self.valid_mesh_id[index])
        uni = self.load_from_slice(
            mesh_dir,
            self.chunk_cfg["uniform"],
            self.n_uniform,
            file_prefix="uni",
            train=self.mode == "train",
        )["points"]
        ns = self.load_from_slice(
            mesh_dir,
            self.chunk_cfg["near"],
            self.n_nearsurface,
            file_prefix="near",
            train=self.mode == "train",
        )["points"]
        uni = self.sample(uni, self.n_uniform, train=self.mode == "train")
        ns = self.sample(ns, self.n_nearsurface, train=self.mode == "train")
        ret["points.uni"] = uni[:, :3]
        ret["points.uni.value"] = uni[:, 3]
        ret["points.ns"] = ns[:, :3]
        ret["points.ns.value"] = ns[:, 3]
        # also load pcl for viz
        pcl = self.load_from_slice(
            mesh_dir,
            self.chunk_cfg["pcl"],
            self.n_pcl,
            file_prefix="pointcloud",
            train=self.mode == "train",
        )["points"]
        pcl = self.sample(pcl, self.n_pcl, train=self.mode == "train")
        ret["pointcloud"] = pcl

        meta_info = {"meshid": mesh_id}
        viz_id = f"{mesh_id}-{index}"
        meta_info["viz_id"] = viz_id
        meta_info["mode"] = self.mode
        return ret, meta_info

    def sample(self, pcl, N, train=True):
        if train:
            choice = np.random.choice(len(pcl), N, replace=True)
            return pcl[choice]
        else:
            return pcl[:N]

    def load_from_slice(self, dir, num_slice, target_n, file_prefix, train=True):
        # compute how many slice
        n_load_slice = int(np.ceil(target_n / self.chunk_cfg["size"]))
        if train:
            choice = np.random.choice(num_slice, n_load_slice, replace=True)
        else:
            choice = [i for i in range(n_load_slice)]
        data = {}
        for cid in choice:
            _d = np.load(osp.join(dir, f"{file_prefix}_{cid}.npz"), allow_pickle=True)
            for f in _d.files:
                if f not in data.keys():
                    data[f] = [_d[f]]
                else:
                    data[f].append(_d[f])
        for k in data.keys():
            if data[k][0].ndim == 0:
                data[k] = data[k][0]  # for repeated scale
            elif isinstance(data[k][0], np.ndarray):
                data[k] = np.concatenate(data[k], 0)
            else:
                pass
        return data
