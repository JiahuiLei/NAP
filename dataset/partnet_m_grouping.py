# Load processed PartNet-Mobility graph
# v5: from v4 use new full random permute, not first 1 v_mask

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
from object_utils.arti_graph_utils_v3 import compact_pack, map_upper_triangle_to_list
from copy import deepcopy


class Dataset(Dataset):
    def __init__(self, cfg, mode) -> None:
        super().__init__()

        d_cfg = cfg["dataset"]
        self.mode = mode.lower()
        self.dataset_proportion = d_cfg["dataset_proportion"][cfg["modes"].index(self.mode)]
        self.data_root = join(cfg["root"], d_cfg["data_root"])

        self.pad_nv = d_cfg["max_K"]
        self.pad_np = d_cfg["max_P"]

        self.n_pcl = d_cfg["n_pcl"]

        self.valid_obj_ind = self.load_split(
            d_cfg["split_path"], phase=self.mode, cates=d_cfg["cates"]
        )

        self.balance_flag = cfg_with_default(d_cfg, ["balance_flag"], False)

        ########################################################################

        self.data_list, self.data_partnet_index_list = [], []
        _fn_list = os.listdir(self.data_root)
        _fn_list.sort()
        proportion_step = int(1.0 / self.dataset_proportion)
        _fn_list = _fn_list[::proportion_step]

        self.num_leaves_list, self.num_ins_list = [], []
        self.n_leaf_cnt = {k: 0 for k in range(0, self.pad_np + 1)}
        self.n_ins_cnt = {k: 0 for k in range(0, self.pad_nv + 1)}
        for fn in tqdm(_fn_list):
            partnet_index = fn.split(".")[0]
            if not partnet_index in self.valid_obj_ind:
                continue
            data = np.load(osp.join(self.data_root, fn), allow_pickle=True)
            pcl = data["pcl"]
            ins_id = data["ins_id"]
            center, bbox = data["center"], data["bbox"]
            nv = len(np.unique(ins_id))
            if nv < 2:
                logging.warning(f"Warning, bad datapoint with only {nv} nodes, skip")
                continue
            if nv > self.pad_nv:
                logging.debug(f"Skip object {fn} becuase it has {nv} nodes more than {self.pad_nv}")
                continue
            if len(pcl) > self.pad_np:
                logging.debug(
                    f"Skip object {fn} becuase it has {len(pcl)} parts more than {self.pad_np}"
                )
                continue
            if -1 in ins_id:
                logging.debug(f"Skip object {fn} becuase it has -1 in ins_id")
                continue
            self.data_partnet_index_list.append(partnet_index)
            self.data_list.append((pcl, ins_id, center, bbox))
            self.n_leaf_cnt[len(pcl)] += 1
            self.num_leaves_list.append(len(pcl))
            self.n_ins_cnt[nv] += 1
            self.num_ins_list.append(nv)
        # prepare for balance sampling
        # # based on leaf num
        # self.cls_weight = [
        #     1.0 / self.n_leaf_cnt[k] if self.n_leaf_cnt[k] > 0 else 0.0
        #     for k in range(0, self.pad_np + 1)
        # ]
        # self.cls_weight_list = [self.cls_weight[np] for np in self.num_leaves_list]
        
        # based on nv
        self.cls_weight = [
            1.0 / self.n_ins_cnt[k] if self.n_ins_cnt[k] > 0 else 0.0
            for k in range(0, self.pad_nv + 1)
        ]
        self.cls_weight_list = [self.cls_weight[nv] for nv in self.num_ins_list]

        ########################################################################

        logging.info(
            f"Dataset {mode} with {self.dataset_proportion * 100}% data, dataset len is {len(self)}"
        )
        return

    def get_sampler(self):
        from torch.utils.data import WeightedRandomSampler

        if not self.balance_flag or self.mode != "train":
            return None
        weighted_sampler = WeightedRandomSampler(
            weights=self.cls_weight_list, num_samples=len(self.cls_weight_list), replacement=True
        )
        logging.info("Warning, use weighted sampler, with nv weight: {}".format(self.cls_weight))
        return weighted_sampler

    def load_split(self, json_fn, phase, cates):
        with open(json_fn, "r") as f:
            data = json.load(f)
        logging.info(f"valid cates in split: {[k for k in data.keys()]}")
        if cates == ["all"]:
            cates = [k for k in data.keys()]
        id_list = []
        for k in cates:
            id_list += data[k][phase]
        return id_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        pcl, ins_id, center, bbox = self.data_list[index]
        num_parts = len(pcl)
        if self.mode == "train":
            sampled_pcl = []
            for i in range(len(pcl)):
                choice = np.random.choice(len(pcl[i]), self.n_pcl, replace=True)
                sampled_pcl.append(pcl[i][choice])
            sampled_pcl = np.stack(sampled_pcl, axis=0)
        else:
            sampled_pcl = pcl[:, : self.n_pcl, :]

        mask = np.zeros((self.pad_np), dtype=np.float32)
        mask[:num_parts] = 1.0
        ret_pcl = np.zeros((self.pad_np, self.n_pcl, 3), dtype=np.float32)
        ret_pcl[:num_parts, :, :] = sampled_pcl

        ret_ins_id = -np.ones((self.pad_np), dtype=np.int64)
        ret_ins_id[:num_parts] = ins_id

        ret_box, ret_center = np.zeros((self.pad_np, 3), dtype=np.float32), np.zeros(
            (self.pad_np, 3), dtype=np.float32
        )
        ret_box[:num_parts, :] = bbox
        ret_center[:num_parts, :] = center
        
        num_ins = len(np.unique(ins_id))
        ret = {
            "mask": mask,
            "pcl": ret_pcl,
            "ins_id": ret_ins_id,
            "box": ret_box,
            "center": ret_center,
            "ins_cnt": num_ins,
        }

        meta_info = {
            "partnet-m-id": self.data_partnet_index_list[index],
        }
        viz_id = f"partnet-{self.data_partnet_index_list[index]}-{index}-{self.mode}"
        meta_info["viz_id"] = viz_id
        meta_info["mode"] = self.mode
        return ret, meta_info
