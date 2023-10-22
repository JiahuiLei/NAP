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
from object_utils.arti_graph_utils_v3 import (
    compact_pack,
    autoregressive_to_complete,
    complete_to_autoregressive,
)
from copy import deepcopy


class Dataset(Dataset):
    def __init__(self, cfg, mode) -> None:
        super().__init__()

        d_cfg = cfg["dataset"]
        self.mode = mode.lower()
        self.dataset_proportion = d_cfg["dataset_proportion"][cfg["modes"].index(self.mode)]
        self.data_root = join(cfg["root"], d_cfg["data_root"])

        self.auto_regressive_flag = cfg_with_default(d_cfg, ["auto_regressive_flag"], False)

        self.pad_K = d_cfg["max_K"]
        self.valid_obj_ind = self.load_split(
            d_cfg["split_path"], phase=self.mode, cates=d_cfg["cates"]
        )

        # config the scale: if scale all, scale all V and E, otherwise scale only shapecode
        self.scale_all = cfg_with_default(d_cfg, ["scale_all"], False)
        # if use std, then use std, then use the maximum
        self.scale_mode = cfg_with_default(d_cfg, ["scale_mode"], "std")

        # prepare the code book
        mesh_index_fn = d_cfg["embedding_index_file"]
        with open(mesh_index_fn, "r") as f:
            embedding_index = json.load(f)
        self.embedding_index = (
            embedding_index["train"] + embedding_index["val"] + embedding_index["test"]
        )
        self.training_embedding_index = embedding_index["train"]
        N_training_embeddings = len(embedding_index["train"])
        self.embedding_precompute_path = cfg_with_default(
            d_cfg, ["embedding_precompute_path"], None
        )
        self.load_code_flag = self.embedding_precompute_path is not None
        if self.load_code_flag:
            data = np.load(self.embedding_precompute_path, allow_pickle=True)
            self.embedding = data["embedding"]
            # ! all the embedding codebook is saved in order [train, val, test] !!
            self.training_embedding = self.embedding[:N_training_embeddings]
            if self.scale_mode == "std":
                self.embedding_scale = data["std"] + 1e-8
            elif self.scale_mode == "max":
                self.embedding_scale = abs(self.embedding).max(axis=0) + 1e-8
            else:
                raise NotImplementedError()
            self.embedding_valid = data["valid_mask"]
            logging.info(
                f"Load embedding from {self.embedding_precompute_path}, totally valid precompute {self.embedding_valid.sum()} embeddings"
            )

        # prepare balance sampling
        self.balance_flag = cfg_with_default(d_cfg, ["balance_sampling"], False)
        self._cache_data()
        logging.info(
            f"Dataset {mode} with {self.dataset_proportion * 100}% data, dataset len is {len(self)}"
        )
        self.permute_nodes = (
            cfg_with_default(d_cfg, ["permute_nodes"], False) and self.mode == "train"
        )  # only training permute nodes
        return
    
    def get_phase_embedding(self, phase):
        return

    def _cache_data(self):
        self.data_list, self.data_partnet_index_list = [], []
        self.data_nv_list = []
        _fn_list = os.listdir(self.data_root)
        _fn_list.sort()
        proportion_step = int(1.0 / self.dataset_proportion)
        _fn_list = _fn_list[::proportion_step]
        self.balance_cnt_dict = {k: [] for k in range(2, self.pad_K + 1)}
        # stat
        all_V, all_E = [], []
        for fn in tqdm(_fn_list):
            partnet_index = fn.split(".")[0]
            if not partnet_index in self.valid_obj_ind:
                continue
            data = np.load(osp.join(self.data_root, fn), allow_pickle=True)
            E, V = data["E"].tolist(), data["V"].tolist()
            nv = len(V)
            assert nv >= 2
            if len(V) > self.pad_K:
                logging.debug(
                    f"Skip object {fn} becuase it has {len(V)} nodes more than {self.pad_K}"
                )
                continue
            self.data_nv_list.append(nv)
            self.balance_cnt_dict[nv].append(partnet_index)
            self.data_partnet_index_list.append(partnet_index)
            self.data_list.append((E, V))
            _V, _E, _ = compact_pack(V, E, permute=False, K=self.pad_K)
            if self.auto_regressive_flag:
                _V, _E, _ = complete_to_autoregressive(_V, _E, False)
            all_V.append(_V), all_E.append(_E)
        self.balance_cnt = {k: len(self.balance_cnt_dict[k]) for k in range(2, self.pad_K + 1)}
        # for cnt in self.balance_cnt.values():
        #     assert cnt > 0, NotImplementedError("TODO: support zero cnt")
        logging.info(f"Balance cnt {self.balance_cnt}")

        self.cls_weight = [0.0, 0.0] + [
            1.0 / self.balance_cnt[k] if self.balance_cnt[k] > 0 else 0.0
            for k in range(2, self.pad_K + 1)
        ]
        self.cls_weight_list = [self.cls_weight[nv] for nv in self.data_nv_list]

        # Also stat the dataset
        if self.auto_regressive_flag:
            v_occ_cnt, e_type_cnt = 1, 2  # autoregressive only have 2 head
        else:
            v_occ_cnt, e_type_cnt = 1, 3
        all_V = np.concatenate(all_V, axis=0)
        all_E = np.concatenate(all_E, axis=0)
        if self.scale_mode == "std":
            self.V_scale = all_V.std(axis=0)[v_occ_cnt:] + 1e-8
            self.E_scale = all_E.std(axis=0)[e_type_cnt:] + 1e-8
        elif self.scale_mode == "max":
            self.V_scale = abs(all_V).max(axis=0)[v_occ_cnt:] + 1e-8
            self.E_scale = abs(all_E).max(axis=0)[e_type_cnt:] + 1e-8
        else:
            raise NotImplementedError()
        self.V_scale = np.concatenate([np.ones(v_occ_cnt), self.V_scale], axis=0)
        self.E_scale = np.concatenate([np.ones(e_type_cnt), self.E_scale], axis=0)
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
        E_list, V_list = self.data_list[index]
        nv = len(V_list)
        V, E, v_map = compact_pack(V_list, E_list, permute=self.permute_nodes, K=self.pad_K)
        V = V * V[:, 0:1]

        # convert to autoregressive
        if self.auto_regressive_flag:
            V, E, v_map2 = complete_to_autoregressive(V, E, self.permute_nodes)
            _v_map = [v_map[i] for i in v_map2]
            v_map = _v_map

        # scale
        if self.scale_all:
            V = V / self.V_scale
            E = E / self.E_scale
            V_scale = self.V_scale
            E_scale = self.E_scale
        else:
            V_scale = np.ones_like(self.V_scale)
            E_scale = np.ones_like(self.E_scale)

        ret = {"V": V, "E": E}
        # Complete
        # v: [mask_occ(1), bbox(3), r_gl(3), t_gl(3) | additional codes in the future]
        # e: [type(3), plucker(6), rlim(2), plim(2)]
        mesh_name_list = []
        if self.load_code_flag:
            code_list = []
            for _i, occ in zip(v_map, V[:, 0]):
                if occ < 0.5:
                    code_list.append(np.zeros_like(self.embedding[0]))
                else:
                    key = f"{self.data_partnet_index_list[index]}_{_i}"
                    mesh_name_list.append(key)
                    code_id = self.embedding_index.index(key)
                    code_list.append(self.embedding[code_id])
            code_list = np.stack(code_list, 0)
            std = self.embedding_scale.copy()
            # ! warning, the code is already divided
            code_list = code_list / std[None, ...]
            precompute_code = torch.from_numpy(code_list).float()
            ret["V"] = torch.cat([ret["V"], precompute_code], dim=-1)
            V_scale = np.concatenate([V_scale, std], axis=0)

        ret["V_scale"] = torch.from_numpy(V_scale).float()
        ret["E_scale"] = torch.from_numpy(E_scale).float()

        if self.auto_regressive_flag:
            # also randomly sample the seen mask and termination sign
            seen_n = np.random.randint(0, nv + 1)
            termination_sign = seen_n == nv
            ret["seen_n"] = seen_n
            seen_mask = np.zeros(len(V))
            seen_mask[:seen_n] = 1.0  # first compact
            ret["seen_mask"] = seen_mask
            ret["stop"] = termination_sign
        
        # append dataset index for auto-decoding
        ret["dataset_index"] = index

        meta_info = {
            "partnet-m-id": self.data_partnet_index_list[index],
            "joint_mapping": ["revolute", "prismatic"],
            "auto_regressive_flag": self.auto_regressive_flag,
            "mesh_name_list": mesh_name_list + [""] * (self.pad_K - len(mesh_name_list)),
        }
        viz_id = f"partnet-{self.data_partnet_index_list[index] }-{index}"
        meta_info["viz_id"] = viz_id
        meta_info["mode"] = self.mode
        return ret, meta_info
