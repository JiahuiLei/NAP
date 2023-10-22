# V5: from v2, remove the v_mask, everyone is valid!
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from object_utils.arti_graph_utils_v3 import get_G_from_VE, map_upper_triangle_to_list
from object_utils.arti_viz_utils import *


class NaiveMLP(torch.nn.Module):
    def __init__(self, z_dim, out_dim, hidden_dims, use_bn=False) -> None:
        super().__init__()
        self.use_bn = use_bn
        c_wide_out_cnt = z_dim
        self.layers = nn.ModuleList()
        c_in = z_dim
        for i, c_out in enumerate(hidden_dims):
            if self.use_bn:
                normalization = BN_lastdim(c_out)
            else:
                normalization = nn.Identity()
            self.layers.append(
                nn.Sequential(
                    nn.Linear(c_in, c_out),
                    normalization,
                    nn.LeakyReLU(),
                )
            )
            c_wide_out_cnt += c_out
            c_in = c_out
        self.out_fc0 = nn.Linear(c_wide_out_cnt, out_dim)
        return

    def forward(self, x):
        f_list = [x]
        for l in self.layers:
            x = l(x)
            f_list.append(x)
        f = torch.cat(f_list, -1)
        y = self.out_fc0(f)
        # y1 = self.out_fc1(f)
        return y


def tri_ind_to_full_ind(K):
    tri_ind, full_ind = [], []
    src_dst_ind = []
    for i in range(K):
        for j in range(K):
            if i < j:
                src_dst_ind.append([i, j])
                tri_ind.append(map_upper_triangle_to_list(i, j, K))
                full_ind.append(i * K + j)
    assert tri_ind == [i for i in range(len(tri_ind))]
    src_dst_ind = np.array(src_dst_ind).T.astype(np.long)
    return torch.Tensor(full_ind).long(), torch.from_numpy(src_dst_ind)


class BN_lastdim(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, x):
        x.shape
        y = self.bn(x.transpose(1, -1)).transpose(1, -1)
        return y


class NaiveDenoiser(torch.nn.Module):
    def __init__(
        self,
        K,  #
        M,  # time steps
        t_pe_dim=200,
        v_enc_dim=256,
        e_enc_dim=128,
        V_dims: list = [1, 3, 3, 128],
        E_dims: list = [3, 6, 4],
        h_dim=512,
        n_layers=6,
        use_bn=True,
    ) -> None:
        super().__init__()

        self.K, self.M = K, M
        self.E_num = K * (K - 1) // 2

        self.V_dims = V_dims
        self.E_dims = E_dims
        self.t_pe_dim = t_pe_dim

        self.v_enc_dim = v_enc_dim
        self.e_enc_dim = e_enc_dim

        self.register_buffer("t_pe", self.sinusoidal_embedding(M, self.t_pe_dim))

        # input layers
        self.v_in_layers = nn.ModuleList([nn.Linear(v_in, v_enc_dim) for v_in in self.V_dims])
        self.fc_v_in = nn.Sequential(
            nn.Linear(v_enc_dim * len(self.V_dims), v_enc_dim), nn.LeakyReLU()
        )

        self.e_in_layers = nn.ModuleList([nn.Linear(e_in, e_enc_dim) for e_in in self.E_dims])
        self.fc_e_in = nn.Sequential(
            nn.Linear(e_enc_dim * len(self.E_dims), self.e_enc_dim), nn.LeakyReLU()
        )

        self.fc_v_h = nn.Sequential(nn.Linear(self.K * v_enc_dim, h_dim), nn.LeakyReLU())
        self.fc_e_h = nn.Sequential(nn.Linear(self.E_num * e_enc_dim, h_dim), nn.LeakyReLU())
        self.fc_in = nn.Sequential(nn.Linear(h_dim * 2 + t_pe_dim, h_dim), nn.LeakyReLU())
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(h_dim + t_pe_dim, h_dim),
                    nn.BatchNorm1d(h_dim) if use_bn else nn.Identity(),
                    nn.LeakyReLU(),
                    nn.Linear(h_dim, h_dim),
                    nn.BatchNorm1d(h_dim) if use_bn else nn.Identity(),
                    nn.LeakyReLU(),
                )
            )
        self.wide_fc_v = nn.Linear(h_dim * n_layers, self.K * v_enc_dim)
        self.wide_fc_e = nn.Linear(h_dim * n_layers, self.E_num * e_enc_dim)

        self.v_output_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(v_enc_dim, v_enc_dim),
                    nn.LeakyReLU(),
                    nn.Linear(v_enc_dim, v_enc_dim),
                    nn.LeakyReLU(),
                    nn.Linear(v_enc_dim, v_out),
                )
                for v_out in self.V_dims
            ]
        )
        self.e_output_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(e_enc_dim, e_enc_dim),
                    nn.LeakyReLU(),
                    nn.Linear(e_enc_dim, e_enc_dim),
                    nn.LeakyReLU(),
                    nn.Linear(e_enc_dim, e_out),
                )
                for e_out in self.E_dims
            ]
        )

        self.tri_ind_to_full_ind, self.src_dst_ind = tri_ind_to_full_ind(self.K)
        return

    @staticmethod
    def sinusoidal_embedding(n, d):
        # Returns the standard positional embedding
        embedding = torch.zeros(n, d)
        wk = torch.tensor([1 / 10_000 ** (2 * j / d) for j in range(d)])
        wk = wk.reshape((1, d))
        t = torch.arange(n).reshape((n, 1))
        embedding[:, ::2] = torch.sin(t * wk[:, ::2])
        embedding[:, 1::2] = torch.cos(t * wk[:, ::2])
        return embedding

    def scatter_trilist_to_mtx(self, buffer):
        # buffer: B,|E|,F
        B, E, F = buffer.shape
        assert E == self.K * (self.K - 1) // 2, "E should be K(K-1)/2"
        ind = self.tri_ind_to_full_ind.to(buffer.device)[None, :, None]
        ind = ind.expand(B, -1, F)
        ret = torch.zeros(B, self.K * self.K, F, device=buffer.device)
        ret = torch.scatter(ret, 1, ind, buffer)
        ret = ret.reshape(B, self.K, self.K, F)
        return ret  # zero padding

    def forward(self, V_noisy, E_noisy, t, V_mask=None):
        # first get symmetry nodes
        B, K, _ = V_noisy.shape

        emb_t = self.t_pe[t]  # B,PE
        # prepare init feature
        cur = 0
        v_f = []
        v_cur_occ = V_noisy[..., :1]
        for i, v_in_c in enumerate(self.V_dims):
            v_input = V_noisy[..., cur : cur + v_in_c]
            v_f.append(self.v_in_layers[i](v_input))
            cur += v_in_c
        v_f = self.fc_v_in(torch.cat(v_f, -1))
        v_f = v_f * v_cur_occ

        cur = 0
        e_f = []
        for i, e_in_c in enumerate(self.E_dims):
            e_input = E_noisy[..., cur : cur + e_in_c]
            e_f.append(self.e_in_layers[i](e_input))
            cur += e_in_c
        e_f = self.fc_e_in(torch.cat(e_f, -1))

        v_f = self.fc_v_h(v_f.reshape(B, -1))
        e_f = self.fc_e_h(e_f.reshape(B, -1))
        f = self.fc_in(torch.cat([v_f, e_f, emb_t], -1))
        f_list = []
        for l in self.layers:
            f = l(torch.cat([f, emb_t], -1))
            f_list.append(f)
        f = torch.cat(f_list, -1)
        v_f = self.wide_fc_v(f).reshape(B, K, self.v_enc_dim)
        e_f = self.wide_fc_e(f).reshape(B, self.E_num, self.e_enc_dim)

        v_out = torch.cat([l(v_f) for l in self.v_output_layers], -1)
        e_out = torch.cat([l(e_f) for l in self.e_output_layers], -1)

        return v_out, e_out
