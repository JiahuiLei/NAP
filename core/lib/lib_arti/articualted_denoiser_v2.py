# V2: upgrade for edge
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from object_utils.arti_graph_utils_v2 import get_G_from_VE, map_upper_triangle_to_list
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


class GraphDenoiseConvV2(torch.nn.Module):
    def __init__(
        self,
        K,  #
        M,  # time steps
        V_dim: int = 6 + 128,
        E_sym_dim: int = 10,
        E_dir_dim: int = 3,
        # Backbone
        node_conv_dims: list = [512, 512, 512, 512, 512, 512],
        p_pe_dim: int = 200,
        t_pe_dim: int = 100,
        # HEADS
        node_pred_head_dims: list = [512, 256, 256],
        edge_sym_head_dims: list = [512, 256, 128],
        edge_dir_head_dims: list = [512, 256, 128],
        use_bn=False,
        # others
        use_wide=False,
        use_global_pooling=False,
        edge_guide_factor: int = 10.0,
    ) -> None:
        super().__init__()
        self.K, self.M = K, M
        self.v_dim, self.e_s_dim, self.e_d_dim = V_dim, E_sym_dim, E_dir_dim
        self.p_pe_dim, self.t_pe_dim = p_pe_dim, t_pe_dim
        self.register_buffer("t_pe", self.sinusoidal_embedding(M, self.t_pe_dim))
        if self.p_pe_dim > 0:
            self.register_buffer("p_pe", self.sinusoidal_embedding(K, self.p_pe_dim))

        self.te_layers, self.pe_layers = nn.ModuleList(), nn.ModuleList()
        self.v_self_layers0, self.v_self_layers1 = nn.ModuleList(), nn.ModuleList()
        self.v_self_layers2 = nn.ModuleList()
        self.e_dir_conv_layers, self.e_sym_conv_layers = nn.ModuleList(), nn.ModuleList()
        self.e_conv_layers = nn.ModuleList()

        self.use_wide = use_wide
        self.use_global_pooling = use_global_pooling

        c_in = self.v_dim
        out_cnt = c_in
        edge_wide_cnt = 0
        for c_out in node_conv_dims:
            self.te_layers.append(
                nn.Sequential(
                    nn.Linear(self.t_pe_dim, self.t_pe_dim),
                    nn.LeakyReLU(),
                    nn.Linear(self.t_pe_dim, self.t_pe_dim),
                )
            )
            self.pe_layers.append(
                nn.Sequential(
                    nn.Linear(self.p_pe_dim, self.p_pe_dim),
                    nn.LeakyReLU(),
                    nn.Linear(self.p_pe_dim, self.p_pe_dim),
                )
            )
            # * fisrt do positional
            self.v_self_layers0.append(
                nn.Sequential(
                    nn.Linear(c_in + self.p_pe_dim, c_out // 2),
                    BN_lastdim(c_out // 2) if use_bn else nn.Identity(),
                    nn.LeakyReLU(),
                )
            )
            self.v_self_layers1.append(
                nn.Sequential(
                    nn.Linear(c_out // 2 + self.t_pe_dim, c_out // 2),
                    BN_lastdim(c_out // 2) if use_bn else nn.Identity(),
                    nn.LeakyReLU(),
                )
            )

            self.e_dir_conv_layers.append(
                nn.Sequential(
                    nn.Linear(self.e_d_dim, c_out // 2),
                    BN_lastdim(c_out // 2) if use_bn else nn.Identity(),
                    nn.LeakyReLU(),
                )
            )
            self.e_sym_conv_layers.append(
                nn.Sequential(
                    nn.Linear(self.e_s_dim, c_out // 2),
                    BN_lastdim(c_out // 2) if use_bn else nn.Identity(),
                    nn.LeakyReLU(),
                )
            )
            e_conv_in = c_out * 2
            if use_global_pooling:
                e_conv_in += c_in
            self.e_conv_layers.append(
                nn.Sequential(
                    nn.Linear(e_conv_in, c_out),
                    BN_lastdim(c_out) if use_bn else nn.Identity(),
                    nn.LeakyReLU(),
                    nn.Linear(c_out, c_out),
                    BN_lastdim(c_out) if use_bn else nn.Identity(),
                    nn.LeakyReLU(),
                )
            )
            self.v_self_layers2.append(
                nn.Sequential(
                    nn.Linear(c_out + c_in, c_out),
                    BN_lastdim(c_out) if use_bn else nn.Identity(),
                    nn.LeakyReLU(),
                )
            )

            c_in = c_out
            out_cnt += c_out
            edge_wide_cnt += c_out

        if self.use_wide:
            self.wide_fc = nn.Linear(out_cnt, c_out)

        self.node_mlp = NaiveMLP(c_in, self.v_dim, node_pred_head_dims, use_bn=use_bn)
        self.edge_sym_mlp = NaiveMLP(
            c_in * 2 + edge_wide_cnt + 10, self.e_s_dim, edge_sym_head_dims, use_bn=use_bn
        )
        self.edge_dir_mlp = NaiveMLP(
            c_in * 2 + edge_wide_cnt + 3, self.e_d_dim, edge_dir_head_dims, use_bn=use_bn
        )

        self.tri_ind_to_full_ind, self.src_dst_ind = tri_ind_to_full_ind(self.K)

        # for edge connectivity guidance
        self.edge_guide_factor = edge_guide_factor
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

    def forward(self, V_noisy, E_noisy, V_mask, t):
        # first get symmetry nodes
        B, K, _ = V_noisy.shape
        V = V_noisy

        E_sym, E_dir = E_noisy[..., 3:], E_noisy[..., :3]
        E_sym = self.scatter_trilist_to_mtx(E_sym)
        E_sym = E_sym + E_sym.permute(0, 2, 1, 3)
        # * swap the 1 and 2 slots for type
        E_dir = self.scatter_trilist_to_mtx(E_dir)
        E_dir_negative = E_dir[..., [0, 2, 1]]
        E_dir = E_dir + E_dir_negative.permute(0, 2, 1, 3)

        # ! use current valid edge to guide the aggregation
        E_cur_connect = 1.0 - E_dir[..., 0]  # if empty zero
        diag_mask = torch.eye(K, device=V.device)[None, ...]
        E_cur_connect = torch.sigmoid((E_cur_connect - 0.5) * self.edge_guide_factor)
        E_cur_connect = E_cur_connect * (1.0 - diag_mask)

        E_sym = E_sym.reshape(B, K**2, -1)
        E_dir = E_dir.reshape(B, K**2, -1)

        emb_t = self.t_pe[t]  # B,PE
        if self.p_pe_dim > 0:
            emb_p = self.p_pe.clone().unsqueeze(0).expand(B, -1, -1)  # B,K,PE

        # ! warning, here only support first K mask
        valid_agg_mask = V_mask.permute(0, 2, 1).expand(-1, K, -1).float()
        diag_mask = torch.eye(K, device=V.device)[None, ...]
        valid_agg_mask = valid_agg_mask * (1.0 - diag_mask)
        # ! use current connectivity to guide
        valid_agg_mask = valid_agg_mask * E_cur_connect

        node_f = V
        node_f_list = [node_f]
        edge_f_list = []
        for i in range(len(self.v_self_layers0)):
            # process self
            _pe_t = self.te_layers[i](emb_t)
            _pe_t = _pe_t.unsqueeze(1).expand(-1, K, -1)
            if self.p_pe_dim > 0:
                _pe_p = self.pe_layers[i](emb_p)
                _node_f = torch.cat([node_f, _pe_p], -1)
            else:
                _node_f = node_f
            _node_f = self.v_self_layers0[i](_node_f)
            _node_f = self.v_self_layers1[i](torch.cat([_node_f, _pe_t], -1))
            # process edges
            # During aggregation, the row is dst, each col is a neighbor
            e_f_sym = self.e_sym_conv_layers[i](E_sym)
            e_f_dir = self.e_dir_conv_layers[i](E_dir)
            node_f_self = _node_f[:, :, None, :].expand(-1, -1, K, -1)
            node_f_neighbor = _node_f[:, None, :, :].expand(-1, K, -1, -1)
            e_f_node = torch.cat([node_f_self, node_f_neighbor], -1).reshape(B, K * K, -1)
            e_f = torch.cat([e_f_sym, e_f_dir, e_f_node], -1)
            if self.use_global_pooling:
                pooled = (node_f * V_mask).sum(1) / (V_mask.sum(1) + 1e-6)
                pooled = pooled[:, None, :].expand(-1, K**2, -1)
                e_f = torch.cat([e_f, pooled], -1)
            e_f = self.e_conv_layers[i](e_f)
            e_f = e_f.reshape(B, K, K, -1)
            edge_f_list.append(e_f.clone())
            # mask out the diagonal and invalid nodes col
            # gather edge features
            e_f = e_f * valid_agg_mask[..., None]
            # pooling
            agg_f = e_f.sum(2) / (valid_agg_mask.sum(2) + 1e-6)[..., None]
            node_f = self.v_self_layers2[i](torch.cat([node_f, agg_f], -1))
            if self.use_wide:
                node_f_list.append(node_f)
        if self.use_wide:
            node_f = torch.cat(node_f_list, -1)
            node_f = self.wide_fc(node_f)
        edge_f_list = torch.cat(edge_f_list, -1)  # B,K,K,ALL EF

        v_pred = self.node_mlp(node_f)
        src_dst_ind = self.src_dst_ind.to(V.device)
        src_ind = src_dst_ind[0][None, :, None].expand(B, -1, node_f.shape[-1])
        dst_ind = src_dst_ind[1][None, :, None].expand(B, -1, node_f.shape[-1])
        src_f = torch.gather(node_f, 1, src_ind)
        dst_f = torch.gather(node_f, 1, dst_ind)

        edge_f_dir = torch.cat([E_dir.reshape(B, K, K, -1), edge_f_list], -1)
        edge_f_sym = (edge_f_list + edge_f_list.permute(0, 2, 1, 3)) / 2.0
        edge_f_sym = torch.cat([edge_f_sym, E_sym.reshape(B, K, K, -1)], -1)

        gather_ind = self.tri_ind_to_full_ind.to(V.device)[None, :, None]
        edge_f_dir = torch.gather(
            edge_f_dir.reshape(B, K**2, -1), 1, gather_ind.expand(B, -1, edge_f_dir.shape[-1])
        )
        edge_f_sym = torch.gather(
            edge_f_sym.reshape(B, K**2, -1), 1, gather_ind.expand(B, -1, edge_f_sym.shape[-1])
        )

        e_f = torch.stack([src_f, dst_f], -1)
        e_f_sym = torch.cat([e_f.max(-1).values, e_f.mean(-1)], -1)
        e_f_dir = e_f.reshape(B, e_f.shape[1], 2 * node_f.shape[-1])

        e_f_sym = torch.cat([e_f_sym, edge_f_sym], -1)
        e_f_dir = torch.cat([e_f_dir, edge_f_dir], -1)

        e_pred_sym = self.edge_sym_mlp(e_f_sym)
        e_pred_dir = self.edge_dir_mlp(e_f_dir)

        e_pred = torch.cat([e_pred_dir, e_pred_sym], -1)
        return v_pred, e_pred
