# V5: from v2, remove the v_mask, every node is valid
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


class GraphLayer(nn.Module):
    def __init__(
        self,
        v_in_dim,
        v_out_dim,
        e_in_dim,
        e_out_dim,
        p_pe_dim,
        t_pe_dim,
        use_bn=False,
        c_atten_head=8,
        # for ablation
        use_transformer=True,
        use_graph_conv=True,
        node_dominated=False,
        edge_dominate=False,
    ) -> None:
        super().__init__()
        self.v_in, self.v_out = v_in_dim, v_out_dim
        self.e_in, self.e_out = e_in_dim, e_out_dim
        self.p_pe_dim, self.t_pe_dim = p_pe_dim, t_pe_dim
        self.c_atten_head = c_atten_head

        self.use_transformer = use_transformer
        if not self.use_transformer:
            logging.warning("Not using transformer in GraphLayer")
        self.use_graph_conv = use_graph_conv
        if not self.use_graph_conv:
            logging.warning("Not using graph conv in GraphLayer")
        self.node_dominated = node_dominated
        if self.node_dominated:
            logging.warning("Using node dominated in GraphLayer")
        self.edge_dominate = edge_dominate
        if self.edge_dominate:
            logging.warning("Using edge dominated in GraphLayer")
            assert not self.node_dominated

        self.v_pe_binding_fc = nn.Sequential(
            nn.Linear(self.v_in + self.t_pe_dim + self.p_pe_dim, self.v_in),
            BN_lastdim(self.v_in) if use_bn else nn.Identity(),
            nn.SiLU(),
        )
        self.e_pe_binding_fc = nn.Sequential(
            nn.Linear(self.e_in + self.t_pe_dim, self.e_in),
            BN_lastdim(self.e_in) if use_bn else nn.Identity(),
            nn.SiLU(),
        )

        if self.use_graph_conv:
            if self.node_dominated:
                self.edge_value_fc = nn.Sequential(
                    nn.Linear(2 * self.v_in, self.e_out),
                    BN_lastdim(self.e_in) if use_bn else nn.Identity(),
                    nn.SiLU(),
                    nn.Linear(self.e_out, self.e_out),
                )
                self.edge_output = nn.Sequential(
                    nn.Linear(self.e_out + self.e_in, self.e_out),
                    BN_lastdim(self.e_out) if use_bn else nn.Identity(),
                    nn.SiLU(),
                    nn.Linear(self.e_out, self.e_out),
                )
            else:
                self.edge_value_fc = nn.Sequential(
                    nn.Linear(self.e_in + 2 * self.v_in, self.e_out),
                    BN_lastdim(self.e_in) if use_bn else nn.Identity(),
                    nn.SiLU(),
                    nn.Linear(self.e_out, self.e_out),
                )
            if self.edge_dominate:
                # Edge need info propagation between each other, not from the node
                self.edge_output = nn.Sequential(
                    nn.Linear(self.e_in * 2, self.e_out),
                    BN_lastdim(self.e_in) if use_bn else nn.Identity(),
                    nn.SiLU(),
                    nn.Linear(self.e_out, self.e_out),
                )
            if self.use_transformer:
                self.node_query_fc = nn.Linear(self.v_in, self.e_out)
                self.node_key_fc = nn.Linear(self.v_in, self.e_out)
            self.node_self = nn.Sequential(
                nn.Linear(self.v_in, self.e_out),
                BN_lastdim(self.e_out) if use_bn else nn.Identity(),
                nn.SiLU(),
            )
            self.node_out = nn.Sequential(
                nn.Linear(2 * self.e_out, self.v_out),
                BN_lastdim(self.v_out) if use_bn else nn.Identity(),
                nn.SiLU(),
            )
        else:  # use a point net to do global talk
            self.fc_v = nn.Sequential(
                nn.Linear(self.v_in, self.v_out),
                BN_lastdim(self.e_out) if use_bn else nn.Identity(),
                nn.SiLU(),
            )
            self.fc_e = nn.Sequential(
                nn.Linear(self.v_in, self.v_out),
                BN_lastdim(self.e_out) if use_bn else nn.Identity(),
                nn.SiLU(),
            )
            self.fc_v_out = nn.Sequential(
                nn.Linear(self.v_out * 2 + self.e_out, self.v_out),
                BN_lastdim(self.v_out) if use_bn else nn.Identity(),
                nn.SiLU(),
            )
            self.fc_e_out = nn.Sequential(
                nn.Linear(self.e_out * 2 + self.v_out, self.e_out),
                BN_lastdim(self.e_out) if use_bn else nn.Identity(),
                nn.SiLU(),
            )

        return

    def forward(self, v_f, e_f, p_pe, t_pe, e_agg_mask=None, v_agg_mask=None):
        # v_f: B,K,F, e_f: B,K^2,F, p_pe: B,K,F; t_pe: B,F
        B, K, _ = v_f.shape
        # first bind the PEs
        if p_pe is None:
            v_f = self.v_pe_binding_fc(
                torch.cat([v_f, t_pe[:, None, :].expand(-1, v_f.shape[1], -1)], -1)
            )
        else:
            v_f = self.v_pe_binding_fc(
                torch.cat([v_f, p_pe, t_pe[:, None, :].expand(-1, v_f.shape[1], -1)], -1)
            )
        e_f = self.e_pe_binding_fc(
            torch.cat([e_f, t_pe[:, None, :].expand(-1, e_f.shape[1], -1)], -1)
        )

        if not self.use_graph_conv:
            # ablation
            v_f = self.fc_v(v_f)
            e_f = self.fc_e(e_f)
            v_pool = v_f.max(1)[0]
            e_pool = e_f.max(1)[0]
            pooled = torch.cat([v_pool, e_pool], -1)[:, None, :]
            v_f = self.fc_v_out(torch.cat([v_f, pooled.expand(-1, v_f.shape[1], -1)], -1))
            e_f = self.fc_e_out(torch.cat([e_f, pooled.expand(-1, e_f.shape[1], -1)], -1))
            return v_f, e_f

        # get edge Value
        e_f_self = v_f[:, :, None, :].expand(-1, -1, K, -1)
        e_f_neighbor = v_f[:, None, :, :].expand(-1, K, -1, -1)
        e_v_feat = torch.cat([e_f_self, e_f_neighbor], -1).reshape(B, K * K, -1)
        if self.node_dominated:
            # only depends on the nodes feature
            e_value = self.edge_value_fc(torch.cat([e_v_feat], -1))
        else:
            e_value = self.edge_value_fc(torch.cat([e_v_feat, e_f], -1))
        agg_e_value = e_value.reshape(B, K, K, -1)

        # get node Q,K, self-attention and agg mask
        if self.use_transformer:
            v_query = self.node_query_fc(v_f)[:, :, None, :].expand(-1, -1, K, -1)
            v_key = self.node_key_fc(v_f)[:, None, :, :].expand(-1, K, -1, -1)
            atten_mtx = v_query * v_key
            atten_head = atten_mtx.reshape(B, K, K, -1, self.c_atten_head).sum(
                -1, keepdim=True
            ) / np.sqrt(self.c_atten_head * 3.0)
            # ! warning, here not using softmax
            if e_agg_mask is not None:
                atten_head = torch.sigmoid(atten_head)
                atten_head = atten_head * e_agg_mask[..., None, None]
                atten_head = atten_head / (atten_head.sum(2, keepdim=True) + 1e-6)
            else:
                atten_head = torch.softmax(atten_head, 2)
            atten_head = atten_head.expand(-1, -1, -1, -1, self.c_atten_head)
            atten_head = atten_head.reshape(B, K, K, -1)
            # agg
            agg_v_f = (atten_head * agg_e_value).sum(2)
        else:
            agg_v_f = agg_e_value.mean(2)

        self_v_f = self.node_self(v_f)
        v_f = agg_v_f + self_v_f

        # global
        if v_agg_mask is not None:
            v_pooling = v_f - 1e9 * (1.0 - v_agg_mask)[..., None]
            v_pooling = v_pooling.max(1, keepdim=True).values.expand(-1, K, -1)
        else:
            v_pooling = v_f.max(1, keepdim=True).values.expand(-1, K, -1)
        v_f = self.node_out(torch.cat([v_f, v_pooling], -1))

        # output
        if self.node_dominated:
            e_value = self.edge_output(torch.cat([e_value, e_f], -1))
        elif self.edge_dominate:
            e_f_mtx = e_f.reshape(B, K, K, -1)
            e_f_mtx_pool1 = e_f_mtx.mean(1, keepdim=True).expand(-1, K, -1, -1)
            e_f_mtx_pool2 = e_f_mtx.mean(2, keepdim=True).expand(-1, -1, K, -1)
            e_f_mtx_pool = (e_f_mtx_pool1 + e_f_mtx_pool2)/2.0
            e_f_mtx_pool = e_f_mtx_pool.reshape(B, K * K, -1)
            e_value = self.edge_output(torch.cat([e_value, e_f_mtx_pool], -1))
            
        return v_f, e_value


class GraphDenoiseConvV60(torch.nn.Module):
    # A large upgrade
    def __init__(
        self,
        K,  #
        M,  # time steps
        V_dims: list = [1, 3, 3, 128],
        E_sym_dims: list = [6, 4],
        E_dir_dim: int = 3,
        # Backbone
        v_conv_dims=[256, 256 + 16, 256 + 32, 256],
        e_conv_dims=[512, 512 + 32, 512 + 16, 512],
        p_pe_dim: int = 200,
        t_pe_dim: int = 100,
        use_bn=False,
        c_atten_head=16,
        v_out_hidden_dims=[128, 128],
        e_out_hidden_dims=[128, 64],
        # others
        connectivity_guidance_factor: int = 10.0,
        # for ablation
        E_guidance=False,
        use_transformer=True,
        use_graph_conv=True,
        dir_handling=True,
        node_dominated=False,
        edge_dominate=False,
    ) -> None:
        super().__init__()

        # ! for ablation
        self.E_guidance_flag = E_guidance
        if self.E_guidance_flag:
            logging.warning("E_guidance is set to True, will use edge guidance")
        self.dir_handling = dir_handling
        if not self.dir_handling:
            logging.warning("dir_handling is set to False, will not use dir handling")
        # ! END ablation
        self.K, self.M = K, M

        self.V_dims = V_dims
        self.E_s_dims, self.E_d_dim = E_sym_dims, E_dir_dim
        self.p_pe_dim, self.t_pe_dim = p_pe_dim, t_pe_dim
        self.v_conv_dims, self.e_conv_dims = v_conv_dims, e_conv_dims
        assert len(self.v_conv_dims) == len(self.e_conv_dims)

        self.register_buffer("t_pe", self.sinusoidal_embedding(M, self.t_pe_dim))
        if self.p_pe_dim > 0:
            self.register_buffer("p_pe", self.sinusoidal_embedding(K, self.p_pe_dim))
        self.tri_ind_to_full_ind, self.src_dst_ind = tri_ind_to_full_ind(self.K)

        # input layers
        self.v_in_layers = nn.ModuleList(
            [nn.Linear(v_in, self.v_conv_dims[0]) for v_in in self.V_dims]
        )
        self.e_in_layers = nn.ModuleList(
            [nn.Linear(e_in, self.e_conv_dims[0]) for e_in in [self.E_d_dim] + self.E_s_dims]
        )

        self.layers = nn.ModuleList()
        v_in, e_in = self.v_conv_dims[0], self.e_conv_dims[0]
        v_wide_c, e_wide_c = v_in, e_in
        for v_out, e_out in zip(self.v_conv_dims[1:], self.e_conv_dims[1:]):
            self.layers.append(
                GraphLayer(
                    v_in,
                    v_out,
                    e_in,
                    e_out,
                    p_pe_dim,
                    t_pe_dim,
                    use_bn=use_bn,
                    c_atten_head=c_atten_head,
                    use_transformer=use_transformer,
                    use_graph_conv=use_graph_conv,
                    node_dominated=node_dominated,
                    edge_dominate=edge_dominate,
                )
            )
            v_in, e_in = v_out, e_out
            v_wide_c += v_in
            e_wide_c += e_in

        self.v_wide_fc = nn.Linear(v_wide_c, self.v_conv_dims[-1])
        self.e_wide_fc = nn.Linear(e_wide_c, self.e_conv_dims[-1])

        # out layers
        self.v_out_layers, self.e_out_layers = nn.ModuleList(), nn.ModuleList()
        for c_out in self.V_dims:
            self.v_out_layers.append(
                NaiveMLP(self.v_conv_dims[-1] + c_out, c_out, v_out_hidden_dims, use_bn=use_bn)
            )
        for c_out in [self.E_d_dim] + self.E_s_dims:
            self.e_out_layers.append(
                NaiveMLP(
                    self.e_conv_dims[-1] + c_out,
                    c_out,
                    e_out_hidden_dims,
                    use_bn=use_bn,
                )
            )

        # for edge connectivity guidance
        self.edge_guide_factor = connectivity_guidance_factor
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

    def get_E_mask_gt(self, V_mask_gt, return_trilist=True):
        B = V_mask_gt.shape[0]
        E_mask_gt = ((V_mask_gt[:, :, None] + V_mask_gt[:, None, :]) > 0.0).float()
        if return_trilist:
            gather_ind = self.tri_ind_to_full_ind.clone()
            E_mask_gt = E_mask_gt.reshape(B, -1)
            E_mask_gt = torch.gather(
                E_mask_gt, 1, gather_ind[None, :].expand(B, -1).to(E_mask_gt.device)
            )
        return E_mask_gt

    def forward(self, V_noisy, E_noisy, t, V_mask=None):
        # first get symmetry nodes
        B, K, _ = V_noisy.shape
        if V_mask is not None:
            # First mask!
            _E_mask = self.get_E_mask_gt(V_mask, return_trilist=True)
            E_noisy = E_noisy * _E_mask[..., None]
            V_noisy = V_noisy * V_mask[..., None]

        V = V_noisy
        E_sym, E_dir = E_noisy[..., 3:], E_noisy[..., :3]
        E_sym = self.scatter_trilist_to_mtx(E_sym)
        E_sym = E_sym + E_sym.permute(0, 2, 1, 3)
        # * swap the 1 and 2 slots for type
        E_dir = self.scatter_trilist_to_mtx(E_dir)
        if self.dir_handling:
            E_dir_negative = E_dir[..., [0, 2, 1]]
            E_dir = E_dir + E_dir_negative.permute(0, 2, 1, 3)

        # ! use current valid edge to guide the aggregation
        if self.E_guidance_flag:
            E_cur_connect = 1.0 - E_dir[..., 0]  # if empty zero
            diag_mask = torch.eye(K, device=V.device)[None, ...]
            E_cur_connect = torch.sigmoid((E_cur_connect - 0.5) * self.edge_guide_factor)
            E_cur_connect = E_cur_connect * (1.0 - diag_mask)
            if V_mask is not None:
                assert V_mask.ndim == 2
                E_valid_v = self.get_E_mask_gt(V_mask, return_trilist=False)
                E_cur_connect = E_cur_connect * E_valid_v
                # E_sym = E_sym * E_valid_v[..., None] # no need to mask again
                # E_dir = E_dir * E_valid_v[..., None]
        else:
            E_cur_connect = None

        E_sym = E_sym.reshape(B, K**2, -1)
        E_dir = E_dir.reshape(B, K**2, -1)

        # prepare PE
        emb_t = self.t_pe[t]  # B,PE
        if self.p_pe_dim > 0:
            emb_p = self.p_pe.clone().unsqueeze(0).expand(B, -1, -1)  # B,K,PE
        else:
            emb_p = None

        # prepare init feature
        cur = 0
        node_f = None
        for i, v_in_c in enumerate(self.V_dims):
            v_input = V[..., cur : cur + v_in_c]
            if node_f is None:
                node_f = self.v_in_layers[i](v_input)
            else:
                node_f = node_f + self.v_in_layers[i](v_input)
            cur += v_in_c
        edge_f = self.e_in_layers[0](E_dir)
        cur = 3
        for i, e_in_c in enumerate(self.E_s_dims):
            e_input = E_sym[..., cur : cur + e_in_c]
            edge_f = edge_f + self.e_in_layers[i + 1](e_input)
        # node_f: B,K,F; edge_f: B,K^2,F

        node_f_list, edge_f_list = [node_f], [edge_f]
        for i in range(len(self.layers)):
            node_f, edge_f = self.layers[i](
                node_f, edge_f, emb_p, emb_t, e_agg_mask=E_cur_connect, v_agg_mask=V_mask
            )
            node_f_list.append(node_f)
            edge_f_list.append(edge_f)

        node_f = torch.cat(node_f_list, -1)
        edge_f_list = torch.cat(edge_f_list, -1)  # B,K,K,ALL EF
        node_f = self.v_wide_fc(node_f)
        edge_f_list = self.e_wide_fc(edge_f_list).reshape(B, K, K, -1)

        # prepare final feature
        edge_f_dir = edge_f_list
        if self.dir_handling:
            edge_f_sym = (edge_f_list + edge_f_list.permute(0, 2, 1, 3)) / 2.0
        else:
            edge_f_sym = edge_f_list
        gather_ind = self.tri_ind_to_full_ind.to(V.device)[None, :, None]
        edge_f_dir = torch.gather(
            edge_f_dir.reshape(B, K**2, -1), 1, gather_ind.expand(B, -1, edge_f_dir.shape[-1])
        )
        
        # ! Warning, below is a flaw found after the paper submission, actually the edge_f_sym is not used; but since the network is not permutation equivariant, practically this is fine to make the prediction, the network never guarantees that when flipping the parent, child order, the undirected edge prediction will be the same. But setting the predicting target to be un-directional may still be helpful for training.
        edge_f_sym = torch.gather(
            edge_f_sym.reshape(B, K**2, -1), 1, gather_ind.expand(B, -1, edge_f_sym.shape[-1])
        )

        # final output
        cur = 0
        v_pred = []
        for i in range(len(self.V_dims)):
            v_pred.append(
                self.v_out_layers[i](
                    torch.cat([node_f, V_noisy[..., cur : cur + self.V_dims[i]]], -1)
                )
            )
            cur += self.V_dims[i]
        v_pred = torch.cat(v_pred, -1)

        e_pred = [self.e_out_layers[0](torch.cat([edge_f_dir, E_noisy[..., : self.E_d_dim]], -1))]
        cur = self.E_d_dim
        for i in range(len(self.E_s_dims)):
            e_pred.append(
                self.e_out_layers[i + 1](
                    torch.cat([edge_f_dir, E_noisy[..., cur : cur + self.E_s_dims[i]]], -1)
                )
            )
            cur += self.E_s_dims[i]
        e_pred = torch.cat(e_pred, -1)
        return v_pred, e_pred
