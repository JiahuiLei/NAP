"""
PointNet ResNet
from https://github.com/autonomousvision/occupancy_flow
"""

import torch
import torch.nn as nn


# Resnet Blocks
class ResnetBlockFC(nn.Module):
    """Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    """

    def __init__(self, size_in, size_out=None, size_h=None, act_method=nn.ReLU):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = act_method()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class ResnetBlockFC_BN(nn.Module):
    """Fully connected ResNet Block class.
    # ! use BN
    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    """

    def __init__(self, size_in, size_out=None, size_h=None, act_method=nn.ReLU):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.bn_0 = nn.BatchNorm1d(size_in)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.bn_1 = nn.BatchNorm1d(size_h)
        self.actvn = act_method()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        x = self.bn_0(x.permute(0, 2, 1)).permute(0, 2, 1)
        net = self.fc_0(self.actvn(x))
        net = self.bn_1(net.permute(0, 2, 1)).permute(0, 2, 1)
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


def maxpool(x, dim=-1, keepdim=False):
    """Performs a maxpooling operation.

    Args:
        x (tensor): input
        dim (int): dimension of pooling
        keepdim (bool): whether to keep dimensions
    """
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out


class ResnetPointnet(nn.Module):
    """PointNet-based encoder network with ResNet blocks.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    """

    def __init__(self, c_dim=128, dim=3, hidden_dim=512):
        super().__init__()
        self.c_dim = c_dim

        self.fc_pos = nn.Linear(dim, 2 * hidden_dim)
        self.block_0 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_1 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_2 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_3 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_4 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.pool = maxpool

    def forward(self, p, return_unpooled=False):
        batch_size, T, D = p.size()

        # output size: B x T X F
        net = self.fc_pos(p)
        net = self.block_0(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_1(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_2(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_3(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_4(net)

        # Recude to  B x F
        ret = self.pool(net, dim=1)

        c = self.fc_c(self.actvn(ret))

        if return_unpooled:
            return c, net
        else:
            return c


class AttentionPooling(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim
        self.K = ResnetBlockFC(dim, dim, dim)

    def forward(self, x, keepdim=True):
        # B,N,C, pool over dim 1
        assert x.ndim == 3 and x.shape[-1] == self.dim
        k = self.K(x.max(dim=1, keepdim=True).values)  # B,1,C
        alpha = (k * x).sum(dim=-1, keepdim=True)
        weight = torch.softmax(alpha, dim=1)
        pooled = (weight * x).sum(dim=1, keepdim=keepdim)
        return pooled


class ResnetPointnetAttenPooling(nn.Module):
    """PointNet-based encoder network with ResNet blocks.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    """

    def __init__(self, c_dim=128, dim=3, hidden_dim=512):
        super().__init__()
        self.c_dim = c_dim

        self.fc_pos = nn.Linear(dim, 2 * hidden_dim)
        self.block_0 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.atten_pool_0 = AttentionPooling(hidden_dim)
        self.block_1 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.atten_pool_1 = AttentionPooling(hidden_dim)
        self.block_2 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.atten_pool_2 = AttentionPooling(hidden_dim)
        self.block_3 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.atten_pool_3 = AttentionPooling(hidden_dim)
        self.block_4 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.atten_pool_4 = AttentionPooling(hidden_dim)
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.pool = maxpool

    def forward(self, p, return_unpooled=False):
        batch_size, T, D = p.size()

        # output size: B x T X F
        net = self.fc_pos(p)
        net = self.block_0(net)
        pooled = self.atten_pool_0(net, keepdim=True).expand_as(net)
        net = torch.cat([net, pooled], dim=2)

        net = self.block_1(net)
        pooled = self.atten_pool_1(net, keepdim=True).expand_as(net)
        net = torch.cat([net, pooled], dim=2)

        net = self.block_2(net)
        pooled = self.atten_pool_2(net, keepdim=True).expand_as(net)
        net = torch.cat([net, pooled], dim=2)

        net = self.block_3(net)
        pooled = self.atten_pool_3(net, keepdim=True).expand_as(net)
        net = torch.cat([net, pooled], dim=2)

        net = self.block_4(net)

        # Recude to  B x F
        ret = self.atten_pool_4(net, keepdim=False)

        c = self.fc_c(self.actvn(ret))

        if return_unpooled:
            return c, net
        else:
            return c


class ResnetPointnetMasked(nn.Module):
    """PointNet-based encoder network with ResNet blocks.
    Masked mean pooling

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    """

    def __init__(self, c_dim=128, dim=3, hidden_dim=512, use_bn=False):
        super().__init__()
        self.c_dim = c_dim
        if use_bn:
            blk_class = ResnetBlockFC_BN
        else:
            blk_class = ResnetBlockFC

        self.fc_pos = nn.Linear(dim, 2 * hidden_dim)
        self.block_0 = blk_class(2 * hidden_dim, hidden_dim)
        self.block_1 = blk_class(2 * hidden_dim, hidden_dim)
        self.block_2 = blk_class(2 * hidden_dim, hidden_dim)
        self.block_3 = blk_class(2 * hidden_dim, hidden_dim)
        self.block_4 = blk_class(2 * hidden_dim, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()

    def pool(self, x, m, keepdim=False):
        # the mask is along the T direction
        y = (x * m.float().unsqueeze(-1)).sum(1, keepdim=keepdim)
        return y

    def forward(self, p, m, return_unpooled=False):
        B, N, D = p.size()

        net = self.fc_pos(p)
        net = self.block_0(net)
        pooled = self.pool(net, m, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_1(net)
        pooled = self.pool(net, m, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_2(net)
        pooled = self.pool(net, m, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_3(net)
        pooled = self.pool(net, m, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_4(net)

        # Recude to  B x C
        ret = self.pool(net, m)

        c = self.fc_c(self.actvn(ret))

        if return_unpooled:
            return c, net
        else:
            return c


if __name__ == "__main__":
    device = torch.device("cuda")
    B, N = 5, 1024
    # net = ResnetPointnetAttenPooling(c_dim=128, dim=3, hidden_dim=256).to(device)
    net = ResnetPointnetMasked(c_dim=128, dim=3, hidden_dim=256).to(device)
    x = torch.rand(B, N, 3).to(device)
    m = (torch.rand(B, N) > 0.5).float().to(device)
    y = net(x, m)
    print(x.shape)
    print(y.shape)
