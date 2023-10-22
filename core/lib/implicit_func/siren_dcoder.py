# ! warning, naive implementation, not calibrated with their paper
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from .onet_decoder import CBatchNorm1d, CBatchNorm1d_legacy

# Resnet Blocks
class ResnetBlockFC(nn.Module):
    """Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    """

    def __init__(self, size_in, size_out=None, size_h=None, actvn=torch.cos):
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
        self.actvn = actvn

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


class CResnetBlockConv1d(nn.Module):
    """Conditional batch normalization-based Resnet block class.

    Args:
        c_dim (int): dimension of latend conditioned code c
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
        norm_method (str): normalization method
        legacy (bool): whether to use legacy blocks
    """

    def __init__(
        self,
        c_dim,
        size_in,
        size_h=None,
        size_out=None,
        norm_method="batch_norm",
        legacy=False,
        actvn=torch.cos,
    ):
        super().__init__()
        # Attributes
        if size_h is None:
            size_h = size_in
        if size_out is None:
            size_out = size_in

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        if not legacy:
            self.bn_0 = CBatchNorm1d(c_dim, size_in, norm_method=norm_method)
            self.bn_1 = CBatchNorm1d(c_dim, size_h, norm_method=norm_method)
        else:
            self.bn_0 = CBatchNorm1d_legacy(c_dim, size_in, norm_method=norm_method)
            self.bn_1 = CBatchNorm1d_legacy(c_dim, size_h, norm_method=norm_method)

        self.fc_0 = nn.Conv1d(size_in, size_h, 1)
        self.fc_1 = nn.Conv1d(size_h, size_out, 1)
        self.actvn = actvn

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv1d(size_in, size_out, 1, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x, c):
        net = self.fc_0(self.actvn(self.bn_0(x, c)))
        dx = self.fc_1(self.actvn(self.bn_1(net, c)))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class SIRENDecoder(nn.Module):
    """Basic Decoder network for OFlow class.
    # TODO: What's the difference between Z and C?????
    The decoder network maps points together with latent conditioned codes
    c and z to log probabilities of occupancy for the points. This basic
    decoder does not use batch normalization.

    Args:
        dim (int): dimension of input points
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): dimension of hidden size
        leaky (bool): whether to use leaky ReLUs as activation
    """

    def __init__(self, dim=3, z_dim=128, c_dim=128, hidden_size=128, out_dim=1, pe_n=4, **kwargs):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.dim = dim

        # Submodules
        self.pe_n = pe_n
        pe_dim = dim
        if self.pe_n > 0:
            self.sigma = np.pi * torch.pow(2, torch.linspace(0, self.pe_n - 1, self.pe_n))
            pe_dim = dim * (2 * self.pe_n + 1)

        self.fc_p = nn.Linear(pe_dim, hidden_size)

        if not z_dim == 0:
            self.fc_z = nn.Linear(z_dim, hidden_size)
        if not c_dim == 0:
            self.fc_c = nn.Linear(c_dim, hidden_size)

        self.block0 = ResnetBlockFC(hidden_size)
        self.block1 = ResnetBlockFC(hidden_size)
        self.block2 = ResnetBlockFC(hidden_size)
        self.block3 = ResnetBlockFC(hidden_size)
        self.block4 = ResnetBlockFC(hidden_size)

        self.fc_out = nn.Linear(hidden_size, out_dim)

        self.actvn = torch.cos

    def pe(self, x):
        device = x.device
        y = torch.cat(
            [
                x[..., None],
                torch.sin(x[:, :, :, None] * self.sigma[None, None, None].to(device)),
                torch.cos(x[:, :, :, None] * self.sigma[None, None, None].to(device)),
            ],
            dim=-1,
        ).reshape(x.shape[0], x.shape[1], -1)
        return y

    def forward(self, p, z=None, c=None, **kwargs):
        """Performs a forward pass through the network.

        Args:
            p (tensor): points tensor
            z (tensor): latent code z
            c (tensor): latent conditioned code c
        """
        batch_size = p.shape[0]
        p = p.view(batch_size, -1, self.dim)

        # fixed_pe
        if self.pe_n > 0:
            p = self.pe(p)
        net = self.fc_p(p)

        if self.z_dim != 0:
            net_z = self.fc_z(z).unsqueeze(1)
            net = net + net_z

        if self.c_dim != 0:
            net_c = self.fc_c(c).unsqueeze(1)
            net = net + net_c

        net = self.block0(net)
        net = self.block1(net)
        net = self.block2(net)
        net = self.block3(net)
        net = self.block4(net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)

        return out


class SIRENDecoderCBatchNorm(nn.Module):
    """Conditioned Batch Norm Decoder network for OFlow class.

    The decoder network maps points together with latent conditioned codes
    c and z to log probabilities of occupancy for the points. This decoder
    uses conditioned batch normalization to inject the latent codes.

    Args:
        dim (int): dimension of input points
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): dimension of hidden size
        leaky (bool): whether to use leaky ReLUs as activation

    """

    def __init__(
        self, dim=3, z_dim=128, c_dim=128, hidden_size=256, out_dim=1, legacy=False, pe_n=4,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.dim = dim
        if not z_dim == 0:
            self.fc_z = nn.Linear(z_dim, hidden_size)
        
        self.pe_n = pe_n
        pe_dim = dim
        if self.pe_n > 0:
            self.sigma = np.pi * torch.pow(2, torch.linspace(0, self.pe_n - 1, self.pe_n))
            pe_dim = dim * (2 * self.pe_n + 1)

        self.fc_p = nn.Conv1d(pe_dim, hidden_size, 1)
        self.block0 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block1 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block2 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block3 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block4 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)

        if not legacy:
            self.bn = CBatchNorm1d(c_dim, hidden_size)
        else:
            self.bn = CBatchNorm1d_legacy(c_dim, hidden_size)

        self.fc_out = nn.Conv1d(hidden_size, out_dim, 1)

        self.actvn = torch.cos
    
    def pe(self, x):
        device = x.device
        y = torch.cat(
            [
                x[..., None],
                torch.sin(x[:, :, :, None] * self.sigma[None, None, None].to(device)),
                torch.cos(x[:, :, :, None] * self.sigma[None, None, None].to(device)),
            ],
            dim=-1,
        ).reshape(x.shape[0], x.shape[1], -1)
        return y

    def forward(self, p, z, c, **kwargs):
        """Performs a forward pass through the network.

        Args:
            p (tensor): points tensor
            z (tensor): latent code z
            c (tensor): latent conditioned code c
        """
        
        # fixed_pe
        if self.pe_n > 0:
            p = self.pe(p)

        p = p.transpose(1, 2)
        batch_size, D, T = p.size()
        net = self.fc_p(p)

        if self.z_dim != 0:
            net_z = self.fc_z(z).unsqueeze(2)
            net = net + net_z

        net = self.block0(net, c)
        net = self.block1(net, c)
        net = self.block2(net, c)
        net = self.block3(net, c)
        net = self.block4(net, c)

        out = self.fc_out(self.actvn(self.bn(net, c)))
        out = out.squeeze(1)

        return out
