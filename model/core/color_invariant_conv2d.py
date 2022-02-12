"""
Copy from CIConv
"""

# Import generatl dependencies
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_basis_filters(scale, kernel_size=3):
    std = torch.pow(2, scale)
    # Define the basis vector for the current scale
    filter_size = torch.ceil(kernel_size * std + 0.5).item()
    x = torch.arange(start=-filter_size, end=filter_size + 1).cuda()

    x = x.cuda()
    std = std.cuda()
    x = torch.meshgrid([x, x])
    # Calculate Gaussian filter base
    # Only exponent part of Gaussian function since it is normalized anyway
    g = torch.exp(-(x[0] / std)**2 / 2) * torch.exp(-(x[1] / std)**2 / 2)
    g = g / torch.sum(g)  # Normalize

    # Gaussian derivative dg/dx filter base
    dgdx = -x[0] / (std**3 * 2 * math.pi) * torch.exp(
        -(x[0] / std)**2 / 2) * torch.exp(-(x[1] / std)**2 / 2)
    dgdx = dgdx / torch.sum(torch.abs(dgdx))  # Normalize

    # Gaussian derivative dg/dy filter base
    dgdy = -x[1] / (std**3 * 2 * math.pi) * torch.exp(
        -(x[1] / std)**2 / 2) * torch.exp(-(x[0] / std)**2 / 2)
    dgdy = dgdy / torch.sum(torch.abs(dgdy))

    # Stack and expand dim
    basis_filter = torch.stack([g, dgdx, dgdy], dim=0)[:, None, :, :]

    return basis_filter


# =================================
# == Color invariant definitions ==
# =================================

eps = 1e-5


def E_inv(E, Ex, Ey, El, Elx, Ely, Ell, Ellx, Elly):
    E = Ex**2 + Ey**2 + Elx**2 + Ely**2 + Ellx**2 + Elly**2
    return E


def W_inv(E, Ex, Ey, El, Elx, Ely, Ell, Ellx, Elly):
    Wx = Ex / (E + eps)
    Wlx = Elx / (E + eps)
    Wllx = Ellx / (E + eps)
    Wy = Ey / (E + eps)
    Wly = Ely / (E + eps)
    Wlly = Elly / (E + eps)

    W = Wx**2 + Wlx**2 + Wllx**2 + Wy**2 + Wly**2 + Wlly**2
    return W


def C_inv(E, Ex, Ey, El, Elx, Ely, Ell, Ellx, Elly):
    Clx = (Elx * E - El * Ex) / (E**2 + eps)
    Cly = (Ely * E - El * Ey) / (E**2 + eps)
    Cllx = (Ellx * E - Ell * Ex) / (E**2 + eps)
    Clly = (Elly * E - Ell * Ey) / (E**2 + eps)

    C = Clx**2 + Cly**2 + Cllx**2 + Clly**2
    return C


def N_inv(E, Ex, Ey, El, Elx, Ely, Ell, Ellx, Elly):
    Nlx = (Elx * E - El * Ex) / (E**2 + eps)
    Nly = (Ely * E - El * Ey) / (E**2 + eps)
    Nllx = (Ellx * E**2 - Ell * Ex * E - 2 * Elx * El * E +
            2 * El**2 * Ex) / (E**3 + eps)
    Nlly = (Elly * E**2 - Ell * Ey * E - 2 * Ely * El * E +
            2 * El**2 * Ey) / (E**3 + eps)

    N = Nlx**2 + Nly**2 + Nllx**2 + Nlly**2
    return N


def H_inv(E, Ex, Ey, El, Elx, Ely, Ell, Ellx, Elly):
    Hx = (Ell * Elx - El * Ellx) / (El**2 + Ell**2 + eps)
    Hy = (Ell * Ely - El * Elly) / (El**2 + Ell**2 + eps)
    H = Hx**2 + Hy**2
    return H


# =================================
# == Color invariant convolution ==
# =================================

inv_switcher = {'E': E_inv, 'W': W_inv, 'C': C_inv, 'N': N_inv, 'H': H_inv}


class CIConv2d(nn.Module):

    def __init__(self, invariant: str, kernel_size=3, scale=0.0):
        super(CIConv2d, self).__init__()
        assert invariant in ['E', 'W', 'C', 'N', 'H'], 'invalid invariant'
        self.inv_function = inv_switcher[invariant]

        self.use_cuda = torch.cuda.is_available()
        self.gpu = torch.cuda.current_device() if self.use_cuda else None

        # Constants
        self.gcm = torch.tensor([[0.06, 0.63, 0.27], [0.3, 0.04, -0.35],
                                 [0.34, -0.6, 0.17]])
        # if self.use_cuda:
        #     self.gcm = self.gcm.cuda()
        self.kernel_size = kernel_size

        # Learnable parameters
        self.scale = torch.nn.Parameter(torch.tensor([scale]),
                                        requires_grad=True)

    def forward(self, batch):
        # Make sure scale does not explode: clamp to max abs value of 2.5
        self.scale.data = torch.clamp(self.scale.data, min=-2.5, max=2.5)

        # Measure E, El, Ell by Gaussian color model
        # NCHW
        in_shape = batch.shape
        # flatten image
        batch = batch.view((in_shape[:2] + (-1, )))
        # estimate E, EL, Ell
        batch = torch.matmul(self.gcm.cuda(torch.cuda.current_device()), batch)
        # reshape to original image size
        batch = batch.view((in_shape[0], ) + (3, ) + in_shape[2:])
        E, El, Ell = torch.split(batch, 1, dim=1)
        # Convolve with Gaussian filters
        # KCHW
        w = gaussian_basis_filters(scale=self.scale)
        # the padding here works as "same" for odd kernel sizes
        E_out = F.conv2d(input=E, weight=w, padding=int(w.shape[2] / 2))
        El_out = F.conv2d(input=El, weight=w, padding=int(w.shape[2] / 2))
        Ell_out = F.conv2d(input=Ell, weight=w, padding=int(w.shape[2] / 2))

        E, Ex, Ey = torch.split(E_out, 1, dim=1)
        El, Elx, Ely = torch.split(El_out, 1, dim=1)
        Ell, Ellx, Elly = torch.split(Ell_out, 1, dim=1)

        inv_out = self.inv_function(E, Ex, Ey, El, Elx, Ely, Ell, Ellx, Elly)
        inv_out = F.instance_norm(torch.log(inv_out + eps))

        return inv_out
