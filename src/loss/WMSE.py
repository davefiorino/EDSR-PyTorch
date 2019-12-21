import torch
import torch.nn as nn
import torch.nn.functional as F

# Sober operator from: https://torchgeometry.readthedocs.io/en/latest/_modules/kornia/filters/sobel.html

class Sobel(nn.Module):
    r"""Computes the Sobel operator and returns the magnitude per channel.

    Return:
        torch.Tensor: the sobel edge gradient maginitudes map.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Examples:
        >>> input = torch.rand(1, 3, 4, 4)
        >>> output = kornia.filters.Sobel()(input)  # 1x3x4x4
    """

    def __init__(self) -> None:
        super(Sobel, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                             .format(input.shape))
        # comput the x/y gradients
        edges: torch.Tensor = spatial_gradient(input)

        # unpack the edges
        gx: torch.Tensor = edges[:, :, 0]
        gy: torch.Tensor = edges[:, :, 1]

        # compute gradient maginitude
        magnitude: torch.Tensor = torch.sqrt(gx * gx + gy * gy)
        return magnitude

def sobel(input: torch.Tensor) -> torch.Tensor:
    r"""Computes the Sobel operator and returns the magnitude per channel.

    See :class:`~kornia.filters.Sobel` for details.
    """
    return Sobel()(input)


# WMSE: MSE weighted by sobel grandient map

def weighted_mse_loss(input, target, weight):
    return torch.sum(weight * torch.add(input, -target) ** 2)
    

class WMSE(nn.Module):

    def __init__(self):
        super(WMSE, self).__init__()

    def forward(self, X, Y):
        grad = sobel(Y)
        
        loss = weighted_mse_loss(X, Y, grad)
        return loss 