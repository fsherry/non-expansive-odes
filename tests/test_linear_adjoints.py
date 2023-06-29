import torch

from non_expansive_odes.models.linear import (
    Conv1dNormalised,
    Conv2dNormalised,
    Conv3dNormalised,
    LinearNormalised,
)

torch.manual_seed(13)


def test_linear_adjoint():
    x = torch.randn(100, 10)
    A = LinearNormalised(in_features=10, out_features=30)
    y = torch.randn(100, 30)
    with torch.no_grad():
        Ax = A.tangent(x)
        ATy = A.transpose(y)
    assert torch.isclose(torch.sum(y * Ax), torch.sum(ATy * x))


def test_conv1d_adjoint():
    input_size = (130,)
    in_channels = 10
    out_channels = 20
    x = torch.randn(100, in_channels, *input_size)
    padding = (13,)
    kernel_size = (8,)
    stride = (7,)  # (7,)
    dilation = (3,)  # (3,)
    conv = Conv1dNormalised(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        input_size=input_size,
        padding=padding,
        stride=stride,
        dilation=dilation,
    )
    with torch.no_grad():
        Ax = conv.tangent(x)
        y = torch.randn_like(Ax)
        ATy = conv.transpose(y)
    assert torch.isclose(torch.sum(y * Ax), torch.sum(ATy * x))


def test_conv2d_adjoint():
    input_size = (25, 83)
    in_channels = 2
    out_channels = 4
    x = torch.randn(100, in_channels, *input_size)
    padding = (13, 49)
    kernel_size = (8, 34)
    stride = (2, 5)
    dilation = (3, 4)
    conv = Conv2dNormalised(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        input_size=input_size,
        padding=padding,
        stride=stride,
        dilation=dilation,
    )
    with torch.no_grad():
        Ax = conv.tangent(x)
        y = torch.randn_like(Ax)
        ATy = conv.transpose(y)
    assert torch.isclose(torch.sum(y * Ax), torch.sum(ATy * x))


def test_conv3d_adjoint():
    input_size = (14, 18, 13)
    in_channels = 2
    out_channels = 4
    x = torch.randn(100, in_channels, *input_size)
    padding = (2, 6, 5)
    kernel_size = (2, 4, 3)
    stride = (2, 5, 3)
    dilation = (3, 2, 4)
    conv = Conv3dNormalised(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        input_size=input_size,
        padding=padding,
        stride=stride,
        dilation=dilation,
    )
    with torch.no_grad():
        Ax = conv.tangent(x)
        y = torch.randn_like(Ax)
        ATy = conv.transpose(y)
    assert torch.isclose(torch.sum(y * Ax), torch.sum(ATy * x))
