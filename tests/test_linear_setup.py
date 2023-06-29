from random import random

import pytest
import torch

from non_expansive_odes.models.linear import (
    Conv1dNormalised,
    Conv2dNormalised,
    Conv3dNormalised,
    LinearNormalised,
)


@pytest.fixture
def linear(op_type):
    if op_type == "linear":
        lin = LinearNormalised(17, 39)
    elif op_type == "conv1d":
        lin = Conv1dNormalised(7, 12, (5,), input_size=(32,))
    elif op_type == "conv2d":
        lin = Conv2dNormalised(3, 5, (7, 5), input_size=(16, 16))
    elif op_type == "conv3d":
        lin = Conv3dNormalised(5, 3, (3, 5, 3), input_size=(8, 8, 8))
    else:
        raise ValueError(f"Test fixture not implemented for {op_type}")
    return lin


@pytest.mark.parametrize("op_type", ["linear", "conv1d", "conv2d", "conv3d"])
def test_norm_setup(linear):
    assert torch.isclose(
        linear.u.norm(), torch.tensor(1.0)
    ), "Singular vector is not a unit vector"
    with torch.no_grad():
        assert torch.isclose(
            linear.tangent(linear.u).norm(), torch.tensor(linear.norm)
        ), "Operator norm does not match output of power method"


@pytest.mark.parametrize("op_type", ["linear", "conv1d", "conv2d", "conv3d"])
def test_rescale(linear):
    rescaling = random()
    linear.rescale(rescaling)
    assert torch.isclose(
        linear.u.norm(), torch.tensor(1.0)
    ), "Singular vector is not a unit vector"
    with torch.no_grad():
        assert torch.isclose(
            linear.tangent(linear.u).norm(), torch.tensor(linear.norm)
        ), "Operator norm does not match output of power method"


@pytest.mark.parametrize("op_type", ["linear", "conv1d", "conv2d", "conv3d"])
def test_swap_weights(linear):
    linear.weight = torch.nn.Parameter(torch.randn_like(linear.weight))
    linear.compute_spectral_norm(k=10000)
    assert torch.isclose(
        linear.u.norm(), torch.tensor(1.0)
    ), "Singular vector is not a unit vector"
    with torch.no_grad():
        assert torch.isclose(
            linear.tangent(linear.u).norm(), torch.tensor(linear.norm)
        ), "Operator norm does not match output of power method"
