import pytest
import torch

from non_expansive_odes.models.activation_functions import (
    LeakyReLU,
    ReLU,
    Sigmoid,
    Tanh,
)
from non_expansive_odes.models.blocks import NonexpansiveBlock
from non_expansive_odes.models.integrators import euler, heun, rk4
from non_expansive_odes.models.linear import (
    Conv1dNormalised,
    Conv2dNormalised,
    Conv3dNormalised,
    LinearNormalised,
)


@pytest.fixture
def integrator(int_method):
    if int_method == "euler":
        return euler
    elif int_method == "heun":
        return heun
    elif int_method == "rk4":
        return rk4
    else:
        raise ValueError(f"Test fixture not implemented for {int_method}")


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


@pytest.fixture
def activation(act_func):
    if act_func == "ReLU":
        return ReLU()
    elif act_func == "LeakyReLU":
        return LeakyReLU()
    elif act_func == "Sigmoid":
        return Sigmoid()
    elif act_func == "Tanh":
        return Tanh()
    else:
        raise ValueError(f"Test fixture not implemented for {act_func}")


@pytest.mark.parametrize("int_method", ["euler", "heun", "rk4"])
@pytest.mark.parametrize("op_type", ["linear", "conv1d", "conv2d", "conv3d"])
@pytest.mark.parametrize("act_func", ["ReLU", "LeakyReLU", "Sigmoid", "Tanh"])
def test_n_steps(integrator, linear, activation):
    block = NonexpansiveBlock(linear, activation, integrator)
    assert (
        block.timespan / block.n_steps
    ) * block.norm**2 * block.L <= 2.0 * block.r_cc, (
        "Circle contractivity condition violated"
    )
    block.linear.weight = torch.nn.Parameter(torch.randn_like(block.linear.weight))
    block.linear.compute_spectral_norm(k=1000)
    assert (
        block.timespan / block.n_steps
    ) * block.norm**2 * block.L <= 2.0 * block.r_cc, (
        "Circle contractivity condition violated after changing weights"
    )
