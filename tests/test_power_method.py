import pytest
import torch

from non_expansive_odes.models.linear import (
    Conv1dNormalised,
    Conv2dNormalised,
    Conv3dNormalised,
    LinearNormalised,
)
from non_expansive_odes.utils.power_method import power_method_nonsquare


@pytest.fixture
def op_u(op_type):
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
    return lin, lin._generate_random_input()


@pytest.mark.parametrize("op_type", ["linear", "conv1d", "conv2d", "conv3d"])
def test_sv_power_method(op_u):
    lin, u_init = op_u
    norm, u = power_method_nonsquare(lin.tangent, lin.transpose, u_init=u_init, k=10000)
    assert torch.isclose(
        u.norm(), torch.tensor(1.0)
    ), "Singular vector output by power method is not a unit vector"
    with torch.no_grad():
        assert torch.isclose(
            torch.tensor(norm), lin.tangent(u).norm()
        ), "Singular value and vector output by power method do not match"
