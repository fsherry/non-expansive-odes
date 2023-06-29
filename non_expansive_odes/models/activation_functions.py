from torch.nn import ELU, LeakyReLU, ReLU, Sigmoid, Tanh


class ReLU(ReLU):
    def __init__(self, inplace=False):
        super().__init__(inplace)

    @property
    def L(self):
        return 1.0


class LeakyReLU(LeakyReLU):
    def __init__(self, negative_slope=0.01, inplace=False):
        super().__init__(negative_slope, inplace)

    @property
    def L(self):
        return max(abs(self.negative_slope), 1.0)


class ELU(ELU):
    def __init__(self, alpha=1.0, inplace=False):
        super().__init__(alpha, inplace)

    @property
    def L(self):
        return max(abs(self.alpha), 1.0)


class Sigmoid(Sigmoid):
    def __init__(self):
        super().__init__()

    @property
    def L(self):
        return 0.25


class Tanh(Tanh):
    def __init__(self):
        super().__init__()

    @property
    def L(self):
        return 1.0
