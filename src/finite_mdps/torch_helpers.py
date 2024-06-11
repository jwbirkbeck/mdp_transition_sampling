import torch


class TorchBox:
    def __init__(self, low, high):
        assert low.shape == high.shape
        self.low = low
        self.high = high
        self.shape = low.shape

    def sample(self):
        raise NotImplementedError

    def contains(self):
        raise NotImplementedError

    def to_jsonable(self):
        raise NotImplementedError

    def from_jsonable(self):
        raise NotImplementedError

    def __repr__(self):
        return f"TorchBox({self.low}, {self.high}, {self.shape})"

    # def __eq__(self, other):
    #     raise NotImplementedError

    # def __setstate__(self):
    #     raise NotImplementedError
