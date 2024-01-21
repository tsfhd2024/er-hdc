import copy
import math

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


class Encoder(object):
    """
    The nonlinear encoder class maps data nonlinearly to high dimensional space.
    To do this task, it uses two randomly generated tensors:

    :math:`B`. The `(dim, features)` sized random basis hypervectors, drawn
    from a standard normal distribution
    :math:`b`. An additional `(dim,)` sized base, drawn from a uniform
    distribution between :math:`[0, 2 \ pi]`.

    The hypervector :math:`H \in \mathbb{R}^D` of :math:`X \in \mathbb{R}^f`
    is:

    .. math:: H_i = \cos(X \cdot B_i + b_i) \sin(X \cdot B_i)

    Args:
        features (int, > 0): Dimensionality of original data.

        dim (int, > 0): Target dimension for output data.
    """

    def genLevelHVs(self, totalLevel, D):
        baseVal = -1
        levelHVs = dict()
        indexVector = range(D)
        nextLevel = int((D / 2 / totalLevel))
        change = int(D / 2)
        for level in range(totalLevel):
            name = level
            if level == 0:
                base = np.full(D, baseVal)
                toOne = np.random.permutation(indexVector)[:change]
            else:
                toOne = np.random.permutation(indexVector)[:nextLevel]
            for index in toOne:
                base[index] = base[index] * -1
            levelHVs[name] = copy.deepcopy(torch.tensor(base))
        return levelHVs

    def __init__(
        self,
        features: int,
        n_levels: int = 100,
        dim: int = 4000,
        encoding_system: str = "kernel",
    ):
        self.dim = dim
        self.features = features
        self.basis = (torch.randn(self.dim, self.features).int()).float()
        self.base = torch.empty(self.dim).uniform_(0.0, 2 * math.pi)
        self.n_levels = n_levels
        self.levels = self.genLevelHVs(n_levels, self.dim)
        self.IDs = 2 * torch.randint(0, 2, (self.features, self.dim)) - 1
        if encoding_system in {"kernel", "record", "n-gram"}:
            self.encoding_system = encoding_system
        else:
            raise Exception(
                """Encoding system should belong to ["kernel", "record", "n-gram"]"""
            )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.encoding_system == "kernel":
            n = x.size(0)
            bsize = math.ceil(0.01 * n)
            h = torch.empty(n, self.dim, device=x.device, dtype=x.dtype)
            temp = torch.empty(bsize, self.dim, device=x.device, dtype=x.dtype)

            # we need batches to remove memory usage
            for i in range(0, n, bsize):
                torch.matmul(x[i : i + bsize], self.basis.T, out=temp)
                torch.add(temp, self.base, out=h[i : i + bsize])
                h[i : i + bsize].cos_().mul_(temp.sin_())
            return h
        elif self.encoding_system == "n-gram":
            p_min = x.min()
            p_max = x.max()
            H = torch.zeros(x.size(0), self.dim)

            for i, in_x in enumerate(x):
                p = in_x
                l = (
                    (self.n_levels - 1) * (p - p_min) / (p_max - p_min)
                ).int()  # This is now a vector of l values

                h = sum(
                    torch.roll(self.levels[l_j.item()], j) for j, l_j in enumerate(l)
                )

                H[i] = h / h.max()

            return H
        else:
            p_min = x.min()
            p_max = x.max()
            H = torch.zeros(x.size(0), self.dim)
            for i, in_x in enumerate(x):
                p = in_x
                l = (
                    (self.n_levels - 1) * (p - p_min) / (p_max - p_min)
                ).int()  # This is now a vector of l values

                h = sum(
                    torch.multiply(self.IDs[j], self.levels[l_j.item()])
                    for j, l_j in enumerate(l)
                )

                H[i] = h / h.max()

            return H

    def to(self, *args):
        """Moves data to the device specified, e.g. cuda, cpu or changes
        dtype of the data representation, e.g. half or double.
        Because the internal data is saved as torch.tensor, the parameter
        can be anything that torch accepts. The change is done in-place.

        Args:
            device (str or :class:`torch.torch.device`) Device to move data.

        Returns:
            :class:`Encoder`: self
        """

        self.basis = self.basis.to(*args)
        self.base = self.base.to(*args)
        return self
