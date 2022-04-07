# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)
from __future__ import annotations
try:


    from abc import ABC, abstractmethod

    import torch
    from torch import Tensor
    from typing import List, Optional, Tuple
    from torch.nn import Module
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    import matplotlib.pyplot as plt
except:
    pass
import numpy as np

from ..util.general import reshape


class function2d:
    '''
    This is a benchmark of bi-dimensional functions interesting to optimize. 

    '''

    def plot(self):
        bounds = self.bounds
        x1 = np.linspace(bounds[0][0], bounds[0][1], 100)
        x2 = np.linspace(bounds[1][0], bounds[1][1], 100)
        X1, X2 = np.meshgrid(x1, x2)
        X = np.hstack((X1.reshape(100 * 100, 1), X2.reshape(100 * 100, 1)))
        Y = self.f(X)

        plt.figure()
        plt.contourf(X1, X2, Y.reshape((100, 100)), 100)
        if (len(self.min) > 1):
            plt.plot(np.array(self.min)[:, 0], np.array(self.min)[:, 1], 'w.', markersize=20, label=u'Observations')
        else:
            plt.plot(self.min[0][0], self.min[0][1], 'w.', markersize=20, label=u'Observations')
        plt.colorbar()
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title(self.name)
        plt.show()





class BaseTestProblem(Module, ABC):
    r"""Base class for test functions."""

    dim: int
    _bounds: List[Tuple[float, float]]
    _check_grad_at_opt: bool = True

    def __init__(self, noise_std: Optional[float] = None, negate: bool = False) -> None:
        r"""Base constructor for test functions.

        Args:
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
        """
        super().__init__()
        self.noise_std = noise_std
        self.negate = negate
        self.register_buffer(
            "bounds", torch.tensor(self._bounds, dtype=torch.float).transpose(-1, -2)
        )

    def forward(self, X: Tensor, noise: bool = True) -> Tensor:
        r"""Evaluate the function on a set of points.

        Args:
            X: A `batch_shape x d`-dim tensor of point(s) at which to evaluate the
                function.
            noise: If `True`, add observation noise as specified by `noise_std`.

        Returns:
            A `batch_shape`-dim tensor ouf function evaluations.
        """
        batch = X.ndimension() > 1
        X = X if batch else X.unsqueeze(0)
        f = self.evaluate_true(X=X)
        if noise and self.noise_std is not None:
            f += self.noise_std * torch.randn_like(f)
        if self.negate:
            f = -f
        return f if batch else f.squeeze(0)

    @abstractmethod
    def evaluate_true(self, X: Tensor) -> Tensor:
        r"""Evaluate the function (w/o observation noise) on a set of points."""
        pass  # pragma: no cover


class MultiObjectiveTestProblem(BaseTestProblem):
    r"""Base class for test multi-objective test functions.

    TODO: add a pareto distance function that returns the distance
    between a provided point and the closest point on the true pareto front.
    """

    num_objectives: int
    _ref_point: List[float]
    _max_hv: float

    def __init__(self, noise_std: Optional[float] = None, negate: bool = False) -> None:
        r"""Base constructor for multi-objective test functions.

        Args:
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the objectives.
        """
        super().__init__(noise_std=noise_std, negate=negate)
        ref_point = torch.tensor(self._ref_point, dtype=torch.float)
        if negate:
            ref_point *= -1
        self.register_buffer("ref_point", ref_point)

    @property
    def max_hv(self) -> float:
        try:
            return self._max_hv
        except AttributeError:
            raise NotImplementedError(
                f"Problem {self.__class__.__name__} does not specify maximal "
                "hypervolume."
            )

    def gen_pareto_front(self, n: int) -> Tensor:
        r"""Generate `n` pareto optimal points."""
        raise NotImplementedError


class RocketInjector(MultiObjectiveTestProblem):
    r"""Optimize Vehicle crash-worthiness.

    See [Tanabe2020]_ for details.

    The reference point is 1.1 * the nadir point from
    approximate front provided by [Tanabe2020]_.

    The maximum hypervolume is computed using the approximate
    pareto front from [Tanabe2020]_.
    """

    _ref_point = [1864.72022, 11.81993945, 0.2903999384]
    _max_hv = 246.81607081187002
    _bounds = [(0.0, 1.0)] * 4
    dim = 4
    num_objectives = 3

    def evaluate_true(self, X: Tensor) -> Tensor:
        X1, X2, X3, X4 = torch.split(X, 1, -1)
        f1 = (0.692
              + (0.477 * X1) \
              - (0.687 * X2) \
              - (0.080 * X3) \
              - (0.0650 * X4) \
              - (0.167 * X1 * X1)
              - (0.0129 * X2 * X1) \
              + (0.0796 * X2 * X2) \
              - (0.0634 * X3 * X1) \
              - (0.0257 * X3 * X2)
              + (0.0877 * X3 * X3) \
              - (0.0521 * X4 * X1) \
              + (0.00156 * X4 * X2) \
              + (0.00198 * X4 * X3) + (0.0184 * X4 * X4))

        f2 = (
                0.153 - (0.322 * X1) + (0.396 * X2) + (0.424 * X3) + (0.0226 * X4) + (0.175 * X1 * X1) + (
                0.0185 * X2 * X1) - (0.0701 * X2 * X2) - (0.251 * X3 * X1) + (0.179 * X3 * X2) + (0.0150 * X3 * X3) + (
                        0.0134 * X4 * X1) + (0.0296 * X4 * X2) + (0.0752 * X4 * X3) + (0.0192 * X4 * X4)
        )
        f3 = (0.370 - (0.205 * X1) + (0.0307 * X2) + (0.108 * X3) + (1.019 * X4) - (0.135 * X1 * X1) + (
                    0.0141 * X2 * X1) + (0.0998 * X2 * X2) + (0.208 * X3 * X1) - (0.0301 * X3 * X2) - (
                          0.226 * X3 * X3) + (0.353 * X4 * X1) - (0.0497 * X4 * X3) - (0.423*X4*X4) + (0.202*X2*X1*X1)
             - (0.281*X3*X1*X1) - (0.342*X2*X2*X1) - (0.245*X2*X2*X3) + (0.281*X3*X3*X2) - (0.184*X4*X4*X1) - (0.281*X2*X1*X3))

        f_X = torch.cat([f1, f2, f3], dim=-1)
        return f_X


class rosenbrock(function2d):
    '''
    Cosines function
    
    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''

    def __init__(self, bounds=None, sd=None):
        self.input_dim = 2
        if bounds is None:
            self.bounds = [(-0.5, 3), (-1.5, 2)]
        else:
            self.bounds = bounds
        self.min = [(0, 0)]
        self.fmin = 0
        if sd == None:
            self.sd = 0
        else:
            self.sd = sd
        self.name = 'Rosenbrock'

    def f(self, X):
        X = reshape(X, self.input_dim)
        n = X.shape[0]
        if X.shape[1] != self.input_dim:
            return 'Wrong input dimension'
        else:
            fval = 100 * (X[:, 1] - X[:, 0] ** 2) ** 2 + (X[:, 0] - 1) ** 2
            if self.sd == 0:
                noise = np.zeros(n).reshape(n, 1)
            else:
                noise = np.random.normal(0, self.sd, n).reshape(n, 1)
            return -fval.reshape(n, 1) + noise


class f1_Binh(function2d):
    '''
    Goldstein function

    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''

    def __init__(self, bounds=None, sd=None):
        self.input_dim = 2
        if bounds is None:
            self.bounds = [(0, 5), (0, 3)]
        else:
            self.bounds = bounds

        self.ref_point = [1.0, 1.0]
        self.min = [(5, 3)]
        self.fmin = 4
        self.ref_point
        if sd == None:
            self.sd = 0
        else:
            self.sd = sd
        self.name = 'f2_Binh'

    def f1(self, X, true_val=False):
        X = reshape(X, self.input_dim)
        n = X.shape[0]
        if X.shape[1] != self.input_dim:
            return 'Wrong input dimension'
        else:
            x1 = X[:, 0]
            x2 = X[:, 1]
            fval = 4 * x1 ** 2.0 + 4 * x2 ** 2.0
            if self.sd == 0:
                noise = np.zeros(n).reshape(n, 1)
            else:
                noise = np.random.normal(0, self.sd, n).reshape(n, 1)
            return -fval.reshape(n, 1) + noise

    def f2(self, X, true_val=False):
        X = reshape(X, self.input_dim)
        n = X.shape[0]
        if X.shape[1] != self.input_dim:
            return 'Wrong input dimension'
        else:
            x1 = X[:, 0]
            x2 = X[:, 1]
            fval = (x1 - 5) ** 2.0 + (x2 - 5) ** 2.0
            if self.sd == 0:
                noise = np.zeros(n).reshape(n, 1)
            else:
                noise = np.random.normal(0, self.sd, n).reshape(n, 1)
            return -fval.reshape(n, 1) + noise


class POL(function2d):
    def __init__(self, bounds=None, sd=None):
        self.input_dim = 2
        if bounds is None:
            self.bounds = [(-np.pi, np.pi), (-np.pi, np.pi)]
        else:
            self.bounds = bounds

        self.ref_point = [0.0, 0.0]
        self.min = [(5, 3)]
        self.fmin = 4
        if sd == None:
            self.sd = 0
        else:
            self.sd = sd
        self.name = 'POL'

    def f1(self, X, true_val=False):
        X = reshape(X, self.input_dim)
        n = X.shape[0]

        if X.shape[1] != self.input_dim:
            return 'Wrong input dimension'
        else:
            x1 = X[:, 0]
            x2 = X[:, 1]

            a = 0.5 * np.sin(1.0) - 2.0 * np.cos(1.0) + 1.0 * np.sin(2.0) - 1.5 * np.cos(2.0)
            b = 0.5 * np.sin(x1) - 2.0 * np.cos(x1) + 1.0 * np.sin(x2) - 1.5 * np.cos(x2)
            c = 1.5 * np.sin(1.0) - 1.0 * np.cos(1.0) + 2.0 * np.sin(2.0) - 0.5 * np.cos(2.0)
            d = 1.5 * np.sin(x1) - 1.0 * np.cos(x1) + 2.0 * np.sin(x2) - 0.5 * np.cos(x2)

            fval = 1 + (a - b) ** 2.0 + (c - d) ** 2.0
            return -(fval.reshape(n, 1))

    def f2(self, X, true_val=False):
        X = reshape(X, self.input_dim)
        n = X.shape[0]
        if X.shape[1] != self.input_dim:
            return 'Wrong input dimension'
        else:
            x1 = X[:, 0]
            x2 = X[:, 1]
            fval = (x1 + 3) ** 2.0 + (x2 + 1) ** 2.0
            return -(fval.reshape(n, 1))


class NO_HOLE(function2d):
    def __init__(self, bounds=None, sd=None):
        self.input_dim = 2
        if bounds is None:
            self.bounds = [(-1, 1), (-1, 1)]
        else:
            self.bounds = bounds

        self.ref_point = [80.0, 80.0]
        self.min = [(5, 3)]
        self.fmin = 4
        self.ref_point
        self.q = 0.2
        self.p = 2.0
        self.d0 = 0.02
        self.h = 2
        self.delta = 1 - (np.sqrt(2.0) / 2.0)
        self.alpha = np.pi / 4.0
        if sd == None:
            self.sd = 0
        else:
            self.sd = sd
        self.name = 'POL'

    def f1(self, X, true_val=False):
        X = reshape(X, self.input_dim)
        n = X.shape[0]

        if X.shape[1] != self.input_dim:
            return 'Wrong input dimension'
        else:
            x1 = X[:, 0]
            x2 = X[:, 1]

            x1p = x1 + self.delta
            x2p = x2 - self.delta

            x1pp = x1p * np.cos(self.alpha) + x2p * np.sin(self.alpha)
            x2pp = -x1p * np.sin(self.alpha) + x2p * np.cos(self.alpha)

            x1ppp = x1pp * np.pi
            x2ppp = x2pp * np.pi

            u = np.sin(x1ppp / 2.0)
            v = (np.sin(x2ppp / 2.0)) ** 2.0

            if u < 0:
                up = -(-u) ** (self.h)
            else:
                up = u ** (self.h)

            vp = v ** (1.0 / self.h)

            t = up
            a = vp * 2.0 * self.p

            if a < self.p:
                b = (self.p - a) * np.exp(self.q)
            else:
                b = 0

            d = (self.q / (2.0 * a)) + self.d0
            c = self.q / (d ** 2.0)

            fval = (t + 1) ** 2.0 + a  # + b* np.exp( -c * ( t - d) ** 2.0)
            # print("fval", fval)
            return -fval.reshape(n, 1)

    def f2(self, X, true_val=False):
        X = reshape(X, self.input_dim)
        n = X.shape[0]
        if X.shape[1] != self.input_dim:
            return 'Wrong input dimension'
        else:
            x1 = X[:, 0]
            x2 = X[:, 1]
            x1p = x1 + self.delta
            x2p = x2 - self.delta

            x1pp = x1p * np.cos(self.alpha) + x2p * np.sin(self.alpha)
            x2pp = -x1p * np.sin(self.alpha) + x2p * np.cos(self.alpha)

            x1ppp = x1pp * np.pi
            x2ppp = x2pp * np.pi

            u = np.sin(x1ppp / 2.0)
            v = (np.sin(x2ppp / 2.0)) ** 2.0

            if u < 0:
                up = -(-u) ** (self.h)
            else:
                up = u ** (self.h)
            vp = v ** (1.0 / self.h)

            t = up
            a = vp * 2.0 * self.p

            if a < self.p:
                b = (self.p - a) * np.exp(self.q)
            else:
                b = 0

            d = (self.q / (2.0 * a)) + self.d0
            c = self.q / (d ** 2.0)

            fval = (t - 1) ** 2.0 + a  # + b * np.exp(-c * (t + d) ** 2.0)
            # print("fval", fval)
            return -fval.reshape(n, 1)


class HOLE(function2d):
    def __init__(self, bounds=None, sd=None):
        self.input_dim = 2
        if bounds is None:
            self.bounds = [(-1, 1), (-1, 1)]
        else:
            self.bounds = bounds

        self.ref_point = [80.0, 80.0]
        self.min = [(5, 3)]
        self.fmin = 4
        self.ref_point
        self.q = 0.2
        self.p = 2.0
        self.d0 = 0.02
        self.h = 2.0
        self.delta = 1 - (np.sqrt(2.0) / 2.0)
        self.alpha = np.pi / 4.0
        if sd == None:
            self.sd = 0
        else:
            self.sd = sd
        self.name = 'POL'

    def f1(self, X, true_val=False):
        X = reshape(X, self.input_dim)
        n = X.shape[0]

        if X.shape[1] != self.input_dim:
            return 'Wrong input dimension'
        else:
            x1 = X[:, 0]
            x2 = X[:, 1]

            x1p = x1 + self.delta
            x2p = x2 - self.delta

            x1pp = x1p * np.cos(self.alpha) + x2p * np.sin(self.alpha)
            x2pp = -x1p * np.sin(self.alpha) + x2p * np.cos(self.alpha)

            x1ppp = x1pp * np.pi
            x2ppp = x2pp * np.pi

            u = np.sin(x1ppp / 2.0)
            v = np.sin(x2ppp / 2.0) ** 2.0

            up = np.zeros(len(u))
            up[u < 0] = -(-u) ** (self.h)
            up[u >= 0] = u ** (self.h)

            vp = v ** (1.0 / self.h)

            t = up
            a = vp * 2.0 * self.p

            b = np.zeros(len(a))
            b[a <= self.p] = (self.p - a) * np.exp(self.q)

            d = (self.q / 2.0) * a + self.d0
            c = self.q / (d ** 2.0)

            fval = ((t + 1) ** 2.0) + a + b * np.exp(-c * (t - d) ** 2.0)
            # print("fval", fval)
            return -fval.reshape(n, 1)

    def f2(self, X, true_val=False):
        X = reshape(X, self.input_dim)
        n = X.shape[0]
        if X.shape[1] != self.input_dim:
            return 'Wrong input dimension'
        else:
            x1 = X[:, 0]
            x2 = X[:, 1]
            x1p = x1 + self.delta
            x2p = x2 - self.delta

            x1pp = x1p * np.cos(self.alpha) + x2p * np.sin(self.alpha)
            x2pp = -x1p * np.sin(self.alpha) + x2p * np.cos(self.alpha)

            x1ppp = x1pp * np.pi
            x2ppp = x2pp * np.pi

            u = np.sin(x1ppp / 2.0)
            v = (np.sin(x2ppp / 2.0)) ** 2.0

            up = np.zeros(len(u))
            up[u < 0] = -(-u) ** (self.h)
            up[u >= 0] = u ** (self.h)

            vp = v ** (1.0 / self.h)

            t = up
            a = vp * 2.0 * self.p

            b = np.zeros(len(a))
            b[a <= self.p] = (self.p - a) * np.exp(self.q)

            # print("b", b)
            d = (self.q / 2.0) * a + self.d0
            c = self.q / (d ** 2.0)

            fval = ((t - 1) ** 2.0) + a + b * np.exp(-c * (t + d) ** 2.0)
            # print("fval", fval)
            return -fval.reshape(n, 1)


class DEB(function2d):
    def __init__(self, bounds=None, sd=None):
        self.input_dim = 2
        if bounds is None:
            self.bounds = [(0.1, 1), (0, 5)]
        else:
            self.bounds = bounds

        self.ref_point = [0, 0]
        self.min = [(5, 3)]
        self.fmin = 4
        if sd == None:
            self.sd = 0
        else:
            self.sd = sd
        self.name = 'DEB'

    def f1(self, X, true_val=False):
        X = reshape(X, self.input_dim)
        n = X.shape[0]

        if X.shape[1] != self.input_dim:
            return 'Wrong input dimension'
        else:
            x1 = X[:, 0]
            x2 = X[:, 1]

            fval = x1
            return -(fval.reshape(n, 1) - 1)  # /0.6

    def f2(self, X, true_val=False):
        X = reshape(X, self.input_dim)
        n = X.shape[0]
        if X.shape[1] != self.input_dim:
            return 'Wrong input dimension'
        else:
            x1 = X[:, 0]
            x2 = X[:, 1]
            fval = (1 + x2) / x1
            return -(fval.reshape(n, 1) - 9)  # /8

    def c1(self, x, true_val=False):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        print("x", x)
        n = x.shape[0]
        x1 = x[:, 0]
        x2 = x[:, 1]

        fval = x2 + 9 * x1 - 6
        fval = -fval  # make the restriction less and equal
        return fval.reshape(n, 1)

    def c2(self, x, true_val=False):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        n = x.shape[0]
        x1 = x[:, 0]
        x2 = x[:, 1]

        fval = -x2 + 9 * x1 - 1
        fval = -fval  # make the restriction less and equal
        return fval.reshape(n, 1)

    def c(self, x, true_val=False):
        return self.c1(x), self.c2(x)


class SRN(function2d):
    def __init__(self, bounds=None, sd=None):
        self.input_dim = 2
        if bounds is None:
            self.bounds = [(-20, 20), (-20, 20)]
        else:
            self.bounds = bounds

        self.ref_point = [0, 0]
        self.min = [(5, 3)]
        self.fmin = 4
        if sd == None:
            self.sd = 0
        else:
            self.sd = sd
        self.name = 'DEB'

    def f1(self, X, true_val=False):
        X = reshape(X, self.input_dim)
        n = X.shape[0]

        if X.shape[1] != self.input_dim:
            return 'Wrong input dimension'
        else:
            x1 = X[:, 0]
            x2 = X[:, 1]

            fval = (x1 - 2) ** 2.0 + (x2 - 1) ** 2.0 + 2
            return -(fval.reshape(n, 1) - 300)

    def f2(self, X, true_val=False):
        X = reshape(X, self.input_dim)
        n = X.shape[0]
        if X.shape[1] != self.input_dim:
            return 'Wrong input dimension'
        else:
            x1 = X[:, 0]
            x2 = X[:, 1]
            fval = 9 * x1 - (x2 - 1) ** 2
            return -(fval.reshape(n, 1) - 100)

    def c1(self, x, true_val=False):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        n = x.shape[0]
        x1 = x[:, 0]
        x2 = x[:, 1]

        fval = x1 ** 2.0 + x2 ** 2.0 - 225
        fval = fval  # make the restriction less and equal
        return fval.reshape(n, 1)

    def c2(self, x, true_val=False):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        n = x.shape[0]
        x1 = x[:, 0]
        x2 = x[:, 1]

        fval = x1 - 3 * x2 + 10.0
        fval = fval  # make the restriction less and equal
        return fval.reshape(n, 1)

    def c(self, x, true_val=False):
        return self.c1(x), self.c2(x)


class BNH(function2d):
    def __init__(self, bounds=None, sd=None):
        self.input_dim = 2
        if bounds is None:
            self.bounds = [(0, 5), (0, 3)]
        else:
            self.bounds = bounds

        self.ref_point = [0, 0]
        self.min = [(5, 3)]
        self.fmin = 4
        if sd == None:
            self.sd = 0
        else:
            self.sd = sd
        self.name = 'BNH'

    def f1(self, X, true_val=False):
        X = reshape(X, self.input_dim)
        n = X.shape[0]

        if X.shape[1] != self.input_dim:
            return 'Wrong input dimension'
        else:
            x1 = X[:, 0]
            x2 = X[:, 1]

            fval = 4 * (x1) ** 2.0 + 4 * (x2) ** 2.0
            return -(fval.reshape(n, 1) - 120)

    def f2(self, X, true_val=False):
        X = reshape(X, self.input_dim)
        n = X.shape[0]
        if X.shape[1] != self.input_dim:
            return 'Wrong input dimension'
        else:
            x1 = X[:, 0]
            x2 = X[:, 1]
            fval = (x1 - 5) ** 2.0 - (x2 - 5) ** 2
            return -(fval.reshape(n, 1) - 48)

    def c1(self, x, true_val=False):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        n = x.shape[0]
        x1 = x[:, 0]
        x2 = x[:, 1]

        fval = (x1 - 5.0) ** 2.0 + x2 ** 2.0 - 25
        fval = fval  # make the restriction less and equal
        return fval.reshape(n, 1)

    def c2(self, x, true_val=False):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        n = x.shape[0]
        x1 = x[:, 0]
        x2 = x[:, 1]

        fval = (x1 - 8.0) ** 2.0 + (x2 + 3) ** 2.0 - 7.7
        fval = -fval  # make the restriction less and equal
        return fval.reshape(n, 1)

    def c(self, x, true_val=False):
        return self.c1(x), self.c2(x)


class TNK_BNH(function2d):
    def __init__(self, bounds=None, sd=None):
        self.input_dim = 2
        if bounds is None:
            self.bounds = [(0, 5), (0, 3)]
        else:
            self.bounds = bounds

        self.ref_point = [0.0, 0.0]
        self.min = [(5, 3)]
        self.fmin = 4
        if sd == None:
            self.sd = 0
        else:
            self.sd = sd
        self.name = 'TNK'

    def f1(self, X, true_val=False):
        X = reshape(X, self.input_dim)
        n = X.shape[0]

        if X.shape[1] != self.input_dim:
            return 'Wrong input dimension'
        else:
            x1 = X[:, 0]
            x2 = X[:, 1]

            fval = x1
            return -(fval.reshape(n, 1) - 5.0)

    def f2(self, X, true_val=False):
        X = reshape(X, self.input_dim)
        n = X.shape[0]
        if X.shape[1] != self.input_dim:
            return 'Wrong input dimension'
        else:
            x1 = X[:, 0]
            x2 = X[:, 1]
            fval = x2
            return -(fval.reshape(n, 1) - 3.0)

    def c1(self, x, true_val=False):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        n = x.shape[0]
        x1 = x[:, 0]
        x2 = x[:, 1]

        fval = (x1 - 3.5) ** 2.0 + (x2 - 2) ** 2.0 - 2
        fval = fval  # make the restriction less and equal
        return fval.reshape(n, 1)

    def c(self, x, true_val=False):
        return self.c1(x)


class TNK(function2d):
    def __init__(self, bounds=None, sd=None):
        self.input_dim = 2
        if bounds is None:
            self.bounds = [(0, np.pi), (0, np.pi)]
        else:
            self.bounds = bounds

        self.ref_point = [0.0, 0.0]
        self.min = [(5, 3)]
        self.fmin = 4
        if sd == None:
            self.sd = 0
        else:
            self.sd = sd
        self.name = 'TNK'

    def f1(self, X, true_val=False):
        X = reshape(X, self.input_dim)
        n = X.shape[0]

        if X.shape[1] != self.input_dim:
            return 'Wrong input dimension'
        else:
            x1 = X[:, 0]
            x2 = X[:, 1]

            fval = x1
            return -(fval.reshape(n, 1) - 1.2)

    def f2(self, X, true_val=False):
        X = reshape(X, self.input_dim)
        n = X.shape[0]
        if X.shape[1] != self.input_dim:
            return 'Wrong input dimension'
        else:
            x1 = X[:, 0]
            x2 = X[:, 1]
            fval = x2
            return -(fval.reshape(n, 1) - 1.2)

    def c1(self, x, true_val=False):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        n = x.shape[0]
        x1 = x[:, 0]
        x2 = x[:, 1]

        fval = x1 ** 2.0 + x2 ** 2.0 - 1 - 0.1 * np.cos(16.0 * np.arctan(x1 / x2))
        fval = -fval  # make the restriction less and equal
        return fval.reshape(n, 1)

    def c2(self, x, true_val=False):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        n = x.shape[0]
        x1 = x[:, 0]
        x2 = x[:, 1]

        fval = (x1 - 0.5) ** 2.0 + (x2 - 0.5) ** 2.0 - 0.5
        return fval.reshape(n, 1)

    def c(self, x, true_val=False):
        return self.c1(x), self.c2(x)


class beale(function2d):
    '''
    Cosines function
    
    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''

    def __init__(self, bounds=None, sd=None):
        self.input_dim = 2
        if bounds is None:
            self.bounds = [(-1, 1), (-1, 1)]
        else:
            self.bounds = bounds
        self.min = [(0, 0)]
        self.fmin = 0
        if sd == None:
            self.sd = 0
        else:
            self.sd = sd
        self.name = 'Beale'

    def f(self, X):
        X = reshape(X, self.input_dim)
        n = X.shape[0]
        if X.shape[1] != self.input_dim:
            return 'Wrong input dimension'
        else:
            fval = 100 * (X[:, 1] - X[:, 0] ** 2) ** 2 + (X[:, 0] - 1) ** 2
            if self.sd == 0:
                noise = np.zeros(n).reshape(n, 1)
            else:
                noise = np.random.normal(0, self.sd, n).reshape(n, 1)
            return -fval.reshape(n, 1) + noise


class dropwave(function2d):
    '''
    Cosines function
    
    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''

    def __init__(self, bounds=None, sd=None):
        self.input_dim = 2
        if bounds is None:
            self.bounds = [(-1, 1), (-1, 1)]
        else:
            self.bounds = bounds
        self.min = [(0, 0)]
        self.fmin = 0
        if sd == None:
            self.sd = 0
        else:
            self.sd = sd
        self.name = 'dropwave'

    def f(self, X):
        X = reshape(X, self.input_dim)
        n = X.shape[0]
        if X.shape[1] != self.input_dim:
            return 'Wrong input dimension'
        else:
            fval = - (1 + np.cos(12 * np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2))) / (0.5 * (X[:, 0] ** 2 + X[:, 1] ** 2) + 2)
            if self.sd == 0:
                noise = np.zeros(n).reshape(n, 1)
            else:
                noise = np.random.normal(0, self.sd, n).reshape(n, 1)
            return -fval.reshape(n, 1) + noise


class cosines(function2d):
    '''
    Cosines function
    
    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''

    def __init__(self, bounds=None, sd=None):
        self.input_dim = 2
        if bounds is None:
            self.bounds = [(0, 1), (0, 1)]
        else:
            self.bounds = bounds
        self.min = [(0.31426205, 0.30249864)]
        self.fmin = -1.59622468
        if sd == None:
            self.sd = 0
        else:
            self.sd = sd
        self.name = 'Cosines'

    def f(self, X):
        X = reshape(X, self.input_dim)
        n = X.shape[0]
        if X.shape[1] != self.input_dim:
            return 'Wrong input dimension'
        else:
            u = 1.6 * X[:, 0] - 0.5
            v = 1.6 * X[:, 1] - 0.5
            fval = 1 - (u ** 2 + v ** 2 - 0.3 * np.cos(3 * np.pi * u) - 0.3 * np.cos(3 * np.pi * v))
            if self.sd == 0:
                noise = np.zeros(n).reshape(n, 1)
            else:
                noise = np.random.normal(0, self.sd, n).reshape(n, 1)
            return -fval.reshape(n, 1) + noise


class branin(function2d):
    '''
    Branin function
    
    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''

    def __init__(self, bounds=None, a=None, b=None, c=None, r=None, s=None, t=None, sd=None):
        self.input_dim = 2
        if bounds is None:
            self.bounds = [(-5, 10), (1, 15)]
        else:
            self.bounds = bounds
        if a == None:
            self.a = 1
        else:
            self.a = a
        if b == None:
            self.b = 5.1 / (4 * np.pi ** 2)
        else:
            self.b = b
        if c == None:
            self.c = 5 / np.pi
        else:
            self.c = c
        if r == None:
            self.r = 6
        else:
            self.r = r
        if s == None:
            self.s = 10
        else:
            self.s = s
        if t == None:
            self.t = 1 / (8 * np.pi)
        else:
            self.t = t
        if sd == None:
            self.sd = 0
        else:
            self.sd = sd
        self.min = [(-np.pi, 12.275), (np.pi, 2.275), (9.42478, 2.475)]
        self.fmin = 0.397887
        self.name = 'Branin'

    def f(self, X):
        X = reshape(X, self.input_dim)
        n = X.shape[0]
        if X.shape[1] != self.input_dim:
            return 'Wrong input dimension'
        else:
            x1 = X[:, 0]
            x2 = X[:, 1]
            fval = self.a * (x2 - self.b * x1 ** 2 + self.c * x1 - self.r) ** 2 + self.s * (1 - self.t) * np.cos(
                x1) + self.s
            if self.sd == 0:
                noise = np.zeros(n).reshape(n, 1)
            else:
                noise = np.random.normal(0, self.sd, n).reshape(n, 1)
            return -fval.reshape(n, 1) - noise


class goldstein(function2d):
    '''
    Goldstein function
    
    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''

    def __init__(self, bounds=None, sd=None):
        self.input_dim = 2
        if bounds is None:
            self.bounds = [(-2, 2), (-2, 2)]
        else:
            self.bounds = bounds
        self.min = [(0, -1)]
        self.fmin = 3
        if sd == None:
            self.sd = 0
        else:
            self.sd = sd
        self.name = 'Goldstein'

    def f(self, X):
        X = reshape(X, self.input_dim)
        n = X.shape[0]
        if X.shape[1] != self.input_dim:
            return 'Wrong input dimension'
        else:
            x1 = X[:, 0]
            x2 = X[:, 1]
            fact1a = (x1 + x2 + 1) ** 2
            fact1b = 19 - 14 * x1 + 3 * x1 ** 2 - 14 * x2 + 6 * x1 * x2 + 3 * x2 ** 2
            fact1 = 1 + fact1a * fact1b
            fact2a = (2 * x1 - 3 * x2) ** 2
            fact2b = 18 - 32 * x1 + 12 * x1 ** 2 + 48 * x2 - 36 * x1 * x2 + 27 * x2 ** 2
            fact2 = 30 + fact2a * fact2b
            fval = fact1 * fact2
            if self.sd == 0:
                noise = np.zeros(n).reshape(n, 1)
            else:
                noise = np.random.normal(0, self.sd, n).reshape(n, 1)
            return fval.reshape(n, 1) + noise


class test_function_2(function2d):
    def __init__(self, bounds=None, sd=None):
        self.input_dim = 2
        if bounds is None:
            self.bounds = [(0, 1), (0, 1)]
        else:
            self.bounds = bounds
        self.min = [(0.2018, 0.833)]
        self.fmin = 0.748
        self.sd = sd
        self.name = 'test_function_2'

    def f(self, x, offset=0, true_val=False):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        n = x.shape[0]
        x1 = x[:, 0]
        x2 = x[:, 1]
        term2 = -(x1 - 1) ** 2.0
        term3 = -(x2 - 0.5) ** 2.0
        fval = term2 + term3
        if self.sd == 0 or true_val:
            noise = np.zeros(n).reshape(n, 1)
        else:
            noise = np.random.normal(0, self.sd, n).reshape(n, 1)
        # print("fval",-fval.reshape(-1, 1) + noise.reshape(-1, 1))
        return -(fval.reshape(n, 1) + offset) + noise.reshape(-1, 1)

    def c1(self, x, true_val=False):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        n = x.shape[0]
        x1 = x[:, 0]
        x2 = x[:, 1]
        term1 = (x1 - 3) ** 2.0
        term2 = (x2 + 2) ** 2.0
        term3 = -12
        fval = (term1 + term2) * np.exp(-x2 ** 7) + term3
        # print("fval",-fval.reshape(-1, 1))
        return fval.reshape(n, 1)

    def c2(self, x, true_val=False):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        n = x.shape[0]
        x1 = x[:, 0]
        x2 = x[:, 1]
        fval = 10 * x1 + x2 - 7
        # print("fval",-fval.reshape(-1, 1))
        return fval.reshape(n, 1)

    def c3(self, x, true_val=False):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        n = x.shape[0]
        x1 = x[:, 0]
        x2 = x[:, 1]
        term1 = (x1 - 0.5) ** 2.0
        term2 = (x2 - 0.5) ** 2.0
        term3 = -0.2
        fval = term1 + term2 + term3
        # print("fval",-fval.reshape(-1, 1))
        return fval.reshape(n, 1)

    def c(self, x, true_val=False):
        return [self.c1(x), self.c2(x), self.c3(x)]

    def func_val(self, x):
        Y = self.f(x, true_val=True)
        C = self.c(x)
        out = Y.reshape(-1) * np.product(np.concatenate(C, axis=1) < 0, axis=1).reshape(-1)
        out = np.array(out).reshape(-1)
        return -out


class mistery(function2d):
    '''
    Six hump camel function

    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''

    def __init__(self, bounds=None, sd=None):
        self.input_dim = 2
        if bounds is None:
            self.bounds = [(0, 5), (0, 5)]
        else:
            self.bounds = bounds
        self.min = [(2.7450, 2.3523)]
        self.fmin = 1.1743
        self.sd = sd
        self.name = 'Mistery'

    def f(self, x, offset=0.0, true_val=False):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        n = x.shape[0]
        x1 = x[:, 0]
        x2 = x[:, 1]
        term1 = 2
        term2 = 0.01 * (x2 - x1 ** 2.0) ** 2.0
        term3 = (1 - x1) ** 2
        term4 = 2 * (2 - x2) ** 2
        term5 = 7 * np.sin(0.5 * x1) * np.sin(0.7 * x1 * x2)
        fval = term1 + term2 + term3 + term4 + term5
        if self.sd == 0 or true_val:
            noise = np.zeros(n).reshape(n, 1)
        else:
            noise = np.random.normal(0, self.sd, n).reshape(n, 1)
        # print("fval",-fval.reshape(-1, 1) + noise.reshape(-1, 1))

        return -(fval.reshape(n, 1) + offset) + noise.reshape(-1, 1)

    def c(self, x, true_val=False):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        n = x.shape[0]
        x1 = x[:, 0]
        x2 = x[:, 1]
        fval = -np.sin(x1 - x2 - np.pi / 8.0)
        # print("fval",-fval.reshape(-1, 1))
        return fval.reshape(n, 1)

    def func_val(self, x):
        Y = self.f(x, true_val=True)
        C = self.c(x)
        out = Y * (C < 0)
        out = np.array(out).reshape(-1)
        return -out


class new_brannin(function2d):
    '''
    Six hump camel function

    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''

    def __init__(self, bounds=None, sd=None):
        self.input_dim = 2
        if bounds is None:
            self.bounds = [(-5, 10), (0, 15)]
        else:
            self.bounds = bounds
        self.min = [(3.26, 0.05)]
        self.fmin = 268.781
        self.sd = sd
        self.name = 'new_brannin'

    def f(self, x, offset=0, true_val=False):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        n = x.shape[0]
        x1 = x[:, 0]
        x2 = x[:, 1]
        term1 = -(x1 - 10) ** 2
        term2 = -(x2 - 15) ** 2.0
        fval = term1 + term2
        if self.sd == 0 or true_val:
            noise = np.zeros(n).reshape(n, 1)
        else:
            noise = np.random.normal(0, self.sd, n).reshape(n, 1)
        # print("fval",-fval.reshape(-1, 1) + noise.reshape(-1, 1))
        return -(fval.reshape(n, 1) + offset) + noise.reshape(-1, 1)

    def c(self, x, true_val=False):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        n = x.shape[0]
        x1 = x[:, 0]
        x2 = x[:, 1]
        term1 = (x2 - (5.1 / (4 * np.pi ** 2.0)) * x1 ** 2.0 + (5.0 / np.pi) * x1 - 6) ** 2.0
        term2 = 10 * (1 - (1.0 / (8 * np.pi))) * np.cos(x1)
        term3 = 5
        fval = term1 + term2 + term3
        # print("fval",-fval.reshape(-1, 1))
        return fval.reshape(n, 1)

    def func_val(self, x):
        Y = self.f(x, true_val=True)
        C = self.c(x)
        out = Y * (C < 0)
        out = np.array(out).reshape(-1)
        return -out


class sixhumpcamel(function2d):
    '''
    Six hump camel function
    
    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''

    def __init__(self, bounds=None, sd=None):
        self.input_dim = 2
        if bounds is None:
            self.bounds = [(-2, 2), (-1, 1)]
        else:
            self.bounds = bounds
        self.min = [(0.0898, -0.7126), (-0.0898, 0.7126)]
        self.fmin = -1.0316
        if sd == None:
            self.sd = 0
        else:
            self.sd = sd
        self.name = 'Six-hump camel'

    def f(self, x):
        x = reshape(x, self.input_dim)
        n = x.shape[0]
        if x.shape[1] != self.input_dim:
            return 'wrong input dimension'
        else:
            x1 = x[:, 0]
            x2 = x[:, 1]
            term1 = (4 - 2.1 * x1 ** 2 + (x1 ** 4) / 3) * x1 ** 2
            term2 = x1 * x2
            term3 = (-4 + 4 * x2 ** 2) * x2 ** 2
            fval = term1 + term2 + term3
            if self.sd == 0:
                noise = np.zeros(n).reshape(n, 1)
            else:
                noise = np.random.normal(0, self.sd, n).reshape(n, 1)
            return -fval.reshape(n, 1) - noise


class mccormick(function2d):
    '''
    Mccormick function
    
    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''

    def __init__(self, bounds=None, sd=None):
        self.input_dim = 2
        if bounds is None:
            self.bounds = [(-1.5, 4), (-3, 4)]
        else:
            self.bounds = bounds
        self.min = [(-0.54719, -1.54719)]
        self.fmin = -1.9133
        if sd == None:
            self.sd = 0
        else:
            self.sd = sd
        self.name = 'Mccormick'

    def f(self, x):
        x = reshape(x, self.input_dim)
        n = x.shape[0]
        if x.shape[1] != self.input_dim:
            return 'wrong input dimension'
        else:
            x1 = x[:, 0]
            x2 = x[:, 1]
            term1 = np.sin(x1 + x2)
            term2 = (x1 - x2) ** 2
            term3 = -1.5 * x1
            term4 = 2.5 * x2
            fval = term1 + term2 + term3 + term4 + 1
            if self.sd == 0:
                noise = np.zeros(n).reshape(n, 1)
            else:
                noise = np.random.normal(0, self.sd, n).reshape(n, 1)
            return fval.reshape(n, 1) + noise


class powers(function2d):
    '''
    Powers function
    
    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''

    def __init__(self, bounds=None, sd=None):
        self.input_dim = 2
        if bounds is None:
            self.bounds = [(-1, 1), (-1, 1)]
        else:
            self.bounds = bounds
        self.min = [(0, 0)]
        self.fmin = 0
        if sd == None:
            self.sd = 0
        else:
            self.sd = sd
        self.name = 'Sum of Powers'

    def f(self, x):
        x = reshape(x, self.input_dim)
        n = x.shape[0]
        if x.shape[1] != self.input_dim:
            return 'wrong input dimension'
        else:
            x1 = x[:, 0]
            x2 = x[:, 1]
            fval = abs(x1) ** 2 + abs(x2) ** 3
            if self.sd == 0:
                noise = np.zeros(n).reshape(n, 1)
            else:
                noise = np.random.normal(0, self.sd, n).reshape(n, 1)
            return fval.reshape(n, 1) + noise


class eggholder:
    def __init__(self, bounds=None, sd=None):
        self.input_dim = 2
        if bounds is None:
            self.bounds = [(-512, 512), (-512, 512)]
        else:
            self.bounds = bounds
        self.min = [(512, 404.2319)]
        self.fmin = -959.6407
        if sd == None:
            self.sd = 0
        else:
            self.sd = sd
        self.name = 'Egg-holder'

    def f(self, X):
        X = reshape(X, self.input_dim)
        n = X.shape[0]
        if X.shape[1] != self.input_dim:
            return 'Wrong input dimension'
        else:
            x1 = X[:, 0]
            x2 = X[:, 1]
            fval = -(x2 + 47) * np.sin(np.sqrt(abs(x2 + x1 / 2 + 47))) + -x1 * np.sin(np.sqrt(abs(x1 - (x2 + 47))))
            if self.sd == 0:
                noise = np.zeros(n).reshape(n, 1)
            else:
                noise = np.random.normal(0, self.sd, n).reshape(n, 1)
            return fval.reshape(n, 1) + noise
