"""
Maintain the kernel core for each species.
Given a set of SOAP features, the kernel core calculates the mean and covariance matrix
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from typing_extensions import override


class KernelCoreBase(ABC):
    """
    Base class for the Kernel Core
    """

    @abstractmethod
    def __init__(self, feature_dim: int, init_eps: float = 1e-3):
        pass

    @abstractmethod
    def step_kernel(self, soap_i: np.ndarray):
        """
        Function to update the intermediate variable of the kernel
        Args:
            soap_i: the soap feature of the ith atom
        """
        pass

    @abstractmethod
    def update_kernel(self):
        """
        Function to update the kernel and its inverse
        """
        pass

    @abstractmethod
    def get_kernel(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return mu, kernel_inv
        """
        pass


class KernelCore(KernelCoreBase):
    """
    Basic Kernel Core for each species
    """

    def __init__(
        self,
        feature_dim: int,
        init_eps: float = 1e-3,
    ):
        """
        args:
            feature_dim: the dimension of the soap features
        """
        self.kernel_An = np.zeros((feature_dim, feature_dim), dtype=np.float32)
        self.soap_mu = np.zeros(feature_dim, dtype=np.float32)
        self.n = 0
        self.kernel = np.eye(feature_dim, dtype=np.float32) * init_eps
        self.kernel_inv = np.linalg.inv(self.kernel)

    @override
    def step_kernel(
        self,
        soap_i: np.ndarray,
    ):
        """
        Function to update the intermediate variable of the kernel
        Args:
            soap_i: the soap feature of the ith atom
        """
        soap_i = soap_i.reshape(-1, 1)  ## [feature_dim, 1]
        self.kernel_An = self.kernel_An + np.matmul(soap_i, soap_i.T)
        self.soap_mu = (self.soap_mu * self.n + soap_i.reshape(-1)) / (self.n + 1)
        self.n += 1

    @override
    def update_kernel(self):
        """
        Function to update the kernel and its inverse
        """
        soap_mu = self.soap_mu.reshape(-1, 1)  # [feature_dim, 1]
        soap_sum = soap_mu * self.n  # [feature_dim, 1]
        self.kernel = (
            self.kernel_An
            - soap_sum @ soap_mu.T
            - soap_mu @ soap_sum.T
            + (soap_mu @ soap_mu.T) * self.n
        ) / self.n

        try:
            self.kernel_inv = np.linalg.inv(self.kernel)
        except np.linalg.LinAlgError:
            self.kernel_inv = np.linalg.pinv(self.kernel)

    @override
    def get_kernel(
        self,
    ):
        """
        Return the kernel and its inverse
        """
        return self.soap_mu, self.kernel_inv


class KernelCoreIncremental(KernelCoreBase):
    """
    Incremental version of Kernel Core for each species.
    Using the MLE covariance (divide by n) and rank-1 update
    to avoid expensive matrix inversion in each step.
    """

    def __init__(self, feature_dim: int, init_eps: float = 1e-3):
        """
        Args:
            feature_dim: the dimension of the soap features
            init_eps: initial diagonal regularization for the covariance
                      to avoid singular matrix at the beginning.
        """
        self.feature_dim = feature_dim
        self.n = 0
        self.soap_mu = np.zeros(feature_dim, dtype=np.float32)
        self.kernel_inv = (1.0 / init_eps) * np.eye(feature_dim, dtype=np.float32)

    @override
    def step_kernel(self, soap_i: np.ndarray):
        """
        Online (incremental) update with a new SOAP vector soap_i.
        This updates both the mean vector and the inverse covariance matrix.
        """
        soap_i = soap_i.reshape(-1)

        if self.n == 0:
            self.soap_mu = soap_i.copy()  # mean = x
            self.n = 1
            return

        u = soap_i - self.soap_mu

        n_new = self.n + 1
        gamma = 1.0 / n_new
        self.soap_mu = self.soap_mu + gamma * u  # mu_{n+1} = mu_n + (1/(n+1))*u

        # 协方差的增量更新:
        #   init_eps_{n+1} = alpha * (init_eps_n + gamma * u*u^T), 其中 alpha = n/(n+1), gamma = 1/(n+1).
        #   => init_eps_{n+1}^{-1} = (1/alpha)*[init_eps_n^{-1} - ... rank-1 update ...]

        alpha = float(self.n) / n_new  # n/(n+1)

        # 使用 Woodbury 公式做逆矩阵的 rank-1 更新
        tmp = self.kernel_inv @ u  # shape (d,)
        denom = 1.0 + gamma * (u @ tmp)  # 标量

        factor = gamma / denom  # 标量
        self.kernel_inv = (1.0 / alpha) * (
            self.kernel_inv - factor * np.outer(tmp, tmp)
        )

        self.n = n_new

    @override
    def update_kernel(self):
        """
        Updates the mean and covariance matrix.
        """
        pass

    @override
    def get_kernel(self):
        """
        Returns the current inverse covariance matrix and the current mean.
        """
        return self.soap_mu, self.kernel_inv
