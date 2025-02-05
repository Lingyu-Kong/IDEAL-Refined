"""
Optimize substructure
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ase import Atoms
from pydantic import BaseModel, NonNegativeFloat, PositiveFloat, PositiveInt
from typing_extensions import override

from .unc_module import UncModuleBase


class SubOptimizerConfigBase(BaseModel, ABC):
    @abstractmethod
    def create_sub_optimizer(self) -> SubOptimizerBase:
        pass


class SubOptimizerBase(ABC):
    """
    Base class for substructure optimizer
    """

    @abstractmethod
    def assign_unc_model(self, unc_model: UncModuleBase):
        pass

    @abstractmethod
    def optimize(self, atoms: Atoms, *args, **kwargs) -> Atoms:
        """
        Optimize the substructure
        """
        pass


class UncGradientOptimizerConfig(SubOptimizerConfigBase):
    max_steps: PositiveInt = 200
    lr: NonNegativeFloat = 0.1
    optimizer: str = "adam"
    scheduler: str = "cosine"
    early_stop: bool = True
    early_stop_patience: PositiveInt = 100
    noise: NonNegativeFloat = 0.0
    noise_decay: NonNegativeFloat = 1.0
    grad_clip: PositiveFloat | None = 1.0

    @override
    def create_sub_optimizer(self) -> UncGradientOptimizer:
        return UncGradientOptimizer(
            max_steps=self.max_steps,
            lr=self.lr,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            early_stop=self.early_stop,
            early_stop_patience=self.early_stop_patience,
            noise=self.noise,
            noise_decay=self.noise_decay,
            grad_clip=self.grad_clip,
        )


class UncGradientOptimizer(SubOptimizerBase):
    """
    Optimize the substructure using the uncertainty gradient descent
    """

    def __init__(
        self,
        max_steps: int = 200,
        lr: float = 0.1,
        optimizer: str = "adam",
        scheduler: str = "cosine",
        early_stop: bool = True,
        early_stop_patience: int = 100,
        noise: float = 0.0,
        noise_decay: float = 1.0,
        grad_clip: float | None = 2.0,
    ):
        self.unc_model = None
        self.max_steps = max_steps
        self.lr = lr
        self.optimizer = optimizer.lower()
        self.scheduler = scheduler.lower()
        self.early_stop = early_stop
        self.early_stop_patience = early_stop_patience
        self.noise = noise
        assert 0 <= noise_decay <= 1
        self.noise_decay = noise_decay
        assert grad_clip is None or grad_clip > 0
        self.grad_clip = grad_clip

    @override
    def assign_unc_model(self, unc_model: UncModuleBase):
        self.unc_model = unc_model

    def _configure_optimizer(
        self,
        positions: torch.Tensor,
        cell: torch.Tensor | None = None,
    ):
        assert cell is None, "Cell is not supported for now"
        if self.optimizer == "adam":
            opt = optim.Adam([positions], lr=self.lr)
        else:
            raise ValueError(f"Unsupported optimizer {self.optimizer}")
        if self.scheduler == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, self.max_steps)
        else:
            raise ValueError(f"Unsupported scheduler {self.scheduler}")
        return opt, scheduler

    def _add_noise(
        self, grad: torch.Tensor, grad_include_indices: list[int]
    ) -> torch.Tensor:
        noise = torch.normal(mean=0, std=self.noise_tmp_copy, size=grad.size())
        noise_mask = torch.zeros_like(grad)
        noise_mask[grad_include_indices] = 1
        grad += noise * noise_mask
        self.noise_tmp_copy *= self.noise_decay
        return grad

    @override
    def optimize(
        self,
        atoms: Atoms,
        *,
        grad_include_indices: list[int],
        unc_include_indices: list[int] | None = None,
    ) -> Atoms:
        if self.unc_model is None:
            raise ValueError(
                "Uncertainty model is not assigned, please assign it first using <assign_unc_model()>"
            )
        if unc_include_indices is None:
            unc_include_indices = list(range(len(atoms)))
        if unc_include_indices == []:
            return atoms
        positions = torch.from_numpy(np.array(atoms.positions)).to(torch.float32)
        best_positions = positions.clone().detach().numpy()
        optimizer, scheduler = self._configure_optimizer(positions)
        unc_ori, _ = self.unc_model.unc_predict(atoms=atoms)
        best_unc = unc_ori
        patience = 0
        self.noise_tmp_copy = copy.deepcopy(self.noise)
        for step in range(self.max_steps):
            optimizer.zero_grad()
            atoms.set_positions(best_positions)
            unc, grad = self.unc_model.unc_predict(
                atoms=atoms,
                include_grad=True,
                unc_include_indices=unc_include_indices,
                grad_include_indices=grad_include_indices,
            )
            grad = torch.from_numpy(grad).to(torch.float32)
            if self.noise > 0:
                grad = self._add_noise(grad, grad_include_indices)
            positions.grad = grad
            if self.grad_clip is not None:
                nn.utils.clip_grad_norm_(positions, self.grad_clip)
            optimizer.step()
            scheduler.step()
            atoms.set_positions(positions.clone().detach().numpy())
            if unc < best_unc:
                best_unc = unc
                best_positions = positions.clone().detach().numpy()
                patience = 0
            else:
                patience += 1
                if self.early_stop and patience >= self.early_stop_patience:
                    break
        atoms.set_positions(best_positions)
        atoms.info["unc_ori"] = unc_ori
        atoms.info["unc_opt"] = best_unc
        return atoms
