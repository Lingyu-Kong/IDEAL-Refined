"""
Sample high-unc local envs, crop off substructure, and optimize
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np
from ase import Atoms
from pydantic import BaseModel

from .cut_strategy import CSCECutStrategy
from .sub_optimizer import SubOptimizerBase
from .unc_module import UncModuleBase


class UncThresholdConfig(BaseModel):
    """
    Configuration for uncertainty threshold calculator.
    """

    window_size: int = 1000
    alpha: float = 0.5
    beta: float = 0.9
    k: float = 1.0
    min_samples: int = 100

    def __init__(self, **data):
        super().__init__(**data)
        assert 0 <= self.alpha <= 1
        assert 0 <= self.beta < 1

    def create_calculator(self):
        return UncThresholdCalculator(
            window_size=self.window_size,
            alpha=self.alpha,
            beta=self.beta,
            k=self.k,
            min_samples=self.min_samples,
        )


class UncThresholdCalculator:
    def __init__(self, window_size, alpha, beta, k, min_samples):
        """
        Initialize the threshold calculator.

        Args:
            window_size (int): Maximum size of the sliding window for replay buffer.
            alpha (float): Weighting factor for combining percentile and mean+std thresholds (0 <= alpha <= 1).
            beta (float): Weight for historical values in weighted mean and std updates (0 <= beta < 1).
            k (float): Scaling factor for std in mean+std threshold.
        """
        self.window_size = window_size
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.min_samples = min_samples

        self.replay_buffer = deque(maxlen=self.window_size)
        self.weighted_mean = 0
        self.weighted_std = 1.0
        self.initialized = False

    def _update_stats(self, new_value):
        """
        Update weighted mean and std using the new value and historical stats.
        """
        if not self.initialized:
            # Initialize with the first value
            self.weighted_mean = new_value
            self.weighted_std = 0
            self.initialized = True
        else:
            # Update weighted mean
            self.weighted_mean = (
                self.beta * self.weighted_mean + (1 - self.beta) * new_value
            )
            # Update weighted std
            self.weighted_std = np.sqrt(
                self.beta * (self.weighted_std**2)
                + (1 - self.beta) * ((new_value - self.weighted_mean) ** 2)
            )

    def update(self, unc_values):
        """
        Update the replay buffer and statistics with new uncertainty values.

        Args:
            unc_values (list[float] | np.ndarray): New uncertainty values to add to the buffer.
        """
        unc_values = np.array(unc_values)
        for value in unc_values:
            self.replay_buffer.append(value)
            self._update_stats(value)

    def get_threshold(self, percentile=95):
        """
        Compute the combined threshold based on percentile and mean+std.

        Args:
            percentile (float): Percentile to calculate (0-100).

        Returns:
            float: The combined uncertainty threshold.
        """
        mean_std_threshold = self.weighted_mean + self.k * self.weighted_std
        if len(self.replay_buffer) > self.min_samples:
            buffer_percentile = np.percentile(self.replay_buffer, percentile)
        else:
            buffer_percentile = mean_std_threshold

        # Combine thresholds using alpha
        combined_threshold = (
            self.alpha * buffer_percentile + (1 - self.alpha) * mean_std_threshold
        )
        return combined_threshold

    def reset(self):
        """
        Reset the replay buffer and statistics to default values.
        """
        self.replay_buffer.clear()
        self.weighted_mean = 0
        self.weighted_std = 1.0
        self.initialized = False


class SubSampler:
    def __init__(
        self,
        species: list[str],
        unc_model: UncModuleBase,
        unc_threshold_config: UncThresholdConfig,
        sub_optimizer: SubOptimizerBase,
        cut_strategy: CSCECutStrategy,
    ):
        self.unc_model = unc_model
        self.sub_optimizer = sub_optimizer
        self.sub_optimizer.assign_unc_model(unc_model)
        self.cut_strategy = cut_strategy
        self.cut_strategy.assign_unc_model(unc_model)
        self.species = species
        self.unc_calculators = {
            specie: unc_threshold_config.create_calculator() for specie in species
        }

    def update_unc_model(
        self,
        atoms: Atoms,
        include_indices: list[int],
    ):
        self.unc_model.update(atoms=atoms, include_indices=include_indices)
        uncs, _ = self.unc_model.unc_predict(
            atoms=atoms,
            include_grad=False,
            unc_include_indices=include_indices,
            reduction=False,
        )
        chemical_symbols = atoms.get_chemical_symbols()
        for specie in self.species:
            specie_indices = [
                include_indices.index(i)
                for i, symbol in enumerate(chemical_symbols)
                if symbol == specie and i in include_indices
            ]
            self.unc_calculators[specie].update(uncs[specie_indices])  # type: ignore

    def get_unc_threshold(self):
        unc_thresholds = {
            specie: calculator.get_threshold()
            for specie, calculator in self.unc_calculators.items()
        }
        return unc_thresholds

    def _normalize_atoms(self, atoms: Atoms):
        scaled_pos = atoms.get_scaled_positions()
        scaled_pos = np.mod(scaled_pos, 1)
        atoms.set_scaled_positions(scaled_pos)
        return atoms

    def sample(
        self,
        atoms: Atoms,
        include_indices: list[int] | None = None,
        max_samples: int | None = None,
    ) -> list[Atoms]:
        atoms = self._normalize_atoms(atoms)
        if include_indices is None:
            include_indices = list(range(len(atoms)))
        chemical_symbols = atoms.get_chemical_symbols()
        uncs, _ = self.unc_model.unc_predict(
            atoms=atoms,
            include_grad=False,
            unc_include_indices=include_indices,
            reduction=False,
        )
        assert type(uncs) == np.ndarray and len(uncs) == len(include_indices)
        unc_thresholds = self.get_unc_threshold()
        print(f"Uncertainty thresholds: {unc_thresholds}")
        sampled_local_env_indices = []
        for i, unc in enumerate(uncs):
            symbol_i = chemical_symbols[include_indices[i]]
            if unc > unc_thresholds[symbol_i]:
                sampled_local_env_indices.append(include_indices[i])
        if max_samples is not None:
            sampled_local_env_indices = np.random.choice(
                sampled_local_env_indices, size=max_samples, replace=False
            ).tolist()

        print(f"Sampled {len(sampled_local_env_indices)} local environments")
        if len(sampled_local_env_indices) == 0:
            return []
        else:
            self.update_unc_model(
                atoms=atoms,
                include_indices=sampled_local_env_indices,
            )
            time1 = time.time()
            subs = self.cut_strategy.cut(
                atoms=atoms,
                center_indices=sampled_local_env_indices,
            )
            sub_sizes = [len(sub) for sub in subs]
            assert len(subs) == len(sampled_local_env_indices)
            print(
                f"Cut {len(subs)} substructures in {time.time() - time1:.2f}s, Average size: {np.mean(sub_sizes):.2f}"
            )
            time2 = time.time()
            opted_subs = []
            for j, sub in enumerate(subs):
                movable_indices = sub.info["movable_indices"]
                opted_sub = self.sub_optimizer.optimize(
                    atoms=sub,
                    grad_include_indices=movable_indices,
                )
                unc_ori, unc_opt = opted_sub.info["unc_ori"], opted_sub.info["unc_opt"]
                opted_subs.append(opted_sub)
                print(
                    f"Optimized substructure: {j+1}/{len(subs)}, [{unc_ori:.2f}->{unc_opt:.2f}, {(unc_ori - unc_opt) / unc_ori:.2f}] in {time.time() - time2:.2f}s"
                )

            return opted_subs

    def sample_all(
        self,
        atoms: Atoms,
        include_indices: list[int] | None = None,
        max_samples: int | None = None,
    ):
        """
        Sample all included local environments and generate optimized substructures.
        This is useful for the first iteration of the subsampling process.
        """
        atoms = self._normalize_atoms(atoms)
        if include_indices is None:
            include_indices = list(range(len(atoms)))
        chemical_symbols = atoms.get_chemical_symbols()
        sampled_local_env_indices = include_indices
        if max_samples is not None:
            sampled_local_env_indices = np.random.choice(
                include_indices, size=max_samples, replace=False
            ).tolist()
        print(f"Sampled {len(sampled_local_env_indices)} local environments")
        if len(sampled_local_env_indices) == 0:
            return []
        else:
            self.update_unc_model(
                atoms=atoms,
                include_indices=sampled_local_env_indices,
            )
            time1 = time.time()
            subs = self.cut_strategy.cut(
                atoms=atoms,
                center_indices=sampled_local_env_indices,
            )
            sub_sizes = [len(sub) for sub in subs]
            assert len(subs) == len(sampled_local_env_indices)
            print(
                f"Cut {len(subs)} substructures in {time.time() - time1:.2f}s, Average size: {np.mean(sub_sizes):.2f}"
            )
            time2 = time.time()
            opted_subs = []
            for j, sub in enumerate(subs):
                movable_indices = sub.info["movable_indices"]
                opted_sub = self.sub_optimizer.optimize(
                    atoms=sub,
                    grad_include_indices=movable_indices,
                )
                unc_ori, unc_opt = opted_sub.info["unc_ori"], opted_sub.info["unc_opt"]
                opted_subs.append(opted_sub)
                print(
                    f"Optimized substructure: {j+1}/{len(subs)}, [{unc_ori:.2f}->{unc_opt:.2f}, {(unc_ori - unc_opt) / unc_ori:.2f}] in {time.time() - time2:.2f}s"
                )

            return opted_subs
