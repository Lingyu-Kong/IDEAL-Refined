"""
Sample high-unc local envs, crop off substructure, and optimize
"""

from __future__ import annotations

import time

import numpy as np
from ase import Atoms
from pydantic import BaseModel

from .cut_strategy import CSCECutStrategyConfig
from .sub_optimizer import SubOptimizerConfigBase
from .threshold import UncThresholdConfig
from .unc_module import UncModuleConfigBase


class SubSamplerConfig(BaseModel):
    species: list[str]
    unc_model: UncModuleConfigBase
    unc_threshold_config: UncThresholdConfig
    sub_optimizer: SubOptimizerConfigBase
    cut_strategy: CSCECutStrategyConfig

    def create_sampler(self) -> SubSampler:
        return SubSampler(
            species=self.species,
            unc_model=self.unc_model,
            unc_threshold_config=self.unc_threshold_config,
            sub_optimizer=self.sub_optimizer,
            cut_strategy=self.cut_strategy,
        )


class SubSampler:
    def __init__(
        self,
        species: list[str],
        unc_model: UncModuleConfigBase,
        unc_threshold_config: UncThresholdConfig,
        sub_optimizer: SubOptimizerConfigBase,
        cut_strategy: CSCECutStrategyConfig,
    ):
        self.unc_model = unc_model.create_unc_module()
        self.sub_optimizer = sub_optimizer.create_sub_optimizer()
        self.sub_optimizer.assign_unc_model(self.unc_model)
        self.cut_strategy = cut_strategy.create_cut_strategy()
        self.cut_strategy.assign_unc_model(self.unc_model)
        self.species = species
        self.threshold_calculators = {
            specie: unc_threshold_config.create_calculator() for specie in species
        }

    def update_unc_model_and_threshold(
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

        for idx in range(len(include_indices)):
            self.threshold_calculators[chemical_symbols[include_indices[idx]]].update(
                uncs[idx]  # type: ignore
            )

    def get_unc_threshold(self):
        unc_thresholds = {
            specie: calculator.get_threshold()
            for specie, calculator in self.threshold_calculators.items()
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
        sampled_uncs = []
        for i, unc in enumerate(uncs):
            symbol_i = chemical_symbols[include_indices[i]]
            if unc > unc_thresholds[symbol_i]:
                sampled_local_env_indices.append(include_indices[i])
                sampled_uncs.append(unc)
        if max_samples is not None:
            sampled_local_env_indices = np.random.choice(
                sampled_local_env_indices,
                size=min(max_samples, len(sampled_local_env_indices)),
                replace=False,
            ).tolist()

        print(
            f"Sampled {len(sampled_local_env_indices)} local environments, indices: {sampled_local_env_indices}, unc: {[f'{unc:.2f}' for unc in sampled_uncs]}"
        )

        if len(sampled_local_env_indices) == 0:
            return []
        else:
            self.update_unc_model_and_threshold(
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
                    f"Optimized substructure: {j + 1}/{len(subs)}, [{unc_ori:.2f}->{unc_opt:.2f}, {(unc_ori - unc_opt) / unc_ori:.2f}] in {time.time() - time2:.2f}s"
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
                sampled_local_env_indices,
                size=min(max_samples, len(sampled_local_env_indices)),
                replace=False,
            ).tolist()
        print(f"Sampled {len(sampled_local_env_indices)} local environments")
        if len(sampled_local_env_indices) == 0:
            return []
        else:
            self.update_unc_model_and_threshold(
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
                    f"Optimized substructure: {j + 1}/{len(subs)}, [{unc_ori:.2f}->{unc_opt:.2f}, {(unc_ori - unc_opt) / unc_ori:.2f}] in {time.time() - time2:.2f}s"
                )

            return opted_subs
