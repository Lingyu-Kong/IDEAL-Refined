"""
The IdealCalculator class is a MLFF based ase calculator.
It implements the ideal active learning algorithm.
"""

from __future__ import annotations

import copy
import time

import numpy as np
import torch
import wandb
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.io import write
from ase.stress import full_3x3_to_voigt_6_stress
from ase.units import GPa
from typing_extensions import TypedDict, override

from ideal.abinitio_interfaces import AbinitioInterfaceConfigBase
from ideal.models.m3gnet.utils.build import build_dataloader
from ideal.models.potential import Potential, batch_to_dict
from ideal.subsampler.sampler import SubSamplerConfig


class ImportanceSamplingConfig(TypedDict):
    """
    This configuration class is used to store the params associated with the importance sampling.
    """

    temperature: float
    temperature_decay: float
    min_temperature: float
    key: str  ## "energy", "forces", "stresses"


class IDEALModelTuningConfig(TypedDict):
    """
    This configuration class is used to store the params associated with the model tuning during the active learning process.
    """

    max_epochs: int
    batch_size: int
    learning_rate: float
    optimizer: str  ## Adam, AdamW, etc.
    scheduler: str  ## StepLR, CosineAnnealingLR, etc.
    loss: torch.nn.modules.Module
    include_energy: bool
    include_forces: bool
    include_stresses: bool
    force_loss_ratio: float
    stress_loss_ratio: float


class IDEALCalculator(Calculator):
    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(
        self,
        *,
        mode: str = "ideal",
        potential: Potential,
        model_tuning_config: IDEALModelTuningConfig,
        importance_sampling: ImportanceSamplingConfig,
        sub_sampler: SubSamplerConfig,
        include_indices: list[int] | None = None,
        max_samples: int | None = None,
        abinitio_interface: AbinitioInterfaceConfigBase,
        compute_stress: bool = True,
        stress_weight: float = 1.0,
        device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
        wandb_log: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert mode in [
            "ideal",
            "offline",
        ], "mode should be in ['ideal', 'offline']"
        self.mode = mode
        self.potential = potential.to(torch.device(device))
        self.model_tuning_config = model_tuning_config
        self.importance_sampling = importance_sampling
        self.temperture = importance_sampling["temperature"]
        self.sub_sampler = sub_sampler.create_sampler()
        self.include_indices = include_indices
        self.max_samples = max_samples
        self.abinitio_interface = abinitio_interface.create_interface()
        self.compute_stress = compute_stress
        self.stress_weight = stress_weight * GPa
        self.device = torch.device(device)
        self.initialized = False
        self.data_buffer: list[Atoms] = []
        self.data_buffer_energies: list[float] = []
        self.data_buffer_forces: list[np.ndarray] = []
        self.data_buffer_stresses: list[np.ndarray] = []
        self.data_buffer_exported = False
        self.error_buffer = np.zeros(0)
        self.wandb_log = wandb_log

    def _configure_optimizers(self):
        """
        This function is used to configure the optimizers and schedulers
        """
        if self.model_tuning_config["optimizer"].lower() == "adam":
            optimizer = torch.optim.Adam(
                self.potential.parameters(),
                lr=self.model_tuning_config["learning_rate"],
            )
        elif self.model_tuning_config["optimizer"].lower() == "adamW":
            optimizer = torch.optim.AdamW(
                self.potential.parameters(),
                lr=self.model_tuning_config["learning_rate"],
            )
        else:
            raise NotImplementedError

        self.potential.optimizer = optimizer

        if self.model_tuning_config["scheduler"].lower() == "steplr":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                gamma=0.8,
                step_size=10,
            )
        elif self.model_tuning_config["scheduler"].lower() == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.model_tuning_config["max_epochs"], eta_min=1e-6
            )
        else:
            raise NotImplementedError

        self.potential.scheduler = scheduler

    def _model_tune(self, indices: list[int], max_epochs: int | None = None):
        """
        This function is used to tune the model
        """
        self._configure_optimizers()
        if torch.cuda.is_available():
            torch.cuda.set_device(self.device)
            torch.cuda.empty_cache()
        dataloader = build_dataloader(
            [self.data_buffer[i] for i in indices],
            energies=[self.data_buffer_energies[i] for i in indices],
            forces=[self.data_buffer_forces[i] for i in indices],
            stresses=[self.data_buffer_stresses[i] for i in indices],
            model_type=self.potential.model_name,
            cutoff=self.potential.model.model_args["cutoff"],
            threebody_cutoff=self.potential.model.model_args["threebody_cutoff"],
            batch_size=self.model_tuning_config["batch_size"],
            shuffle=False,
        )
        loss_, e_mae, f_mae, s_mae = (
            [0] * len(indices),
            [0] * len(indices),
            [0] * len(indices),
            [0] * len(indices),
        )
        if max_epochs is None:
            max_epochs = self.model_tuning_config["max_epochs"]
        for epoch in range(max_epochs):
            loss_avg_, e_mae, f_mae, s_mae = self.potential.train_one_epoch(
                dataloader=dataloader,
                epoch=epoch,
                loss=self.model_tuning_config["loss"],
                include_energy=self.model_tuning_config["include_energy"],
                include_forces=self.model_tuning_config["include_forces"],
                include_stresses=self.model_tuning_config["include_stresses"],
                loss_f=self.model_tuning_config["force_loss_ratio"],
                loss_s=self.model_tuning_config["stress_loss_ratio"],
                wandb=None,
                model="train",
                log=False,
                reduction="none",
            )
            self.potential.scheduler.step()  # type: ignore
            e_mae_mean = np.mean(e_mae).item()
            f_mae_mean = np.mean(f_mae).item()
            s_mae_mean = np.mean(s_mae).item()
            print(
                f"Epoch: {epoch}/{max_epochs}, Loss: {loss_avg_:.4f}, Energy MAE: {e_mae_mean:.4f}, Force MAE: {f_mae_mean:.4f}, Stress MAE: {s_mae_mean:.4f}"
            )
        # Evaluate the model on the dataset
        return loss_, e_mae, f_mae, s_mae

    def _error_buffer_update(self, data_indices, e_mae, f_mae, s_mae):
        if self.importance_sampling["key"].lower() == "e_mae":
            error = e_mae
        elif self.importance_sampling["key"].lower() == "f_mae":
            error = f_mae
        elif self.importance_sampling["key"].lower() == "s_mae":
            error = s_mae
        else:
            raise NotImplementedError
        assert len(data_indices) == len(error), f"{len(data_indices)} != {len(error)}"
        np.put(self.error_buffer, data_indices, error)

    def _importance_sampling(self, size: int):
        """
        This function is used to perform the importance sampling
        """
        self.temperture = max(
            self.temperture * np.exp(-self.importance_sampling["temperature_decay"]),
            self.importance_sampling["min_temperature"],
        )
        error = np.array(self.error_buffer)
        prob = np.exp(-error / self.temperture)
        prob = prob / prob.sum()
        return np.random.choice(
            list(range(len(self.data_buffer))),
            p=prob,
            size=min(size, len(self.data_buffer)),
            replace=False,
        )

    def initialize(
        self,
        atoms_list: list[Atoms],
    ):
        """
        IDEAL should be initialized with a list of atoms as the initial dataset.
        """
        self.data_buffer = [copy.deepcopy(atoms) for atoms in atoms_list]
        self.data_buffer_energies = [
            atoms.get_potential_energy() for atoms in atoms_list
        ]
        self.data_buffer_forces = [np.array(atoms.get_forces()) for atoms in atoms_list]
        self.data_buffer_stresses = [
            np.array(atoms.get_stress(voigt=False)) for atoms in atoms_list
        ]
        print(
            "Initializing IDEAL algorithm with {} structures".format(
                len(self.data_buffer)
            )
        )
        self.error_buffer = np.random.uniform(low=0.1, high=1.0, size=len(atoms_list))
        data_indices = list(range(len(atoms_list)))
        shuffle_indices = np.random.permutation(data_indices).tolist()
        loss_, e_mae, f_mae, s_mae = self._model_tune(shuffle_indices, max_epochs=200)
        self._error_buffer_update(shuffle_indices, e_mae, f_mae, s_mae)
        for atoms in self.data_buffer:
            self.sub_sampler.update_unc_model_and_threshold(
                atoms=atoms, include_indices=[0]
            )
        self.initialized = True

    def _single_point_calculation(self, atoms: Atoms):
        """
        This function is used to perform a single point calculation
        """
        dataloader = build_dataloader(
            [atoms],
            model_type=self.potential.model_name,
            cutoff=self.potential.model.model_args["cutoff"],
            threebody_cutoff=self.potential.model.model_args["threebody_cutoff"],
            batch_size=1,
            shuffle=False,
            only_inference=True,
        )
        for graph_batch in dataloader:  # type: ignore
            graph_batch = graph_batch.to(self.device)
            input = batch_to_dict(graph_batch)
            with self.potential.ema.average_parameters():
                result = self.potential.forward(
                    input, include_forces=True, include_stresses=self.compute_stress
                )
            energy = result["total_energy"].detach().cpu().numpy()[0]
            free_energy = result["total_energy"].detach().cpu().numpy()[0]
            forces = result["forces"].detach().cpu().numpy()
            if self.compute_stress:
                stress = self.stress_weight * full_3x3_to_voigt_6_stress(
                    result["stresses"].detach().cpu().numpy()[0]
                )
            else:
                stress = np.zeros((6,))
            return energy, free_energy, forces, stress
        raise ValueError("No data in the dataloader")

    @override
    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list | None = None,
        system_changes: list | None = None,
    ):
        """
        Args:
            atoms (ase.Atoms): ase Atoms object
            properties (list): list of properties to calculate
            system_changes (list): monitor which properties of atoms were
                changed for new calculation. If not, the previous calculation
                results will be loaded.
        Returns:
        """

        if self.mode.lower() == "ideal" and not self.initialized:
            raise ValueError(
                "IDEAL algorithm should be initialized with a list of atoms as the initial dataset."
            )

        all_changes = [
            "positions",
            "numbers",
            "cell",
            "pbc",
            "initial_charges",
            "initial_magmoms",
        ]
        properties = properties or ["energy"]
        system_changes = system_changes or all_changes
        super().calculate(
            atoms=atoms, properties=properties, system_changes=system_changes
        )

        sub_sample_time = 0
        sub_label_time = 0
        model_tune_time = 0
        single_point_time = 0
        subs = []
        if self.mode.lower() == "ideal":
            time1 = time.time()
            subs = self.sub_sampler.sample(
                atoms=atoms,  # type: ignore
                include_indices=self.include_indices,
                max_samples=self.max_samples,
            )
            sub_sample_time = time.time() - time1

            ## Label the subs and update the model
            if len(subs) > 0:
                time1 = time.time()
                labeled_subs = []
                for sub_label_idx, sub in enumerate(subs):
                    labeled_sub = self.abinitio_interface.run(sub)
                    if labeled_sub is not None:
                        sub_energy = labeled_sub.get_potential_energy()
                        sub_forces = labeled_sub.get_forces()
                        labeled_subs.append(copy.deepcopy(labeled_sub))
                    print(f"Sub {sub_label_idx + 1}/{len(subs)} labeled")
                sub_label_time = time.time() - time1
                time1 = time.time()
                replay_indices = self._importance_sampling(
                    size=max(
                        len(labeled_subs),
                        self.model_tuning_config["batch_size"] - len(labeled_subs),
                    )
                ).tolist()
                data_indices = replay_indices + list(
                    range(
                        len(self.data_buffer), len(self.data_buffer) + len(labeled_subs)
                    )
                )
                self.data_buffer.extend(labeled_subs)
                self.data_buffer_energies.extend(
                    [sub.get_potential_energy() for sub in labeled_subs]
                )
                self.data_buffer_forces.extend(
                    [np.array(sub.get_forces()) for sub in labeled_subs]
                )
                self.data_buffer_stresses.extend(
                    [np.array(sub.get_stress(voigt=False)) for sub in labeled_subs]
                )
                self.data_buffer_exported = False
                self.error_buffer = np.append(
                    self.error_buffer,
                    np.random.uniform(low=0.1, high=1.0, size=len(labeled_subs)),
                )
                shuffle_indices = np.random.permutation(data_indices).tolist()
                loss_, e_mae, f_mae, s_mae = self._model_tune(shuffle_indices)
                self._error_buffer_update(shuffle_indices, e_mae, f_mae, s_mae)
                model_tune_time = time.time() - time1
        elif self.mode.lower() == "offline":
            pass
        else:
            raise NotImplementedError

        time1 = time.time()
        energy, free_energy, forces, stress = self._single_point_calculation(atoms)  # type: ignore
        self.results.update(
            energy=energy,
            free_energy=free_energy,
            forces=forces,
            stress=stress,
        )
        single_point_time = time.time() - time1

        if self.wandb_log:
            wandb.log(
                {
                    "IDEAL-Calc-Time/sub_sample_time": sub_sample_time,
                    "IDEAL-Calc-Time/sub_label_time": sub_label_time,
                    "IDEAL-Calc-Time/model_tune_time": model_tune_time,
                    "IDEAL-Calc-Time/single_point_time": single_point_time,
                    "IDEAL-Calc-Time/sub_sample_time_per_sub": sub_sample_time
                    / len(subs)
                    if len(subs) > 0
                    else 0,
                    "IDEAL-Calc-Time/sub_label_time_per_sub": sub_label_time / len(subs)
                    if len(subs) > 0
                    else 0,
                    "IDEAL-Calc-Time/model_tune_time_per_sub": model_tune_time
                    / len(subs)
                    if len(subs) > 0
                    else 0,
                }
            )

    def export_dataset(self, filename: str):
        """
        This function is used to export the current dataset
        """
        if not self.data_buffer_exported:
            write(filename, self.data_buffer)
            self.data_buffer_exported = True
