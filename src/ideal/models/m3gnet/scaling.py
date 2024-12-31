# my_atom_scaling.py

"""
Atomic scaling module. Used for predicting extensive properties.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from ase import Atoms
from torch_runstats.scatter import scatter_mean
from typing_extensions import override

# 这里假设 solver 返回的是 (mean, std) 之类的 Tuple[torch.Tensor, torch.Tensor]
from .utils.regressor import solver

DATA_INDEX: dict[str, int] = {
    "total_energy": 0,
    "forces": 2,
    "per_atom_energy": 1,
    "per_species_energy": 0,
}


class AtomScaling(nn.Module):
    """
    Atomic extensive property rescaling module.

    This class provides methods to shift and scale atomic energies based on
    per-species or per-atom statistics. It can be used to improve training
    stability and performance when modeling extensive atomic properties.
    """

    def __init__(
        self,
        atoms: list[Atoms] | None = None,
        total_energy: list[float] | None = None,
        forces: list[np.ndarray] | None = None,
        atomic_numbers: list[np.ndarray] | None = None,
        num_atoms: list[float] | None = None,
        max_z: int = 94,
        scale_key: str | None = None,
        shift_key: str | None = None,
        init_scale: float | torch.Tensor | None = None,
        init_shift: float | torch.Tensor | None = None,
        trainable_scale: bool = False,
        trainable_shift: bool = False,
        verbose: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs: Any,
    ) -> None:
        """
        Args:
            atoms (list[Atoms] | None): A list of ASE Atoms objects.
            total_energy (list[float] | None): A list of total energies corresponding to each system.
            forces (list[np.ndarray] | None): A list of force arrays, each shape = (n_atoms, 3).
            atomic_numbers (list[np.ndarray] | None): A list of arrays containing atomic numbers for each system.
            num_atoms (list[float] | None): A list indicating the number of atoms in each system.
            max_z (int): Maximum atomic number. If scale_key or shift_key is used,
                         this should match the maximum Z in the dataset.
            scale_key (str | None): Key specifying how to compute the initial scale values.
            shift_key (str | None): Key specifying how to compute the initial shift values.
            init_scale (float | torch.Tensor | None): If provided, directly sets the initial scale.
            init_shift (float | torch.Tensor | None): If provided, directly sets the initial shift.
            trainable_scale (bool): Whether the scale parameters are learnable.
            trainable_shift (bool): Whether the shift parameters are learnable.
            verbose (bool): If True, print scale and shift at initialization.
            device (str): Torch device, defaults to "cuda" if available, otherwise "cpu".
            **kwargs: Additional arguments for extensibility.
        """
        super().__init__()

        self.max_z = max_z
        self.device = device

        # === Data preprocessing (only if scale_key or shift_key is given) ===
        if scale_key or shift_key:
            if total_energy is None:
                raise ValueError(
                    "`total_energy` cannot be None when scale_key or shift_key is used."
                )
            total_energy_t = torch.from_numpy(
                np.array(total_energy, dtype=np.float32)
            )  # (N,)

            if forces is not None:
                # Concatenate all force arrays into a single tensor
                forces_t = torch.from_numpy(
                    np.concatenate(forces, axis=0).astype(np.float32)
                )
            else:
                forces_t = None

            # If not provided, extract atomic_numbers from Atoms objects
            if atomic_numbers is None and atoms is not None:
                atomic_numbers_list = [atom.get_atomic_numbers() for atom in atoms]
                atomic_numbers_t = torch.from_numpy(
                    np.concatenate(atomic_numbers_list, axis=0)
                )
            elif atomic_numbers is not None:
                atomic_numbers_t = torch.from_numpy(
                    np.concatenate(atomic_numbers, axis=0)
                )
            else:
                raise ValueError("Either `atomic_numbers` or `atoms` must be provided.")

            atomic_numbers_t = atomic_numbers_t.long()  # shape: (sum(n_atoms),)

            # If not provided, compute num_atoms from Atoms objects
            if num_atoms is None and atoms is not None:
                num_atoms_list = [atom.positions.shape[0] for atom in atoms]
                num_atoms_t = torch.tensor(num_atoms_list, dtype=torch.float32)
            elif num_atoms is not None:
                num_atoms_t = torch.tensor(num_atoms, dtype=torch.float32)
            else:
                raise ValueError("Either `num_atoms` or `atoms` must be provided.")

            per_atom_energy_t = total_energy_t / num_atoms_t  # shape: (N,)

            # Sanity checks
            if num_atoms_t.shape[0] != total_energy_t.shape[0]:
                raise ValueError(
                    f"`num_atoms` and `total_energy` must have same length, "
                    f"got {num_atoms_t.shape[0]} vs {total_energy_t.shape[0]}."
                )
            if forces_t is not None and forces_t.shape[0] != atomic_numbers_t.shape[0]:
                raise ValueError(
                    f"`forces` (concatenated) and `atomic_numbers` must have same length, "
                    f"got {forces_t.shape[0]} vs {atomic_numbers_t.shape[0]}."
                )

            data_list = [total_energy_t, per_atom_energy_t, forces_t]

            # === Calculate the scaling factors ===
            # Special case: "per_species_energy_std" & "per_species_energy_mean"
            # get both shift & scale from a GP-based approach
            if (
                scale_key == "per_species_energy_std"
                and shift_key == "per_species_energy_mean"
                and init_shift is None
                and init_scale is None
            ):
                init_shift_t, init_scale_t = self.get_gaussian_statistics(
                    atomic_numbers_t, num_atoms_t, total_energy_t
                )
                init_scale = init_scale_t
                init_shift = init_shift_t
            else:
                # shift
                if shift_key and init_shift is None:
                    init_shift_t = self.get_statistics(
                        key=shift_key,
                        max_z=max_z,
                        data_list=data_list,
                        atomic_numbers=atomic_numbers_t,
                        num_atoms=num_atoms_t,
                    )
                    init_shift = init_shift_t
                # scale
                if scale_key and init_scale is None:
                    init_scale_t = self.get_statistics(
                        key=scale_key,
                        max_z=max_z,
                        data_list=data_list,
                        atomic_numbers=atomic_numbers_t,
                        num_atoms=num_atoms_t,
                    )
                    init_scale = init_scale_t

        # === Initialize scale ===
        if init_scale is None:
            init_scale_tensor = torch.ones(max_z + 1, dtype=torch.float32)
        elif isinstance(init_scale, float):
            init_scale_tensor = torch.tensor(
                [init_scale] * (max_z + 1), dtype=torch.float32
            )
        else:
            assert isinstance(init_scale, torch.Tensor)
            if init_scale.shape[0] != max_z + 1:
                raise ValueError(
                    f"init_scale must have shape [{max_z + 1}], got {init_scale.shape[0]}."
                )
            init_scale_tensor = init_scale.float()

        # === Initialize shift ===
        if init_shift is None:
            init_shift_tensor = torch.zeros(max_z + 1, dtype=torch.float32)
        elif isinstance(init_shift, float):
            init_shift_tensor = torch.tensor(
                [init_shift] * (max_z + 1), dtype=torch.float32
            )
        else:
            # Assume it's a tensor
            assert isinstance(init_shift, torch.Tensor)
            if init_shift.shape[0] != max_z + 1:
                raise ValueError(
                    f"init_shift must have shape [{max_z + 1}], got {init_shift.shape[0]}."
                )
            init_shift_tensor = init_shift.float()

        # === Register as Parameter or Buffer ===
        if trainable_scale:
            self.scale = nn.Parameter(init_scale_tensor)
        else:
            self.register_buffer("scale", init_scale_tensor)

        if trainable_shift:
            self.shift = nn.Parameter(init_shift_tensor)
        else:
            self.register_buffer("shift", init_shift_tensor)

        if verbose:
            print("Current scale:", init_scale_tensor)
            print("Current shift:", init_shift_tensor)

        self.to(device)

    def transform(
        self, atomic_energies: torch.Tensor, atomic_numbers: torch.Tensor
    ) -> torch.Tensor:
        """
        Transform original atomic energies by applying (energy * scale + shift).

        Args:
            atomic_energies (torch.Tensor): Shape (N,) atomic energies.
            atomic_numbers (torch.Tensor):  Shape (N,) atomic numbers.

        Returns:
            torch.Tensor: Transformed atomic energies of shape (N,).
        """
        curr_shift = self.shift[atomic_numbers]
        curr_scale = self.scale[atomic_numbers]
        return curr_scale * atomic_energies + curr_shift

    def inverse_transform(
        self, atomic_energies: torch.Tensor, atomic_numbers: torch.Tensor
    ) -> torch.Tensor:
        """
        Invert the transformation, converting scaled energies back to raw energies.

        Args:
            atomic_energies (torch.Tensor): Shape (N,) scaled atomic energies.
            atomic_numbers (torch.Tensor):  Shape (N,) atomic numbers.

        Returns:
            torch.Tensor: Original atomic energies of shape (N,).
        """
        curr_shift = self.shift[atomic_numbers]
        curr_scale = self.scale[atomic_numbers]
        return (atomic_energies - curr_shift) / curr_scale

    @override
    def forward(
        self, atomic_energies: torch.Tensor, atomic_numbers: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass. Alias for `transform`.

        Args:
            atomic_energies (torch.Tensor): Shape (N,) atomic energies.
            atomic_numbers (torch.Tensor):  Shape (N,) atomic numbers.

        Returns:
            torch.Tensor: Transformed atomic energies (N,).
        """
        return self.transform(atomic_energies, atomic_numbers)

    def get_statistics(
        self,
        key: str,
        max_z: int,
        data_list: list[torch.Tensor | None],
        atomic_numbers: torch.Tensor,
        num_atoms: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute scale/shift statistics according to `key`.

        Valid keys include:
            * total_energy_mean
            * per_atom_energy_mean
            * per_species_energy_mean
            * per_species_energy_mean_linear_reg
            * total_energy_std
            * per_atom_energy_std
            * per_species_energy_std
            * forces_rms
            * per_species_forces_rms

        Args:
            key (str): The string specifying which statistic to compute.
            max_z (int): Maximum Z.
            data_list (list[Optional[torch.Tensor]]): [total_energy, per_atom_energy, forces],
                in positions indicated by DATA_INDEX. Some could be None if not used.
            atomic_numbers (torch.Tensor): Shape (sum(n_atoms),).
            num_atoms (torch.Tensor): Shape (N_graphs,).

        Returns:
            torch.Tensor: A tensor of shape (max_z+1,) with the computed statistic for each species.
        """
        # Find the data by the index in DATA_INDEX
        data: torch.Tensor | None = None
        for data_key, idx in DATA_INDEX.items():
            if data_key in key:
                data = data_list[idx]
                break
        if data is None:
            raise ValueError(f"Could not find valid data for key '{key}'.")

        statistics_t: Any

        # =============== SHIFT KEYS =============== #
        if "mean" in key:
            if "per_species" in key:
                # Per-species approach often uses a solver or regression
                n_atoms = torch.repeat_interleave(num_atoms)  # shape: (sum(n_atoms),)
                if "linear_reg" in key:
                    # Example: linear regression approach
                    features_np = bincount(
                        atomic_numbers, n_atoms, minlength=max_z + 1
                    ).numpy()
                    data_np = data.numpy()

                    # 去掉全零行，以防某些 batch 索引不连续
                    valid_mask = (features_np > 0).any(axis=1)
                    features_np = features_np[valid_mask]
                    data_np = data_np[valid_mask]

                    # (Pseudo) Linear regression
                    pinv = np.linalg.pinv(features_np.T @ features_np)
                    coef = pinv @ features_np.T @ data_np
                    statistics_t = torch.from_numpy(coef)
                else:
                    # GPR / NormalizedGaussianProcess approach
                    N = bincount(atomic_numbers, n_atoms, minlength=max_z + 1)
                    valid_mask = (N > 0).any(dim=1)
                    N = N[valid_mask]
                    # Use solver from your own regressor
                    mean_t, _ = solver(
                        N, data[valid_mask], regressor="NormalizedGaussianProcess"
                    )
                    statistics_t = mean_t
            else:
                # Simple global mean
                statistics_val = torch.mean(data)
                statistics_t = statistics_val.repeat(max_z + 1)

        # =============== SCALE KEYS =============== #
        elif "std" in key:
            if "per_species" in key:
                # Per-species approach
                print(
                    "Warning: calculating per_species_energy_std for "
                    "full periodic table systems can be risky. "
                    "Consider using per_species_forces_rms if you want force scaling."
                )
                n_atoms = torch.repeat_interleave(num_atoms)
                N = bincount(atomic_numbers, n_atoms, minlength=max_z + 1)
                valid_mask = (N > 0).any(dim=1)
                N = N[valid_mask]
                # solver returns (mean, std)
                _, std_t = solver(
                    N, data[valid_mask], regressor="NormalizedGaussianProcess"
                )
                statistics_t = std_t
            else:
                # Global std
                statistics_val = torch.std(data)
                statistics_t = statistics_val.repeat(max_z + 1)

        elif "rms" in key:
            if "per_species" in key:
                # per-species forces RMS
                # data expected shape: (sum(n_atoms), 3) or just (sum(n_atoms),)
                # scatter_mean 不同情况需要你根据实际 reshape
                square_t = scatter_mean(
                    data.square(), atomic_numbers, dim=0, dim_size=max_z + 1
                )
                # square_t 形状如果是 (max_z+1, 3)，可以再做一次 mean(axis=-1)
                # 如果是标量，需要据项目需求调整
                # 这里假设 forces 形状是 (..., 3)
                if square_t.ndim == 2:
                    # shape: (max_z+1, 3)
                    statistics_t = square_t.mean(dim=-1).sqrt()
                else:
                    # shape: (max_z+1,)
                    statistics_t = square_t.sqrt()
            else:
                # global RMS
                statistics_val = torch.sqrt(torch.mean(data.square()))
                statistics_t = statistics_val.repeat(max_z + 1)
        else:
            raise ValueError(f"Unknown key '{key}' for get_statistics.")

        if statistics_t.shape[0] != max_z + 1:
            # 广播或 shape 校验
            statistics_t = statistics_t.reshape(-1).repeat(max_z + 1)[: (max_z + 1)]

        return statistics_t

    def get_gaussian_statistics(
        self,
        atomic_numbers: torch.Tensor,
        num_atoms: torch.Tensor,
        total_energy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Use a Gaussian Process regressor to get per-species mean and std
        for the given total energies.

        Args:
            atomic_numbers (torch.Tensor): Concatenated atomic numbers (sum(n_atoms),).
            num_atoms (torch.Tensor): Number of atoms in each system (N,).
            total_energy (torch.Tensor): Total energy for each system (N,).

        Returns:
            (mean, std):
                mean: torch.Tensor shape (max_z+1,)
                std: torch.Tensor shape (max_z+1,)
        """
        n_atoms = torch.repeat_interleave(num_atoms)  # shape: (sum(n_atoms),)
        N = bincount(atomic_numbers, n_atoms, minlength=self.max_z + 1)
        # 去掉全零行（非连续 batch index 等情况）
        valid_mask = (N > 0).any(dim=1)
        N = N[valid_mask]
        E = total_energy[valid_mask]

        mean_t, std_t = solver(N, E, regressor="NormalizedGaussianProcess")
        assert type(mean_t) == torch.Tensor and type(std_t) == torch.Tensor
        if mean_t.shape[0] != self.max_z + 1:
            raise ValueError(
                f"Mean shape mismatch, expected {self.max_z + 1}, got {mean_t.shape[0]}."
            )
        if std_t.shape[0] != self.max_z + 1:
            raise ValueError(
                f"Std shape mismatch, expected {self.max_z + 1}, got {std_t.shape[0]}."
            )
        return mean_t, std_t


def bincount(
    input: torch.Tensor, batch: torch.Tensor | None = None, minlength: int = 0
) -> torch.Tensor:
    """
    A custom bincount supporting (value, batch) usage similar to PyG.

    Args:
        input (torch.Tensor): 1D tensor of integer class labels.
        batch (torch.Tensor | None): If provided, shape must be the same as `input`.
                                     This is used to group different items into separate rows.
        minlength (int): Minimum number of classes to count (for each batch).

    Returns:
        torch.Tensor: If `batch` is None, shape is (minlength,).
                      If `batch` is given, shape is (num_batches, minlength).
    """
    if input.ndim != 1:
        raise ValueError("`input` must be a 1D tensor.")

    if batch is None:
        return torch.bincount(input, minlength=minlength)

    if batch.shape != input.shape:
        raise ValueError("`batch` must have the same shape as `input`.")

    max_input_val = input.max().item()  # largest class index
    length = int(max_input_val) + 1

    if minlength == 0:
        minlength = length
    if length > minlength:
        raise ValueError(
            f"minlength={minlength} is too small for input with classes up to {length - 1}."
        )

    # Flatten indexes: shift each "class" by its batch index * minlength
    input_ = input + batch * minlength

    num_batch = batch.max().item() + 1
    # total bins = num_batch * minlength
    out = torch.bincount(input_, minlength=minlength * int(num_batch))

    # Reshape back to (num_batch, minlength)
    out = out.reshape(int(num_batch), minlength)
    return out
