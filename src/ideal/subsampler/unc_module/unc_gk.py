"""
Using mahalanobis distance/gaussian kernel to calculate the uncertainty

unc = (x - mu)^T * (K+sigma*I)^-1 * (x - mu)
"""

from __future__ import annotations

import numpy as np
import torch
from ase import Atoms
from ase.build import make_supercell
from dscribe.descriptors import SOAP
from pydantic import PositiveFloat, PositiveInt
from typing_extensions import TypedDict, cast, override

from ._base import UncModuleBase, UncModuleConfigBase
from .kernel_core import KernelCoreBase, KernelCoreIncremental


class SoapCompressConfig(TypedDict):
    """
    Configuration for compressing the soap features
    """

    mode: str
    species_weighting: None | dict[str, float]


class SoapGKConfig(UncModuleConfigBase):
    """
    Configuration for SoapGK
    """

    soap_cutoff: PositiveFloat
    max_l: PositiveInt = 4
    max_n: PositiveInt = 4
    species: list[str]
    soap_compress: dict | SoapCompressConfig | None = None
    kernel_cls: type[KernelCoreBase] = KernelCoreIncremental
    soap_normalization: str | None = "l2"
    supercell_size: PositiveInt = 3
    n_jobs: PositiveInt = 1

    @override
    def create_unc_module(self) -> SoapGK:
        if isinstance(self.soap_compress, dict):
            compress_method = cast(SoapCompressConfig, self.soap_compress)
        else:
            compress_method = self.soap_compress
        return SoapGK(
            soap_cutoff=self.soap_cutoff,
            max_l=self.max_l,
            max_n=self.max_n,
            species=self.species,
            compress_method=compress_method,
            kernel_cls=self.kernel_cls,
            soap_normalization=self.soap_normalization,
            supercell_size=self.supercell_size,
            n_jobs=self.n_jobs,
        )


class SoapGK(UncModuleBase):
    """
    Compute the uncertainty using Gaussian Kernel
    The local_env is encoded using SOAP
    """

    def __init__(
        self,
        soap_cutoff: float,
        max_l: int,
        max_n: int,
        species: list[str],
        compress_method: None | SoapCompressConfig = None,
        kernel_cls: type[KernelCoreBase] = KernelCoreIncremental,
        soap_normalization: str | None = "l2",
        supercell_size: int = 3,
        n_jobs: int = 1,
    ):
        """
        Args:
            soap_cutoff: the cutoff radius of the soap
            max_l: the maximum l of the soap
            max_n: the maximum n of the soap
            species: the species of the atoms
            n_jobs: the number of jobs to run in parallel
        """
        self.soap = SOAP(
            species=species,
            periodic=False,
            r_cut=soap_cutoff,
            n_max=max_n,
            l_max=max_l,
            rbf="gto",
            compression=dict(compress_method)
            if compress_method
            else {"mode": "off", "species_weighting": None},
            dtype="float32",
        )
        self.feature_dim = self.soap.get_number_of_features()
        print(f"Soap feature_dim: {self.feature_dim}")
        self.core_dict = {s: kernel_cls(self.feature_dim) for s in species}
        self.soap_normalization = soap_normalization
        assert supercell_size % 2 == 1 and supercell_size > 1, (
            "supercell_size must be odd and greater than 1"
        )
        self.supercell_size = supercell_size
        self.n_jobs = n_jobs

    def _normalize_atoms(self, atoms: Atoms) -> Atoms:
        """
        Normalize the atoms
        """
        scaled_pos = np.array(atoms.get_scaled_positions())
        # center_pos = np.mean(scaled_pos, axis=0)
        # scaled_pos = scaled_pos + 0.5 - center_pos.reshape(1, -1)
        scaled_pos = np.mod(scaled_pos, 1)
        atoms.set_scaled_positions(scaled_pos)
        return atoms

    def _normalize_soap(self, soap_features: np.ndarray) -> np.ndarray:
        """
        Normalize the soap features
        """
        if self.soap_normalization == "l2":
            soap_features = soap_features / np.linalg.norm(
                soap_features, axis=1, keepdims=True
            )
        else:
            raise NotImplementedError(
                f"soap normalization {self.soap_normalization} is not implemented for now"
            )
        return soap_features

    @override
    def update(
        self,
        *,
        atoms: Atoms,
        include_indices: list[int],
    ):
        atoms = self._normalize_atoms(atoms)
        supercell = make_supercell(
            atoms, np.eye(3) * self.supercell_size, order="cell-major"
        )
        center_unit_indices = np.arange(
            (self.supercell_size**3 // 2) * len(atoms),
            (self.supercell_size**3 // 2 + 1) * len(atoms),
        ).tolist()
        soap_features = self.soap.create(
            supercell, centers=center_unit_indices, n_jobs=self.n_jobs
        )
        soap_features = np.array(soap_features)  # type: ignore
        soap_features = self._normalize_soap(soap_features)
        symbols = atoms.get_chemical_symbols()
        for idx in include_indices:
            self.core_dict[symbols[idx]].step_kernel(soap_features[idx])
        symbols = set([symbols[i] for i in include_indices])
        for s in symbols:
            self.core_dict[s].update_kernel()

    def get_soap_features(
        self,
        atoms: Atoms,
    ) -> np.ndarray:
        """
        Get the soap features of the atoms
        """
        atoms = self._normalize_atoms(atoms)
        supercell = make_supercell(
            atoms, np.eye(3) * self.supercell_size, order="cell-major"
        )
        center_unit_indices = np.arange(
            (self.supercell_size**3 // 2) * len(atoms),
            (self.supercell_size**3 // 2 + 1) * len(atoms),
        ).tolist()
        soap_features = self.soap.create(
            supercell, centers=center_unit_indices, n_jobs=self.n_jobs
        )
        soap_features = np.array(soap_features)[center_unit_indices]
        return self._normalize_soap(soap_features)

    @override
    def unc_predict(
        self,
        *,
        atoms: Atoms,
        include_grad: bool = False,
        unc_include_indices: list[int] | None = None,
        grad_include_indices: list[int] | None = None,
        reduction: bool = True,
    ):
        """
        Predict the uncertainty of the atoms
        """
        if unc_include_indices is None:
            unc_include_indices = list(range(len(atoms)))
        if grad_include_indices is None:
            grad_include_indices = list(range(len(atoms)))

        atoms = self._normalize_atoms(atoms)
        supercell = make_supercell(
            atoms, np.eye(3) * self.supercell_size, order="cell-major"
        )
        center_unit_indices = np.arange(
            (self.supercell_size**3 // 2) * len(atoms),
            (self.supercell_size**3 // 2 + 1) * len(atoms),
        ).tolist()
        super_unc_include_indices = [
            center_unit_indices[i] for i in unc_include_indices
        ]
        super_grad_include_indices = [
            center_unit_indices[i] for i in grad_include_indices
        ]
        if not include_grad:
            soap_features = self.soap.create(
                supercell, centers=super_unc_include_indices, n_jobs=self.n_jobs
            )
            soap_features = np.array(soap_features)
        else:
            derivatives, soap_features = self.soap.derivatives(
                supercell,
                centers=super_unc_include_indices,
                include=super_grad_include_indices,
                return_descriptor=True,
                n_jobs=self.n_jobs,
            )
            derivatives = np.array(derivatives)
            soap_features = np.array(soap_features)
        soap_features = torch.from_numpy(soap_features).float()
        uncs = torch.zeros(
            len(
                unc_include_indices,
            ),
            dtype=torch.float32,
        )
        unc_soap_grad = torch.zeros((len(unc_include_indices), self.feature_dim))
        symbols = atoms.get_chemical_symbols()
        symbols = [symbols[i] for i in unc_include_indices]
        for i in range(len(unc_include_indices)):
            mu, kernel_inv = self.core_dict[symbols[i]].get_kernel()
            soap_i = soap_features[i]
            if include_grad:
                soap_i.requires_grad = True
            soap_i_normed = soap_i / torch.norm(soap_i)
            soap_i_prime = (
                soap_i_normed - torch.from_numpy(mu).to(torch.float32)
            ).reshape(1, -1)
            unc_i = torch.mm(
                soap_i_prime, torch.from_numpy(kernel_inv).to(torch.float32)
            ).mm(soap_i_prime.t())
            if include_grad:
                unc_i.backward()
                unc_soap_grad[i] = soap_i.grad.reshape(-1)  ## type: ignore
            uncs[i] = unc_i
        unc = torch.mean(uncs)
        if include_grad:
            derivatives = np.moveaxis(
                derivatives, source=[0, 1, 2, 3], destination=[0, 2, 3, 1]
            )
            unc_soap_grad = unc_soap_grad.detach().numpy()
            ## [len(unc_include_indices), soap_feature_dim] * [len(unc_include_indices), soap_feature_dim, len(grad_include_indices), 3] -> [len(grad_include_indices), 3]
            _grad = np.einsum(
                "ij,ijkl->kl",
                unc_soap_grad,
                derivatives,  ## type: ignore
            )
        else:
            _grad = np.zeros((len(grad_include_indices), 3))
        grad = np.zeros((len(atoms), 3))
        grad[grad_include_indices] = _grad
        if reduction:
            return float(unc.item()), grad
        else:
            return uncs.detach().numpy(), grad
