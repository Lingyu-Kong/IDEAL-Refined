from __future__ import annotations

import argparse
import os

from ase import Atoms
from ase.io import read, write

from ideal.subsampler.cut_strategy import CSCECutStrategy
from ideal.subsampler.sampler import SubSampler, UncThresholdConfig
from ideal.subsampler.sub_optimizer import UncGradientOptimizer
from ideal.subsampler.unc_module import KernelCoreIncremental
from ideal.subsampler.unc_module.unc_gk import SoapCompressConfig, SoapGK
from ideal.utils.habor_bosch import get_gas_indices


def main(args_dict: dict):
    # Load the structure
    atoms: Atoms = read(args_dict["file"])  # type: ignore
    species = set(atoms.get_chemical_symbols())
    gas_indices = get_gas_indices(
        atoms,
        particle_elements=list(species - set(["H", "N"])),
        gas_elements=["H", "N"],
        cutoff=args_dict["sub_cutoff"] + args_dict["cell_extend_max"],
    )

    # Define the subsampling strategy
    unc_model = SoapGK(
        soap_cutoff=args_dict["soap_cutoff"],
        max_l=args_dict["max_l"],
        max_n=args_dict["max_n"],
        species=list(species),
        compress_method=SoapCompressConfig(
            mode=args_dict["soap_compress"], species_weighting=None
        ),
        kernel_cls=KernelCoreIncremental,
        soap_normalization=args_dict["soap_normalization"],
        supercell_size=args_dict["soap_supercell_size"],
        n_jobs=args_dict["n_jobs"],
    )
    unc_model.update(atoms=atoms, include_indices=list(range(len(atoms))))
    sub_optimizer = UncGradientOptimizer(
        max_steps=args_dict["sub_opt_max_steps"],
        lr=args_dict["sub_opt_lr"],
        optimizer=args_dict["sub_opt_optimizer"],
        scheduler=args_dict["sub_opt_scheduler"],
        early_stop=args_dict["sub_opt_early_stop"],
        early_stop_patience=args_dict["sub_opt_early_stop_patience"],
        noise=args_dict["sub_opt_noise"],
        noise_decay=args_dict["sub_opt_noise_decay"],
    )

    cut_strategy = CSCECutStrategy(
        method="ideal",
        sub_cutoff=args_dict["sub_cutoff"],
        cell_extend_max=args_dict["cell_extend_max"],
        cut_scan_granularity=args_dict["cut_scan_granularity"],
        num_process=args_dict["num_process"],
        max_num_rs=args_dict["max_num_rs"],
    )
    sub_sampler = SubSampler(
        species=list(species),
        unc_model=unc_model,
        unc_threshold_config=UncThresholdConfig(),
        sub_optimizer=sub_optimizer,
        cut_strategy=cut_strategy,
    )

    # Sample substrucutres
    subs = sub_sampler.sample_all(
        atoms=atoms,
        include_indices=gas_indices,
        max_samples=20,
    )
    write("./NHmol.xyz", subs, format="extxyz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subsample a slab")
    parser.add_argument(
        "--file",
        type=str,
        default="../../contents/habor-bosch/Fe939-K50-semi_embedded-326atmH2-326atmN2.xyz",
    )
    # Uncertainty model configuration
    parser.add_argument("--soap_cutoff", type=float, default=3.0)
    parser.add_argument("--max_l", type=int, default=4)
    parser.add_argument("--max_n", type=int, default=4)
    parser.add_argument("--soap_compress", type=str, default="mu2")
    parser.add_argument("--soap_normalization", type=str, default="l2")
    parser.add_argument("--soap_supercell_size", type=int, default=3)
    parser.add_argument("--n_jobs", type=int, default=1)
    # Subsampling optimizer configuration
    parser.add_argument("--sub_opt_max_steps", type=int, default=200)
    parser.add_argument("--sub_opt_lr", type=float, default=0.1)
    parser.add_argument("--sub_opt_optimizer", type=str, default="adam")
    parser.add_argument("--sub_opt_scheduler", type=str, default="cosine")
    parser.add_argument("--sub_opt_early_stop", type=bool, default=True)
    parser.add_argument("--sub_opt_early_stop_patience", type=int, default=100)
    parser.add_argument("--sub_opt_noise", type=float, default=0.0)
    parser.add_argument("--sub_opt_noise_decay", type=float, default=1.0)
    parser.add_argument("--sub_opt_grad_clip", type=float, default=1.0)
    # Cut strategy configuration
    parser.add_argument("--sub_cutoff", type=float, default=3.0)
    parser.add_argument("--cell_extend_max", type=float, default=1.0)
    parser.add_argument("--cut_scan_granularity", type=float, default=0.1)
    parser.add_argument("--num_process", type=int, default=16)
    parser.add_argument("--max_num_rs", type=int, default=2000)
    args = parser.parse_args()
    args_dict = vars(args)
    main(args_dict)
