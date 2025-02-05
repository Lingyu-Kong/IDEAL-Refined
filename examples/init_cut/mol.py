from __future__ import annotations

import argparse
import os

from ase import Atoms
from ase.io import read, write

from ideal.subsampler.cut_strategy import CSCECutStrategyConfig
from ideal.subsampler.sampler import SubSampler
from ideal.subsampler.sub_optimizer import UncGradientOptimizerConfig
from ideal.subsampler.threshold import (
    PercentileUncThresholdConfig,
    ValueUncThresholdConfig,
)
from ideal.subsampler.unc_module import KernelCoreIncremental
from ideal.subsampler.unc_module.unc_gk import SoapCompressConfig, SoapGKConfig
from ideal.utils.habor_bosch import get_surface_indices


def main(args_dict: dict):
    # Load the structure
    atoms: Atoms = read(args_dict["file"])  # type: ignore
    species = set(atoms.get_chemical_symbols())
    symbols = atoms.get_chemical_symbols()
    mol_indices = [i for i, s in enumerate(symbols) if s in ["N", "H"]]

    # Define the subsampling strategy
    unc_model = SoapGKConfig(
        soap_cutoff=args_dict["soap_cutoff"],
        max_l=args_dict["max_l"],
        max_n=args_dict["max_n"],
        species=list(species),
        soap_compress=SoapCompressConfig(
            mode=args_dict["soap_compress"], species_weighting=None
        ),
        kernel_cls=KernelCoreIncremental,
        soap_normalization=args_dict["soap_normalization"],
        supercell_size=args_dict["soap_supercell_size"],
        n_jobs=args_dict["n_jobs"],
    )
    # unc_model.update(atoms=atoms, include_indices=list(range(len(atoms))))
    sub_optimizer = UncGradientOptimizerConfig(
        max_steps=args_dict["sub_opt_max_steps"],
        lr=args_dict["sub_opt_lr"],
        optimizer=args_dict["sub_opt_optimizer"],
        scheduler=args_dict["sub_opt_scheduler"],
        early_stop=args_dict["sub_opt_early_stop"],
        early_stop_patience=args_dict["sub_opt_early_stop_patience"],
        noise=args_dict["sub_opt_noise"],
        noise_decay=args_dict["sub_opt_noise_decay"],
    )

    cut_strategy = CSCECutStrategyConfig(
        sub_cutoff=args_dict["sub_cutoff"],
        cell_extend_max=args_dict["cell_extend_max"],
        cut_scan_granularity=args_dict["cut_scan_granularity"],
        num_process=args_dict["num_process"],
        max_num_rs=args_dict["max_num_rs"],
    )
    sub_sampler = SubSampler(
        species=list(species),
        unc_model=unc_model,
        unc_threshold_config=ValueUncThresholdConfig(
            window_size=args_dict["unc_threshold_window_size"],
            sigma=args_dict["unc_threshold_sigma"],
        ),
        sub_optimizer=sub_optimizer,
        cut_strategy=cut_strategy,
    )
    sub_sampler.update_unc_model_and_threshold(
        atoms=atoms, include_indices=list(range(len(atoms)))
    )

    # Sample substrucutres
    subs = sub_sampler.sample_all(
        atoms=atoms,
        include_indices=mol_indices,
        max_samples=50,
    )
    write("./FeLi_mol.xyz", subs, format="extxyz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subsample a slab")
    parser.add_argument(
        "--file",
        type=str,
        default="/nethome/lkong88/IDEAL/contents/habor-bosch/particles/Relaxed_FeLi-19.82A_surface110-100_H2_793_N2_793.xyz",
    )
    # Uncertainty model configuration
    parser.add_argument("--soap_cutoff", type=float, default=2.0)
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
    parser.add_argument("--cell_extend_max", type=float, default=1.5)
    parser.add_argument("--cut_scan_granularity", type=float, default=0.1)
    parser.add_argument("--num_process", type=int, default=16)
    parser.add_argument("--max_num_rs", type=int, default=8000)
    ## Uncertainty threshold configuration
    parser.add_argument("--unc_threshold_method", type=str, default="percentile")
    parser.add_argument("--unc_threshold_window_size", type=int, default=5000)
    parser.add_argument("--unc_threshold_alpha", type=float, default=0.5)
    parser.add_argument("--unc_threshold_k", type=float, default=2.0)
    parser.add_argument("--unc_threshold_sigma", type=float, default=60.0)
    args = parser.parse_args()
    args_dict = vars(args)
    main(args_dict)
