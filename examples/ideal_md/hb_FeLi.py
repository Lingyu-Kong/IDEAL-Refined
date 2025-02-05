from __future__ import annotations

import argparse
import copy
import json
import os
import rich

import ase.units as units
import numpy as np
import torch
import wandb
from ase import Atoms
from ase.io import read, write
from ase.md.langevin import Langevin
from typing_extensions import cast

from ideal.abinitio_interfaces.vasp import VaspFakeInterfaceConfig, VaspInterfaceConfig
from ideal.ideal_calculator import (
    IDEALCalculator,
    IDEALModelTuningConfig,
    ImportanceSamplingConfig,
)
from ideal.models.potential import Potential
from ideal.subsampler.cut_strategy import (
    CSCECutStrategyCataystConfig,
    CSCECutStrategyConfig,
)
from ideal.subsampler.sampler import SubSamplerConfig
from ideal.subsampler.sub_optimizer import UncGradientOptimizerConfig
from ideal.subsampler.threshold import (
    PercentileUncThresholdConfig,
    ValueUncThresholdConfig,
)
from ideal.subsampler.unc_module import KernelCoreIncremental
from ideal.subsampler.unc_module.unc_gk import SoapCompressConfig, SoapGKConfig
from ideal.utils.habor_bosch import compute_coordination, get_surface_indices


def main(args_dict: dict):
    def get_calculator():
        # Load the structure
        atoms: Atoms = read(args_dict["file"])  # type: ignore
        species = set(atoms.get_chemical_symbols())
        surface_indices = get_surface_indices(
            atoms, particle_elements=list(species - set(["H", "N"]))
        )

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
        if args_dict["catalyst"]:
            cut_strategy = CSCECutStrategyCataystConfig(
                sub_cutoff=args_dict["sub_cutoff"],
                cell_extend_max=args_dict["cell_extend_max"],
                cell_extend_zpos_min=args_dict["cell_extend_zpos_min"],
                cell_extend_zpos_max=args_dict["cell_extend_zpos_max"],
                cut_scan_granularity=args_dict["cut_scan_granularity"],
                num_process=args_dict["num_process"],
                max_num_rs=args_dict["max_num_rs"],
            )
        else:
            cut_strategy = CSCECutStrategyConfig(
                sub_cutoff=args_dict["sub_cutoff"],
                cell_extend_max=args_dict["cell_extend_max"],
                cell_extend_zpos_min=args_dict["cell_extend_zpos_min"],
                cell_extend_zpos_max=args_dict["cell_extend_zpos_max"],
                cut_scan_granularity=args_dict["cut_scan_granularity"],
                num_process=args_dict["num_process"],
                max_num_rs=args_dict["max_num_rs"],
            )
        if args_dict["unc_threshold_method"] == "percentile":
            unc_threshold_config = PercentileUncThresholdConfig(
                window_size=args_dict["unc_threshold_window_size"],
                alpha=args_dict["unc_threshold_alpha"],
                k=args_dict["unc_threshold_k"],
            )
        elif args_dict["unc_threshold_method"] == "value":
            unc_threshold_config = ValueUncThresholdConfig(
                window_size=args_dict["unc_threshold_window_size"],
                sigma=args_dict["unc_threshold_sigma"],
            )
        else:
            raise ValueError("Invalid uncertainty threshold method")
        sub_sampler = SubSamplerConfig(
            species=list(species),
            unc_model=unc_model,
            unc_threshold_config=unc_threshold_config,
            sub_optimizer=sub_optimizer,
            cut_strategy=cut_strategy,
        )

        # Construct the IDEAL calculator
        custom_setting_json = args_dict["vasp_custom_setting_json"]
        if custom_setting_json is not None:
            with open(custom_setting_json, "r") as f:
                custom_setting = json.load(f)
        else:
            custom_setting = None
        abinitio_interface = VaspFakeInterfaceConfig(
            user_incar_settings={
                "IBRION": -1,  # 不进行离子弛豫
                "ENCUT": 520,  # 能量截止
                "EDIFF": 1e-3,  # 电子收敛标准
                "ISMEAR": 0,  # MP展开
                "ALGO": "Normal",  # 电子算法
                "SIGMA": 0.05,  # 展宽
                "Accuracy": "Normal",
                "PREC": "Normal",  # 计算精度
                "ISPIN": 2,  # 开启自旋极化
                "LCHARGE": False,  # 不输出 CHGCAR
                "ISYM": -1,  # 关闭对称性
                "LREAL": "Auto",  # 自动选择是否使用实空间表示（兼顾效率和精度）
                "NSW": 1,  # 不进行离子弛豫
                "LORBIT": 0,  # 不输出轨道信息
                "MAGMOM": {
                    "Fe": 4.0,
                    "K": 0.0,
                    "N": 0.0,
                    "H": 0.0,
                },
                "KPOINTS": "Gamma",  # 仍使用 Gamma 定心网格
            }
            if custom_setting is None
            else custom_setting,
            user_kpoints_settings={
                "reciprocal_density": 1.0,
                "kpoints_scheme": "Gamma",  # 仍使用 Gamma 定心网格
            },
            vasp_cmd=f"mpirun -np {args_dict['vasp_npar']} vasp_gam | tee vasp.log",
        )
        potential = Potential.from_checkpoint(
            load_path=args_dict["potential"],
        )
        calculator = IDEALCalculator(
            potential=potential,
            sub_sampler=sub_sampler,
            max_samples=args_dict["max_ideal_samples"],
            abinitio_interface=abinitio_interface,
            include_indices=surface_indices,
            compute_stress=False,
            precision=args_dict["model_precision"],
            wandb_log=args_dict["wandb"],
            importance_sampling=cast(
                ImportanceSamplingConfig,
                {
                    "temperature": args_dict["IS_temperature"],
                    "temperature_decay": args_dict["IS_temperature_decay"],
                    "min_temperature": args_dict["IS_temperature_min"],
                    "key": args_dict["IS_key"],
                },
            ),
            model_tuning_config=cast(
                IDEALModelTuningConfig,
                {
                    "max_epochs": args_dict["model_max_epochs"],
                    "batch_size": args_dict["model_batch_size"],
                    "learning_rate": args_dict["model_lr"],
                    "optimizer": args_dict["model_optimizer"],
                    "scheduler": args_dict["model_scheduler"],
                    "loss": torch.nn.MSELoss(),
                    "include_energy": True,
                    "include_forces": True,
                    "include_stresses": False,
                    "force_loss_ratio": 10.0,
                    "stress_loss_ratio": 0.0,
                },
            ),
        )

        initialize_atoms_list: list[Atoms] = read(args_dict["initialize_dataset"], ":") # type: ignore
        initialize_atoms_list_single_point: list[Atoms] = read(args_dict["initialize_dataset_single_point"], ":") # type: ignore
        calculator.initialize(
            model_initial_atoms_list=initialize_atoms_list,
            unc_initial_atoms_list=initialize_atoms_list_single_point,
        )

        return calculator

    atoms: Atoms = read(args_dict["file"])  # type: ignore
    base_element = args_dict["file"].split("/")[-1].split("-")[0]
    base_element = "".join([i for i in base_element if not i.isdigit()])
    promoter_element = args_dict["file"].split("/")[-1].split("-")[1]
    promoter_element = "".join([i for i in promoter_element if not i.isdigit()])
    calc = get_calculator()
    atoms.set_calculator(calc)

    # Run the MD simulation
    dyn = Langevin(
        atoms,
        temperature_K=args_dict["temperature"],
        timestep=args_dict["timestep"] * units.fs,
        friction=args_dict["friction"],
        fixcm=True,
    )
    
    traj_dir = f"./md_results/{args_dict['temperature']}K_" + args_dict["file"].split(
        "/"
    )[-1].replace(".xyz", "")
    if args_dict["fix_kernel"]:
        traj_dir = traj_dir + "_fixkernel"

    if os.path.exists(traj_dir):
        os.system(f"rm -r {traj_dir}")
    os.makedirs(traj_dir, exist_ok=True)
    sub_file = f"{traj_dir}/subsampling.xyz"

    def log_traj():
        print(
            f"Step: {dyn.get_number_of_steps()}, Temperature: {atoms.get_temperature()}K"
        )
        print(
            "============================================================================"
        )
        scaled_pos = dyn.atoms.get_scaled_positions()
        scaled_pos = np.mod(scaled_pos, 1)
        dyn.atoms.set_scaled_positions(scaled_pos)
        
        delta_time = int(args_dict["timestep"] * args_dict["saveinterval"] / 1000)
        current_time = int(dyn.get_number_of_steps() * args_dict["timestep"] / 1000)
        current_file = f"{traj_dir}/{current_time}-{current_time + delta_time}ps.xyz"
        write(current_file, atoms, append=True)
        
        coors = compute_coordination(
            atoms, 
            base_element=base_element, 
            promoter_element=promoter_element
        )
        coors_log = {"Coor/" + k: v for k, v in coors.items()}
        if args_dict["wandb"]:
            wandb.log(
                {
                    "Temperature": atoms.get_temperature(),
                    "MD Time (fs)": dyn.get_number_of_steps() * args_dict["timestep"],
                    **coors_log,
                }
            )

        if (dyn.get_number_of_steps() * args_dict["timestep"] + args_dict["loginterval"]) % args_dict["saveinterval"] == 0:
            calc.export_dataset(filename=sub_file)
            wandb.save(sub_file)
            wandb.save(traj_dir)

    dyn.attach(log_traj, interval=args_dict["loginterval"])

    if args_dict["wandb"]:
        wandb.login(key="37f3de06380e350727df28b49712f8b7fe5b14aa")
        wandb.init(project="IDEAL-Habor-Bosch", config=args_dict, name=traj_dir.split("/")[-1])
    rich.print(args_dict)
    dyn.run(args_dict["md_steps"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subsample a slab")
    parser.add_argument(
        "--file",
        type=str,
        default="../../contents/habor-bosch/particles/Relaxed_FeLi-19.82A_surface110-100_H2_793_N2_793.xyz",
    )
    # Uncertainty model configuration
    parser.add_argument("--soap_cutoff", type=float, default=3.5)
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
    parser.add_argument("--sub_opt_early_stop_patience", type=int, default=50)
    parser.add_argument("--sub_opt_noise", type=float, default=0.0)
    parser.add_argument("--sub_opt_noise_decay", type=float, default=1.0)
    parser.add_argument("--sub_opt_grad_clip", type=float, default=1.0)
    parser.add_argument("--max_ideal_samples", type=int, default=20)
    # Cut strategy configuration
    parser.add_argument("--sub_cutoff", type=float, default=4.5)
    parser.add_argument("--cell_extend_max", type=float, default=1.5)
    parser.add_argument("--cell_extend_zpos_min", type=float, default=0.0)
    parser.add_argument("--cell_extend_zpos_max", type=float, default=3.0)
    parser.add_argument("--cut_scan_granularity", type=float, default=0.5)
    parser.add_argument("--num_process", type=int, default=16)
    parser.add_argument("--max_num_rs", type=int, default=1500)
    ## Uncertainty threshold configuration
    parser.add_argument("--unc_threshold_method", type=str, default="percentile")
    parser.add_argument("--unc_threshold_window_size", type=int, default=5000)
    parser.add_argument("--unc_threshold_alpha", type=float, default=0.5)
    parser.add_argument("--unc_threshold_k", type=float, default=2.0)
    parser.add_argument("--unc_threshold_sigma", type=float, default=60.0)
    # VASP configuration
    parser.add_argument("--vasp_npar", type=int, default=16)
    # Model configuration
    parser.add_argument("--potential", type=str, default="MatterSim-v1.0.0-1M")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--IS_temperature", type=float, default=2.0)
    parser.add_argument("--IS_temperature_decay", type=float, default=0.05)
    parser.add_argument("--IS_temperature_min", type=float, default=0.1)
    parser.add_argument("--IS_key", type=str, default="f_mae")
    parser.add_argument("--model_max_epochs", type=int, default=60)
    parser.add_argument("--model_batch_size", type=int, default=32)
    parser.add_argument("--model_lr", type=float, default=1e-4)
    parser.add_argument("--model_optimizer", type=str, default="adam")
    parser.add_argument("--model_scheduler", type=str, default="steplr")
    parser.add_argument("--model_precision", type=str, default="fp32")
    # MD configuration
    parser.add_argument("--timestep", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1400.0)
    parser.add_argument("--friction", type=float, default=0.1)
    parser.add_argument("--md_steps", type=int, default=10000)
    parser.add_argument("--loginterval", type=int, default=2)
    parser.add_argument("--saveinterval", type=int, default=2000)
    ## Initialize Dataset
    parser.add_argument(
        "--initialize_dataset",
        type=str,
        default="../../contents/habor-bosch/init_FeLi_single_point.xyz",
    )
    parser.add_argument(
        "--initialize_dataset_single_point",
        type=str,
        default="../../contents/habor-bosch/init_FeLi_single_point.xyz",
    )
    ## Wandb
    parser.add_argument("--wandb", action="store_true")
    ## Whether use catalyst substructure cutting (extend the cell on z-axis)
    parser.add_argument("--catalyst", action="store_true")
    ## Vasp custom settings in json format
    parser.add_argument("--vasp_custom_setting_json", type=str, default=None)
    parser.add_argument("--fix_kernel", action="store_true")
    args = parser.parse_args()
    args_dict = vars(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args_dict["device"])

    main(args_dict)
