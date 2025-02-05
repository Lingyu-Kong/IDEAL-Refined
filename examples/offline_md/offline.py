from __future__ import annotations

import argparse
import os

import ase.units as units
import numpy as np
import torch
import wandb
from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.optimize import BFGS

from ideal.ideal_calculator import IDEALCalculator
from ideal.models.potential import Potential
from ideal.utils.habor_bosch import compute_coordination


def main(args_dict: dict):
    torch.cuda.set_device(args_dict["cuda_device_idx"])
    potential = Potential.from_checkpoint(
        load_path=args_dict["potential"],
    )
    calculator = IDEALCalculator(
        mode="offline",
        potential=potential,
        device=f"cuda:{args_dict['cuda_device_idx']}",
    )

    atoms: Atoms = read(args_dict["file"])  # type: ignore
    atoms.set_calculator(calculator)

    if args_dict["init_relax"]:
        NH_indices = [
            True if s in ["N", "H"] else False for s in atoms.get_chemical_symbols()
        ]
        atoms.set_constraint(FixAtoms(indices=NH_indices))
        opt = BFGS(atoms)
        opt.run(fmax=0.02, steps=1000)
        atoms.set_constraint()

    Fe_indices = [True if s == "Fe" else False for s in atoms.get_chemical_symbols()]
    if args_dict["fix_kernel"]:
        atoms.set_constraint(FixAtoms(indices=Fe_indices))

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

    if args_dict["wandb"]:
        wandb.login(key="37f3de06380e350727df28b49712f8b7fe5b14aa")
        wandb.init(
            project="Offline MD Test", config=args_dict, name=traj_dir.split("/")[-1]
        )

    def log_traj():
        print(
            f"Step: {dyn.get_number_of_steps()}, Temperature: {atoms.get_temperature()}K"
        )
        print(
            "============================================================================"
        )
        scaled_pos = atoms.get_scaled_positions()
        scaled_pos = np.mod(scaled_pos, 1)
        dyn.atoms.set_scaled_positions(scaled_pos)
        delta_time = int(args_dict["timestep"] * args_dict["saveinterval"] / 1000)
        current_time = int(dyn.get_number_of_steps() * args_dict["timestep"] / 1000)
        current_file = f"{traj_dir}/{current_time}-{current_time + delta_time}ps.xyz"
        write(current_file, atoms, append=True)

        coors = compute_coordination(
            atoms,
            base_element=args_dict["base_element"],
            promoter_element=args_dict["promoter_element"],
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

    dyn.attach(log_traj, interval=args_dict["loginterval"])
    dyn.run(args_dict["md_steps"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subsample a slab")
    parser.add_argument(
        "--file",
        type=str,
        default="/nethome/lkong88/IDEAL/contents/habor-bosch/particles/Relaxed_FeLi-19.82A_surface110-100_H2_793_N2_793.xyz",
    )
    parser.add_argument("--base_element", type=str, default="Fe")
    parser.add_argument("--promoter_element", type=str, default="Li")
    parser.add_argument("--potential", type=str, default="MatterSim-v1.0.0-1M")
    parser.add_argument("--cuda_device_idx", type=int, default=1)
    # MD configuration
    parser.add_argument("--timestep", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1400.0)
    parser.add_argument("--friction", type=float, default=0.1)
    parser.add_argument("--md_steps", type=int, default=200000)
    parser.add_argument("--loginterval", type=int, default=200)
    parser.add_argument("--saveinterval", type=int, default=2000)
    parser.add_argument("--fix_kernel", action="store_true")
    parser.add_argument("--init_relax", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    args_dict = vars(args)

    main(args_dict)
