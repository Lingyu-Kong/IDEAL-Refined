from __future__ import annotations

import argparse

import torch
from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import read, write
from ase.optimize import BFGS, FIRE2, LBFGS

from ideal.ideal_calculator import IDEALCalculator
from ideal.models.potential import Potential


def init_relax(args_dict: dict):
    torch.cuda.set_device(args_dict["device"])
    potential = Potential.from_checkpoint(
        load_path=args_dict["potential"],
    )
    calculator = IDEALCalculator(
        mode="offline",
        potential=potential,
        device=f"cuda:{args_dict['device']}",
    )

    atoms: Atoms = read(args_dict["file"])  # type: ignore
    atoms.set_calculator(calculator)
    symbols = atoms.get_chemical_symbols()
    NH_indices = [True if s == "N" or s == "H" else False for s in symbols]
    atoms.set_constraint(FixAtoms(indices=NH_indices))
    opt = LBFGS(atoms)
    traj_file = args_dict["file"].split("/")[-1].replace(".xyz", "_relax_traj.xyz")

    def log_traj():
        print("Step: ", opt.get_number_of_steps())
        write(traj_file, atoms, append=True)

    opt.attach(log_traj, interval=20)
    opt.run(fmax=0.02, steps=5000)

    write(
        f"../../contents/habor-bosch/particles/Relaxed_{args_dict['file'].split('/')[-1]}",
        atoms,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=str,
        default="/nethome/lkong88/IDEAL/contents/habor-bosch/particles/FeLi-19.82A_surface110-100_degree0-3.xyz",
    )
    parser.add_argument("--potential", type=str, default="MatterSim-v1.0.0-1M")
    parser.add_argument("--device", type=int, default=7)
    args = parser.parse_args()
    init_relax(vars(args))
