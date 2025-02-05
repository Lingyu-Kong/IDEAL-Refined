from __future__ import annotations

import copy
import random

import numpy as np
from ase import Atoms
from ase.build import molecule
from ase.io import read, write
from ase.neighborlist import neighbor_list
from tqdm import tqdm

MAX_TRIAL = 2000


def convert_density_to_num_molecules(
    system: Atoms, density: float, nanoparticle_element: str
):
    symbols = system.get_chemical_symbols()
    positions = np.array(system.get_positions())
    cell = np.array(system.get_cell(complete=True))
    particle_indices = [i for i, s in enumerate(symbols) if s == nanoparticle_element]
    particle_positions = positions[particle_indices]
    center = np.mean(particle_positions, axis=0).reshape(1, 3)
    dist_from_center = np.linalg.norm(particle_positions - center, axis=1)
    approximate_radius = np.max(dist_from_center)

    box_volume = np.linalg.det(cell)
    particle_volume = 4 / 3 * np.pi * approximate_radius**3
    gas_volume = box_volume - particle_volume

    num_moles = density * gas_volume

    return int(num_moles)


def add_molecules(
    system: Atoms,
    molecule_type: str,
    num_molecules: int,
    mol_min_distance: float,
    radius_estimate: float,
) -> Atoms:
    """
    Add molecules (H2 or N2) to the PBC box without overlapping with existing atoms.
    """
    if molecule_type == "H2":
        molecule_template = molecule("H2")
    elif molecule_type == "N2":
        molecule_template = molecule("N2")
    else:
        raise ValueError("Unsupported molecule type. Choose 'H2' or 'N2'.")

    box_size = np.array(system.get_cell(complete=True)).diagonal()
    cell = np.array(system.get_cell(complete=True))
    cell_center = 0.5 * np.sum(cell, axis=0)

    pbar = tqdm(total=num_molecules, desc=f"Adding {molecule_type} molecules")
    for _ in range(num_molecules):
        trial_count = 0
        while trial_count < MAX_TRIAL:
            trial_count += 1
            molecule_copy = copy.deepcopy(molecule_template)
            ## 添加随机旋转
            molecule_copy.rotate(random.random() * 360, "z")
            molecule_copy.rotate(random.random() * 360, "y")
            molecule_copy.rotate(random.random() * 360, "x")
            # find a random position in the box but outside radius_estimate from the center
            position = np.random.rand(3) * box_size
            while np.linalg.norm(position - cell_center) < radius_estimate:
                position = np.random.rand(3) * box_size
            molecule_copy.translate(position)

            new_system_copy = copy.deepcopy(system)
            new_system_copy.pbc = True
            new_system_copy += molecule_copy
            new_indices = list(range(len(system), len(new_system_copy)))

            src_indices, dst_indices, dist = neighbor_list(
                "ijd", new_system_copy, cutoff=mol_min_distance, self_interaction=False
            )
            add_success = True
            for src, dst, d in zip(src_indices, dst_indices, dist):
                if src in new_indices:
                    if d < mol_min_distance:
                        if dst not in new_indices:
                            add_success = False
                            break
            if add_success:
                existing_positions = np.array(system.get_positions())
                existing_symbols = system.get_chemical_symbols()
                molecule_positions = np.array(molecule_copy.get_positions())
                molecule_symbols = molecule_copy.get_chemical_symbols()
                positions = np.concatenate([existing_positions, molecule_positions])
                symbols = existing_symbols + molecule_symbols
                system = Atoms(
                    symbols=symbols,
                    positions=positions,
                    cell=system.get_cell(),
                    pbc=True,
                )
                # system += molecule_copy
                pbar.update(1)
                break
    pbar.close()
    return system


def particle_radius_estimate(
    system: Atoms,
):
    positions = np.array(system.get_positions())
    cell = np.array(system.get_cell(complete=True))

    center = 0.5 * np.sum(cell, axis=0)
    distances = np.linalg.norm(positions - center, axis=1)
    radius = np.max(distances).round(2).item()
    return radius


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--particle_file",
        type=str,
        default="/nethome/lkong88/IDEAL/contents/habor-bosch/particles/Relaxed_FeLi-19.82A_surface110-100.xyz",
    )
    parser.add_argument("--H2_density", type=float, default=0.00402 * 2)
    parser.add_argument("--N2_density", type=float, default=0.00402 * 2)
    parser.add_argument("--mol_min_distance", type=float, default=2.5)
    args = parser.parse_args()

    system: Atoms = read(args.particle_file)  # type: ignore
    radius_estimate = particle_radius_estimate(system)
    print(f"Estimated radius of the nanoparticle: {radius_estimate:.2f}A")

    num_H2 = convert_density_to_num_molecules(
        system, args.H2_density, nanoparticle_element="Fe"
    )
    num_N2 = convert_density_to_num_molecules(
        system, args.N2_density, nanoparticle_element="Fe"
    )
    print(f"Adding {num_H2:.2e} H2 molecules.")
    print(f"Adding {num_N2:.2e} N2 molecules.")

    system = add_molecules(system, "N2", num_N2, args.mol_min_distance, radius_estimate)
    system = add_molecules(system, "H2", num_H2, args.mol_min_distance, radius_estimate)

    final_H2 = [s for s in system.get_chemical_symbols() if s == "H"]
    final_H2 = int(len(final_H2) // 2)
    final_N2 = [s for s in system.get_chemical_symbols() if s == "N"]
    final_N2 = int(len(final_N2) // 2)

    src_indices, dst_indices, d = neighbor_list(
        "ijd", system, cutoff=2.5, self_interaction=False
    )
    symbols = system.get_chemical_symbols()
    H_indices = [i for i, s in enumerate(symbols) if s == "H"]
    N_indices = [i for i, s in enumerate(symbols) if s == "N"]
    Fe_indices = [i for i, s in enumerate(symbols) if s == "Fe"]
    Li_indices = [i for i, s in enumerate(symbols) if s == "Li"]
    HH_d = d[
        np.logical_and(np.isin(src_indices, H_indices), np.isin(dst_indices, H_indices))
    ]
    HN_d = d[
        np.logical_and(np.isin(src_indices, H_indices), np.isin(dst_indices, N_indices))
    ]
    HFe_d = d[
        np.logical_and(
            np.isin(src_indices, H_indices), np.isin(dst_indices, Fe_indices)
        )
    ]
    HLi_d = d[
        np.logical_and(
            np.isin(src_indices, H_indices), np.isin(dst_indices, Li_indices)
        )
    ]
    NN_d = d[
        np.logical_and(np.isin(src_indices, N_indices), np.isin(dst_indices, N_indices))
    ]
    NFe_d = d[
        np.logical_and(
            np.isin(src_indices, N_indices), np.isin(dst_indices, Fe_indices)
        )
    ]
    NLi_d = d[
        np.logical_and(
            np.isin(src_indices, N_indices), np.isin(dst_indices, Li_indices)
        )
    ]
    FeFe_d = d[
        np.logical_and(
            np.isin(src_indices, Fe_indices), np.isin(dst_indices, Fe_indices)
        )
    ]
    FeLi_d = d[
        np.logical_and(
            np.isin(src_indices, Fe_indices), np.isin(dst_indices, Li_indices)
        )
    ]
    LiLi_d = d[
        np.logical_and(
            np.isin(src_indices, Li_indices), np.isin(dst_indices, Li_indices)
        )
    ]
    if len(HH_d) > 0:
        print(f"min HH distance: {np.min(HH_d):.2f}")
    if len(HN_d) > 0:
        print(f"min HN distance: {np.min(HN_d):.2f}")
    if len(HFe_d) > 0:
        print(f"min HFe distance: {np.min(HFe_d):.2f}")
    if len(HLi_d) > 0:
        print(f"min HLi distance: {np.min(HLi_d):.2f}")
    if len(NN_d) > 0:
        print(f"min NN distance: {np.min(NN_d):.2f}")
    if len(NFe_d) > 0:
        print(f"min NFe distance: {np.min(NFe_d):.2f}")
    if len(NLi_d) > 0:
        print(f"min NLi distance: {np.min(NLi_d):.2f}")
    if len(FeFe_d) > 0:
        print(f"min FeFe distance: {np.min(FeFe_d):.2f}")
    if len(FeLi_d) > 0:
        print(f"min FeLi distance: {np.min(FeLi_d):.2f}")
    if len(LiLi_d) > 0:
        print(f"min LiLi distance: {np.min(LiLi_d):.2f}")

    filename = args.particle_file.split("/")[-1].replace(
        ".xyz", f"_H2_{final_H2}_N2_{final_N2}.xyz"
    )
    write(f"../../contents/habor-bosch/particles/{filename}", system)
    print(f"Saved to {filename}")
