from __future__ import annotations

import copy

import numpy as np
from ase import Atoms
from ase.cluster import wulff_construction
from ase.data import chemical_symbols as CHEMICAL_SYMBOLS
from ase.data import covalent_radii as COVALENT_RADII
from ase.io import write
from ase.neighborlist import NeighborList, natural_cutoffs

from ideal.utils.habor_bosch import _rotate_to_z, get_surface_indices

MAX_TRIAL = 2000
R = 8.314  # J/(mol·K)
T = 298.15  # K


def pos_normalize(system: Atoms, nanoparticle_element: str) -> Atoms:
    symbols = system.get_chemical_symbols()
    particle_indices = [i for i, s in enumerate(symbols) if s == nanoparticle_element]

    positions = np.array(system.get_positions())
    particle_positions = positions[particle_indices]
    cell = np.array(system.get_cell(complete=True))
    center_pos = np.mean(particle_positions, axis=0)
    center_cell = np.sum(cell, axis=0) / 2
    offset = center_cell - center_pos

    positions += offset
    system.set_positions(positions)

    scaled_positions = system.get_scaled_positions()
    scaled_positions = np.mod(scaled_positions, 1)
    system.set_scaled_positions(scaled_positions)
    system.pbc = True
    return system


def generate_bcc_Fe_nanoparticle(
    element: str,
    size: int,
    box_extension: float = 10.0,
) -> Atoms:
    surface_energies = {
        "110": 2.45,
        "100": 2.50,
        "320": 2.56,
        "210": 2.57,
        "211": 2.61,
        "311": 2.63,
        "331": 2.63,
        "321": 2.63,
        "221": 2.66,
        "332": 2.68,
    }
    # Convert surface energies to the required format
    surfaces = [
        (1, 1, 0),
        (1, 0, 0),
        (3, 2, 0),
        (2, 1, 0),
        (2, 1, 1),
        (3, 1, 1),
        (3, 3, 1),
        (3, 2, 1),
        (2, 2, 1),
        (3, 3, 2),
    ]
    energies = [
        surface_energies["110"],
        surface_energies["100"],
        surface_energies["320"],
        surface_energies["210"],
        surface_energies["211"],
        surface_energies["311"],
        surface_energies["331"],
        surface_energies["321"],
        surface_energies["221"],
        surface_energies["332"],
    ]

    # Generate Wulff nanoparticle with explicit lattice constant
    nanoparticle = wulff_construction(
        symbol=element,
        surfaces=surfaces,
        energies=energies,
        size=size,
        structure="bcc",
    )

    # Define the PBC box dimensions
    positions = nanoparticle.get_positions()
    box_size = np.ptp(positions, axis=0) + box_extension
    nanoparticle.set_cell(box_size)
    nanoparticle.set_pbc(True)

    return nanoparticle


def _determine_surface_id(
    system: Atoms,
    atom_index: int,
    surface_indices: list,
    degree_tolerance: list[float] = [0, 3],
):
    if atom_index not in surface_indices:
        return None
    positions = np.array(system.get_positions())
    cell = np.array(system.get_cell(complete=True))
    center = 0.5 * np.sum(cell, axis=0)
    pos_i = np.array(system.get_positions())[atom_index].reshape(1, -1)
    surface_indices_no_i = [i for i in surface_indices if i != atom_index]
    surface_pos = np.array(system.get_positions())[surface_indices_no_i]

    distances = np.linalg.norm(surface_pos - pos_i, axis=1)
    four_nearest_indices = np.argsort(distances)[:4]
    four_nearest_indices = [surface_indices_no_i[i] for i in four_nearest_indices]
    # four nearest atoms a, b, c, d
    vertical_abc = np.cross(
        positions[four_nearest_indices[1]] - positions[four_nearest_indices[0]],
        positions[four_nearest_indices[2]] - positions[four_nearest_indices[0]],
    )
    vertical_abd = np.cross(
        positions[four_nearest_indices[1]] - positions[four_nearest_indices[0]],
        positions[four_nearest_indices[3]] - positions[four_nearest_indices[0]],
    )
    vertical_acd = np.cross(
        positions[four_nearest_indices[2]] - positions[four_nearest_indices[0]],
        positions[four_nearest_indices[3]] - positions[four_nearest_indices[0]],
    )
    vertical_bcd = np.cross(
        positions[four_nearest_indices[2]] - positions[four_nearest_indices[1]],
        positions[four_nearest_indices[3]] - positions[four_nearest_indices[1]],
    )

    center_i_vector = pos_i - center
    if np.dot(center_i_vector, vertical_abc) < 0:
        vertical_abc = -vertical_abc
    if np.dot(center_i_vector, vertical_abd) < 0:
        vertical_abd = -vertical_abd
    if np.dot(center_i_vector, vertical_acd) < 0:
        vertical_acd = -vertical_acd
    if np.dot(center_i_vector, vertical_bcd) < 0:
        vertical_bcd = -vertical_bcd

    if not (
        np.allclose(vertical_abc, vertical_abd)
        and np.allclose(vertical_abc, vertical_acd)
        and np.allclose(vertical_abc, vertical_bcd)
    ):
        return None

    vertical_vector = np.mean(
        [vertical_abc, vertical_abd, vertical_acd, vertical_bcd], axis=0
    ).reshape(3)

    abs_vertical_vector = np.sort(np.abs(vertical_vector))[::-1]

    surface_directions = {
        "110": np.array([1, 1, 0]),
        "100": np.array([1, 0, 0]),
        "320": np.array([3, 2, 0]),
        "210": np.array([2, 1, 0]),
        "211": np.array([2, 1, 1]),
        "311": np.array([3, 1, 1]),
        "331": np.array([3, 3, 1]),
        "321": np.array([3, 2, 1]),
        "221": np.array([2, 2, 1]),
        "332": np.array([3, 3, 2]),
    }

    min_angle = np.inf
    min_surface_id = None
    for surface_id, direction in surface_directions.items():
        cos_angle = np.dot(abs_vertical_vector, direction) / (
            np.linalg.norm(abs_vertical_vector) * np.linalg.norm(direction)
        )
        cos_angle = np.abs(np.clip(cos_angle, -1, 1))
        angle = np.arccos(cos_angle)
        if angle < min_angle:
            min_angle = angle
            min_surface_id = surface_id
    if (min_angle >= np.deg2rad(degree_tolerance[0])) and (
        min_angle <= np.deg2rad(degree_tolerance[1])
    ):
        return min_surface_id
    else:
        return None


def add_promoter(
    system: Atoms,
    surface_indices: list,
    base_element: str,
    promoter_element: str,
    num_promoter: int,
    min_promoter_distance: float,
    surface_id_list: list = [],
):
    possible_promoter_indices = []
    for idx in surface_indices:
        surface_id_i = _determine_surface_id(system, idx, surface_indices)
        if surface_id_i in surface_id_list:
            print(f"Surface ID of atom {idx}: {surface_id_i}")
            possible_promoter_indices.append(idx)

    radii_base = COVALENT_RADII[CHEMICAL_SYMBOLS.index(base_element)]
    radii_promoter = COVALENT_RADII[CHEMICAL_SYMBOLS.index(promoter_element)]
    cell = np.array(system.get_cell(complete=True))
    center = 0.5 * np.sum(cell, axis=0)

    selected_promoter_indices = []
    for i in range(num_promoter):
        trials = 0
        while trials < MAX_TRIAL:
            p_index = np.random.choice(possible_promoter_indices)

            positions = _rotate_to_z(system, p_index)
            symbols = system.get_chemical_symbols()

            new_p_pos = np.array(
                [positions[p_index][0], positions[p_index][1], positions[p_index][2]]
            )
            assert (
                np.allclose(new_p_pos[0], center[0])
                and np.allclose(new_p_pos[1], center[1])
                and new_p_pos[2] > center[2]
            ), f"Error: {new_p_pos}, {center}"
            new_p_pos[2] += radii_base + radii_promoter

            dists = np.linalg.norm(
                positions[selected_promoter_indices] - new_p_pos, axis=1
            )
            if len(dists) == 0 or np.min(dists) > min_promoter_distance:
                selected_promoter_indices.append(len(positions))
                positions = np.vstack([positions, new_p_pos.reshape(1, -1)])
                symbols.append(promoter_element)
                system = Atoms(
                    symbols=symbols,
                    positions=positions,
                    cell=system.get_cell(),
                    pbc=system.pbc,
                )
                break
            trials += 1
    print(
        f"Number of selected promoter atoms: {len(selected_promoter_indices)}/{num_promoter}"
    )

    symbols = system.get_chemical_symbols()
    for idx in selected_promoter_indices:
        symbols[idx] = promoter_element
    system.set_chemical_symbols(symbols)
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
    parser.add_argument("--size", type=int, default=2000)
    parser.add_argument("--element", type=str, default="Fe")
    parser.add_argument("--promoter", type=str, default="Li")
    parser.add_argument("--promoter_ratio", type=float, default=0.05)
    parser.add_argument("--min_promoter_distance", type=float, default=10.0)
    parser.add_argument("--box_extension", type=float, default=15.0)
    parser.add_argument("--surface_id_list", type=list, default=["110", "100"])
    parser.add_argument("--degree_tolerance", type=float, nargs="+", default=[0, 3])
    args = parser.parse_args()

    nanoparticle = generate_bcc_Fe_nanoparticle(
        element=args.element,
        size=args.size,
        box_extension=args.box_extension,
    )
    nanoparticle = pos_normalize(nanoparticle, args.element)
    surface_indices = get_surface_indices(nanoparticle, [args.element], [args.promoter])
    num_promoter = int(len(surface_indices) * args.promoter_ratio)
    nanoparticle = add_promoter(
        system=nanoparticle,
        surface_indices=surface_indices,
        base_element=args.element,
        promoter_element=args.promoter,
        num_promoter=num_promoter,
        min_promoter_distance=args.min_promoter_distance,
        surface_id_list=args.surface_id_list,
    )
    radius = particle_radius_estimate(nanoparticle)
    print(f"Estimated particle radius: {radius}A")

    write(
        f"../../contents/habor-bosch/particles/{args.element}{args.promoter}-{radius}A_surface{'-'.join(args.surface_id_list)}_degree{'-'.join(map(str, args.degree_tolerance))}.xyz",
        nanoparticle,
    )
