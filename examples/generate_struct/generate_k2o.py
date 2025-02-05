from __future__ import annotations

import copy
from itertools import combinations

import numpy as np
from ase import Atoms
from ase.cluster import wulff_construction
from ase.io import read, write

from ideal.utils.habor_bosch import get_surface_indices

MAX_TRIAL = 2000


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


def _rotation_matrix_z_to_target(target_vector: np.ndarray) -> np.ndarray:
    target_vector = np.array(target_vector, dtype=float)
    target_vector /= np.linalg.norm(target_vector)
    z_axis = np.array([0, 0, 1])
    rotation_axis = np.cross(z_axis, target_vector)
    rotation_axis_norm = np.linalg.norm(rotation_axis)
    if rotation_axis_norm < 1e-8:
        if np.allclose(target_vector, z_axis):
            matrix = np.eye(3)
        else:
            matrix = np.diag([1, 1, -1])
    else:
        rotation_axis /= rotation_axis_norm
        cos_theta = np.dot(z_axis, target_vector)
        sin_theta = np.sqrt(1 - cos_theta**2)
        K = np.array(
            [
                [0, -rotation_axis[2], rotation_axis[1]],
                [rotation_axis[2], 0, -rotation_axis[0]],
                [-rotation_axis[1], rotation_axis[0], 0],
            ]
        )
        matrix = np.eye(3) + sin_theta * K + (1 - cos_theta) * np.dot(K, K)
    assert np.allclose(np.dot(matrix, z_axis), target_vector)
    return matrix


def add_k2o(
    system: Atoms,
    base_element: str,
    num_promoters: int,
    min_promoter_distance: float,
    surface_id_list: list[str],
    degree_tolerance: list[float] = [0, 3],
):
    surface_indices = get_surface_indices(system, [base_element])
    selected_indices = []
    for idx in surface_indices:
        surface_id = _determine_surface_id(
            system, idx, surface_indices, degree_tolerance
        )
        if surface_id in surface_id_list:
            selected_indices.append(idx)
    positions = np.array(system.get_positions())[selected_indices]
    distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    if base_element == "Fe":
        a = 2.48549
        b = 2.48549
        c = 2.86999
    else:
        raise NotImplementedError
    triangles = []
    for i, j, k in combinations(range(len(selected_indices)), 3):
        if i == j or j == k or k == i:
            continue
        if set([i, j, k]) in triangles:
            continue
        d1 = distances[i, j]
        d2 = distances[j, k]
        d3 = distances[k, i]
        sides = sorted([d1, d2, d3])
        if np.allclose(sides, [a, b, c]):
            triangles.append(set([i, j, k]))
    possible_promoter_positions = []
    for triangle in triangles:
        pos1 = positions[list(triangle)[0]]
        pos2 = positions[list(triangle)[1]]
        pos3 = positions[list(triangle)[2]]
        center = np.mean([pos1, pos2, pos3], axis=0).reshape(-1)
        possible_promoter_positions.append(center)
    possible_promoter_positions = np.array(possible_promoter_positions)
    if len(possible_promoter_positions) < num_promoters:
        print("Not enough possible promoter positions")
        num_promoters = len(possible_promoter_positions)

    k2o_template = np.array([[0, 0, 0], [1.895, 0, 1.895], [-1.895, 0, 1.895]])

    # select positions to insert promoters
    selected_promoter_positions = []
    for _ in range(num_promoters):
        trials = 0
        while trials < MAX_TRIAL:
            p_idx = np.random.randint(len(possible_promoter_positions))
            ppos = possible_promoter_positions[p_idx]
            if len(selected_promoter_positions) == 0:
                selected_promoter_positions.append(ppos)
                break
            distances = np.linalg.norm(
                np.array(selected_promoter_positions) - ppos, axis=1
            )
            if np.all(distances > min_promoter_distance):
                selected_promoter_positions.append(ppos)
                break
            trials += 1
    print(
        f"Selected {len(selected_promoter_positions)}/{num_promoters} positions to intert K2O promoters"
    )
    # insert promoters
    positions = np.array(system.get_positions())
    symbols = system.get_chemical_symbols()
    cell = np.array(system.get_cell(complete=True))
    center = 0.5 * np.sum(cell, axis=0)
    for ppos in selected_promoter_positions:
        center_p_vec = ppos - center
        rotation_matrix = _rotation_matrix_z_to_target(center_p_vec)
        k2o = np.matmul(k2o_template, rotation_matrix.T) + ppos
        positions = np.vstack([positions, k2o])
        symbols.extend(["O", "K", "K"])
    new_atoms = Atoms(
        symbols=symbols,
        positions=positions,
        cell=cell,
        pbc=system.pbc,
    )
    return new_atoms


def add_k2o_bulk(
    system: Atoms,
    base_element: str,
    num_promoters: int,
    min_promoter_distance: float,
    surface_id_list: list[str],
    degree_tolerance: list[float] = [0, 3],
):
    surface_indices = get_surface_indices(system, [base_element])
    selected_indices = []
    for idx in surface_indices:
        surface_id = _determine_surface_id(
            system, idx, surface_indices, degree_tolerance
        )
        if surface_id in surface_id_list:
            selected_indices.append(idx)
    positions = np.array(system.get_positions())[selected_indices]
    distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    if base_element == "Fe":
        a = 2.48549
        b = 2.48549
        c = 2.86999
    else:
        raise NotImplementedError
    triangles = []
    for i, j, k in combinations(range(len(selected_indices)), 3):
        if i == j or j == k or k == i:
            continue
        if set([i, j, k]) in triangles:
            continue
        d1 = distances[i, j]
        d2 = distances[j, k]
        d3 = distances[k, i]
        sides = sorted([d1, d2, d3])
        if np.allclose(sides, [a, b, c]):
            triangles.append(set([i, j, k]))
    possible_promoter_positions = []
    for triangle in triangles:
        pos1 = positions[list(triangle)[0]]
        pos2 = positions[list(triangle)[1]]
        pos3 = positions[list(triangle)[2]]
        center = np.mean([pos1, pos2, pos3], axis=0).reshape(-1)
        possible_promoter_positions.append(center)
    possible_promoter_positions = np.array(possible_promoter_positions)
    if len(possible_promoter_positions) < num_promoters:
        print("Not enough possible promoter positions")
        num_promoters = len(possible_promoter_positions)

    selected_promoter_positions = []
    for _ in range(num_promoters):
        trials = 0
        while trials < MAX_TRIAL:
            p_idx = np.random.randint(len(possible_promoter_positions))
            ppos = possible_promoter_positions[p_idx]
            if len(selected_promoter_positions) == 0:
                selected_promoter_positions.append(ppos)
                break
            distances = np.linalg.norm(
                np.array(selected_promoter_positions) - ppos, axis=1
            )
            if np.all(distances > min_promoter_distance):
                selected_promoter_positions.append(ppos)
                break
            trials += 1
    print(
        f"Selected {len(selected_promoter_positions)}/{num_promoters} positions to intert K2O promoters"
    )

    k2o_bulk_template: Atoms = read("./K2O.cif")  # type: ignore
    k2o_scaled_pos = np.array(k2o_bulk_template.get_scaled_positions())
    k2o_pos = np.array(k2o_bulk_template.get_positions())
    k2o_symbols = k2o_bulk_template.get_chemical_symbols()
    k2o_cell = np.array(k2o_bulk_template.get_cell(complete=True))
    idx = 0
    length = len(k2o_scaled_pos)
    while idx < length:
        spos = k2o_scaled_pos[idx]
        for j in range(3):
            if spos[j] == 0:
                pspos = np.array([spos[0], spos[1], spos[2]])
                pspos[j] = 1
                find_same = False
                for k in range(length):
                    if np.allclose(k2o_scaled_pos[k], pspos):
                        find_same = True
                        break
                if not find_same:
                    k2o_scaled_pos = np.vstack([k2o_scaled_pos, pspos])
                    k2o_symbols.append(k2o_symbols[idx])
        idx += 1
        length = len(k2o_scaled_pos)
    k2o_pos = np.matmul(k2o_scaled_pos, k2o_cell)

    positions = np.array(system.get_positions())
    symbols = system.get_chemical_symbols()
    cell = np.array(system.get_cell(complete=True))
    center = 0.5 * np.sum(cell, axis=0)
    for ppos in selected_promoter_positions:
        center_p_vec = ppos - center
        rotation_matrix = _rotation_matrix_z_to_target(center_p_vec)
        k2o_pos_i = np.matmul(k2o_pos, rotation_matrix.T) + ppos
        positions = np.vstack([positions, k2o_pos_i])
        symbols.extend(k2o_symbols)
    new_atoms = Atoms(
        symbols=symbols,
        positions=positions,
        cell=cell,
        pbc=system.pbc,
    )
    return new_atoms


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--element", type=str, default="Fe")
    parser.add_argument("--size", type=int, default=2000)
    parser.add_argument("--box-extension", type=float, default=15.0)
    parser.add_argument("--k_ratio", type=float, default=0.04)
    parser.add_argument("--min-promoter-distance", type=float, default=15.0)
    parser.add_argument("--surface_id_list", type=str, default="110")
    parser.add_argument("--degree_tolerance", type=float, nargs="+", default=[0, 3])
    parser.add_argument("--export_surface_test", action="store_true")
    args = parser.parse_args()

    nanoparticle = generate_bcc_Fe_nanoparticle(
        element=args.element,
        size=args.size,
        box_extension=args.box_extension,
    )
    nanoparticle = pos_normalize(nanoparticle, args.element)
    surface_indices = get_surface_indices(nanoparticle, [args.element])
    num_k2o_promoters = int(args.k_ratio * len(surface_indices)) // 2
    surface_id_list = args.surface_id_list.split(",")

    # export a structure with only selected surface atoms
    if args.export_surface_test:
        selected_indices = []
        for idx in surface_indices:
            surface_id = _determine_surface_id(
                nanoparticle, idx, surface_indices, args.degree_tolerance
            )
            if surface_id in surface_id_list:
                selected_indices.append(idx)
        positions = np.array(nanoparticle.get_positions())[selected_indices]
        symbols = [args.element] * len(selected_indices)
        selected_surface = Atoms(
            symbols=symbols,
            positions=positions,
            cell=nanoparticle.get_cell(),
            pbc=nanoparticle.pbc,
        )
        write("selected_surface.res", selected_surface)
        exit()

    nanoparticle = add_k2o(
        nanoparticle,
        args.element,
        num_k2o_promoters,
        args.min_promoter_distance,
        surface_id_list,
        args.degree_tolerance,
    )

    def particle_radius_estimate(
        system: Atoms,
    ):
        positions = np.array(system.get_positions())
        cell = np.array(system.get_cell(complete=True))

        center = 0.5 * np.sum(cell, axis=0)
        distances = np.linalg.norm(positions - center, axis=1)
        radius = np.max(distances).round(2).item()
        return radius

    radius = particle_radius_estimate(nanoparticle)
    print(f"Estimated particle radius: {radius}A")

    write(
        f"../../contents/habor-bosch/particles/{args.element}K2O-{radius}A_surface{'-'.join(surface_id_list)}.xyz",
        nanoparticle,
    )
