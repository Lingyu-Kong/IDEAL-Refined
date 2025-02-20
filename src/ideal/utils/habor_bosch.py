"""
Additional tools for the Habor Bosch simulation.
"""

from __future__ import annotations

import ase
import ase.neighborlist as nl
import numpy as np
import torch
from ase import Atoms
from ase.data import chemical_symbols as CHEMICAL_SYMBOLS
from ase.data import covalent_radii


def _rotate_to_z(
    atoms: Atoms,
    index: int,
) -> np.ndarray:
    """
    Rotate the atoms so that the atom at index is aligned to the positive z-axis.
    Return rotated positions array.
    """
    scaled_pos = np.mod(np.array(atoms.get_scaled_positions()), 1)
    cell = np.array(atoms.get_cell(complete=True))
    positions = scaled_pos @ cell
    cell_center_pos = 0.5 * np.sum(cell, axis=0)
    positions = positions - cell_center_pos.reshape(1, 3)
    vec = positions[index]
    z_axis = np.array([0, 0, 1])

    # Check if vec is already aligned with z_axis
    vec_norm = np.linalg.norm(vec)
    if vec_norm < 1e-8 or np.allclose(vec / vec_norm, z_axis):
        return positions + cell_center_pos.reshape(1, 3)

    # Calculate rotation axis and angle
    axis = np.cross(vec, z_axis)
    axis_norm = np.linalg.norm(axis)
    cos_theta = np.clip(np.dot(vec, z_axis) / vec_norm, -1.0, 1.0)
    if axis_norm < 1e-8:
        if np.isclose(cos_theta, -1.0):  # vec is anti-parallel to z_axis
            rotation_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
            rotated_positions = positions @ rotation_matrix.T
            return rotated_positions + cell_center_pos.reshape(1, 3)
        else:
            # vec is already aligned with z_axis
            return positions + cell_center_pos.reshape(1, 3)
    axis = axis / axis_norm
    angle = np.arccos(cos_theta)

    # Construct the rotation matrix using Rodrigues' rotation formula
    K = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    rotation_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)

    # Apply rotation to all positions
    rotated_positions = positions @ rotation_matrix.T

    # Map rotated positions back into the periodic box
    rotated_positions = rotated_positions + cell_center_pos.reshape(1, 3)

    return rotated_positions


def get_surface_indices(
    atoms: Atoms,
    particle_elements: list[str],
    promoter_elements: list[str] = [],
    radius_type: str = "covalent",
):
    """
    Get the indices of the surface atoms.
    Args:
        atoms: the atoms object.
        particle_elements: the elements of the particles.
        gas_elements: the elements of the gas atoms.
        cutoff: cutoff radius for neighbor search.
        radius_type: Type of radius to use ('covalent' or 'atomic').
    Returns:
        list[int]: Indices of surface atoms.
    """
    symbols = atoms.get_chemical_symbols()
    particle_indices = [i for i, s in enumerate(symbols) if s in particle_elements]
    promoter_indices = [i for i, s in enumerate(symbols) if s in promoter_elements]

    # Get radii
    if radius_type == "covalent":
        radii_data = covalent_radii
    else:
        raise ValueError(f"Unsupported radius type: {radius_type}")

    # Calculate diameters for all particle elements
    diameters = {
        element: 2 * radii_data[CHEMICAL_SYMBOLS.index(element)]
        for element in particle_elements
    }
    radii = {
        element: radii_data[CHEMICAL_SYMBOLS.index(element)]
        for element in particle_elements
    }

    # Collect surface atom indices
    surface_indices = promoter_indices

    for i in particle_indices:
        # Rotate the atoms such that atom i aligns with the z-axis
        positions = _rotate_to_z(atoms, i)

        # Get the z position of the current atom
        z_i = positions[i, 2]

        # Check if there's any atom above this atom in +z direction by more than its diameter
        diameter = diameters[symbols[i]]
        radius = radii[symbols[i]]
        higher_atoms = [
            j
            for j, pos in enumerate(positions)
            if (j != i) and (pos[2] > z_i + radius) and symbols[j] in particle_elements
        ]

        # If no atom is higher than this atom in the +z direction, it's a surface atom
        if len(higher_atoms) == 0:
            surface_indices.append(i)

    return surface_indices


def get_sub_surface_indices(
    atoms: Atoms,
    particle_elements: list[str],
    promoter_elements: list[str],
    radius_type: str = "covalent",
):
    symbols = atoms.get_chemical_symbols()
    particle_indices = [i for i, s in enumerate(symbols) if s in particle_elements]
    promoter_indices = [i for i, s in enumerate(symbols) if s in promoter_elements]

    # Get radii
    if radius_type == "covalent":
        radii_data = covalent_radii
    else:
        raise ValueError(f"Unsupported radius type: {radius_type}")

    # Calculate diameters for all particle elements
    diameters = {
        element: 2 * radii_data[CHEMICAL_SYMBOLS.index(element)]
        for element in particle_elements
    }
    radii = {
        element: radii_data[CHEMICAL_SYMBOLS.index(element)]
        for element in particle_elements
    }

    # Collect surface atom indices
    sub_surface_indices = promoter_indices

    for i in particle_indices:
        # Rotate the atoms such that atom i aligns with the z-axis
        positions = _rotate_to_z(atoms, i)

        # Get the z position of the current atom
        z_i = positions[i, 2]

        # Check if there's any atom above this atom in +z direction by more than its diameter
        diameter = diameters[symbols[i]]
        radius = radii[symbols[i]]
        higher_atoms = [
            j
            for j, pos in enumerate(positions)
            if (j != i) and (pos[2] > z_i + radius) and symbols[j] in particle_elements
        ]

        # If only one atom is higher than this atom in the +z direction, it's a sub-surface atom
        if len(higher_atoms) <= 4:
            sub_surface_indices.append(i)

    return sub_surface_indices


def get_non_surface_indices(
    atoms: Atoms,
    particle_elements: list[str],
    promoter_elements: list[str],
    radius_type: str = "covalent",
):
    kernel_elements = list(set(particle_elements).union(set(promoter_elements)))
    symbols = atoms.get_chemical_symbols()
    kernel_indices = [i for i, s in enumerate(symbols) if s in kernel_elements]
    surface_indices = get_surface_indices(
        atoms, particle_elements, promoter_elements, radius_type
    )
    non_surface_indices = [i for i in kernel_indices if i not in surface_indices]
    return non_surface_indices


# def get_surface_indices(
#     atoms: Atoms,
#     particle_elements: list[str],
#     gas_elements: list[str],
#     cutoff: float,
# ):
#     """
#     Get the indices of the surface atoms.
#     Args:
#         atoms: the atoms object
#         particle_elements: the elements of the particles
#         gas_elements: the elements of the gas atoms
#     """
#     scaled_pos = atoms.get_scaled_positions()
#     scaled_pos = np.mod(scaled_pos, 1)
#     atoms.set_scaled_positions(scaled_pos)
#     atoms.pbc = True

#     symbols = atoms.get_chemical_symbols()
#     assert set(particle_elements + gas_elements) == set(symbols)
#     assert set(particle_elements) & set(gas_elements) == set()

#     src_indices, dst_indices = nl.neighbor_list(
#         "ij",
#         atoms,
#         cutoff,
#         self_interaction=True,
#     )
#     src_indices, dst_indices = np.array(src_indices), np.array(dst_indices)

#     ## A surface local env should be centered around a particle atom and contains at least one gas atom in the cutoff
#     surface_indices = []
#     for i in range(len(atoms)):
#         if symbols[i] in particle_elements:
#             # Get the indices of neighbors of atom i
#             neighbor_indices = dst_indices[src_indices == i]
#             # Check if any neighbor's chemical symbol is in gas_elements
#             if any(symbols[j] in gas_elements for j in neighbor_indices):
#                 surface_indices.append(i)

#     return surface_indices


def get_gas_indices(
    atoms: Atoms,
    particle_elements: list[str],
    gas_elements: list[str],
    cutoff: float,
):
    """
    Get the indices of the surface atoms.
    Args:
        atoms: the atoms object
        particle_elements: the elements of the particles
        gas_elements: the elements of the gas atoms
    """
    scaled_pos = atoms.get_scaled_positions()
    scaled_pos = np.mod(scaled_pos, 1)
    atoms.set_scaled_positions(scaled_pos)
    atoms.pbc = True

    symbols = atoms.get_chemical_symbols()
    assert set(particle_elements + gas_elements) == set(symbols)
    assert set(particle_elements) & set(gas_elements) == set()

    src_indices, dst_indices = nl.neighbor_list(
        "ij",
        atoms,
        cutoff,
        self_interaction=True,
    )
    src_indices, dst_indices = np.array(src_indices), np.array(dst_indices)

    ## A surface local env should be centered around a particle atom and contains at least one gas atom in the cutoff
    gas_indices = []
    for i in range(len(atoms)):
        if symbols[i] in gas_elements:
            # Get the indices of neighbors of atom i
            neighbor_indices = dst_indices[src_indices == i]
            # Check if any neighbor's chemical symbol is in gas_elements
            if not any(symbols[j] in particle_elements for j in neighbor_indices):
                gas_indices.append(i)

    return gas_indices


import math

import torch


def compute_coordination(atoms, base_element: str, promoter_element: str):
    positions = torch.as_tensor(atoms.get_positions(), dtype=torch.float32)
    atomic_numbers = torch.as_tensor(atoms.get_atomic_numbers(), dtype=torch.int64)
    box = torch.as_tensor(atoms.get_cell().lengths(), dtype=torch.float32)

    # 构造对角线形式的盒子张量(只适用于正交晶胞)
    cell = torch.diag(box)

    # 构建周期性平移向量 (立方体 3D 网格, -1, 0, 1)
    mesh_grid = (
        torch.stack(
            torch.meshgrid(
                torch.arange(-1, 2),
                torch.arange(-1, 2),
                torch.arange(-1, 2),
                indexing="ij",
            ),
            dim=-1,
        )
        .view([-1, 1, 3])
        .float()
    )
    shift_vectors = torch.matmul(mesh_grid, cell)

    # positions.shape: (n_atoms, 3)
    # shift_vectors.shape: (27, 1, 3)
    # positions + shift_vectors.shape: (27, n_atoms, 3)
    # dist_mat: (n_atoms, 27*n_atoms) 在 cdist 后再取 min 即可得到包含 PBC 的最小距离
    dist_mat = torch.cdist(positions, positions + shift_vectors).min(dim=0).values
    # dist_mat 结果为 (n_atoms, n_atoms): 每个原子对 (i, j) 的最小距离(考虑 PBC)

    # 建立原子 mask
    N_mask = atomic_numbers == 7
    H_mask = atomic_numbers == 1
    base_mask = atomic_numbers == CHEMICAL_SYMBOLS.index(base_element)
    promoter_mask = atomic_numbers == CHEMICAL_SYMBOLS.index(promoter_element)

    n_atoms = len(atomic_numbers)

    # 用 ~torch.eye(...) 来排除 i==j 的自距离
    not_self = ~torch.eye(n_atoms, dtype=torch.bool)

    NH_mask = N_mask.unsqueeze(1) & H_mask.unsqueeze(0) & not_self
    baseN_mask = base_mask.unsqueeze(1) & N_mask.unsqueeze(0) & not_self
    baseH_mask = base_mask.unsqueeze(1) & H_mask.unsqueeze(0) & not_self
    promoterN_mask = promoter_mask.unsqueeze(1) & N_mask.unsqueeze(0) & not_self
    promoterH_mask = promoter_mask.unsqueeze(1) & H_mask.unsqueeze(0) & not_self
    NN_mask = N_mask.unsqueeze(1) & N_mask.unsqueeze(0) & not_self
    HH_mask = H_mask.unsqueeze(1) & H_mask.unsqueeze(0) & not_self

    # 定义合理的截断半径 (单位 Å)，以下仅为示例值
    bond_cutoffs = {
        "NN": 1.5,
        "HH": 0.9,
        "NH": 1.3,
        "FeN": 2.2,
        "FeH": 1.8,
        "RuN": 2.2,
        "RuH": 1.8,
        "KN": 3.2,
        "KH": 2.8,
        "CsN": 3.5,
        "CsH": 3.2,
        "LiN": 2.5,
        "LiH": 2.2,
    }

    # 统计各个成键数
    # dist_mat < cutoff 会返回一个 (n_atoms, n_atoms) 的布尔矩阵，再根据 mask 筛选
    # 然后 sum(dim=1) 后，对于属于该元素的原子再聚合求和
    NN_coor = torch.sum(
        torch.sum((dist_mat < bond_cutoffs["NN"]) & NN_mask, dim=1)[N_mask].int()
    ).item()
    HH_coor = torch.sum(
        torch.sum((dist_mat < bond_cutoffs["HH"]) & HH_mask, dim=1)[H_mask].int()
    ).item()
    NH_coor = torch.sum(
        torch.sum((dist_mat < bond_cutoffs["NH"]) & NH_mask, dim=1)[N_mask].int()
    ).item()
    NH_coor = 0
    NH2_coor = 0
    NH3_coor = 0
    for i in range(n_atoms):
        if N_mask[i]:
            NH_num = torch.sum((dist_mat[i] < bond_cutoffs["NH"]) & H_mask).item()
            if NH_num == 1:
                NH_coor += 1
            elif NH_num == 2:
                NH2_coor += 1
            elif NH_num == 3:
                NH3_coor += 1
    baseN_coor = torch.sum(
        torch.sum((dist_mat < bond_cutoffs[base_element + "N"]) & baseN_mask, dim=1)[
            base_mask
        ].int()
    ).item()
    baseH_coor = torch.sum(
        torch.sum((dist_mat < bond_cutoffs[base_element + "H"]) & baseH_mask, dim=1)[
            base_mask
        ].int()
    ).item()
    promoterN_coor = torch.sum(
        torch.sum(
            (dist_mat < bond_cutoffs[promoter_element + "N"]) & promoterN_mask, dim=1
        )[promoter_mask].int()
    ).item()
    promoterH_coor = torch.sum(
        torch.sum(
            (dist_mat < bond_cutoffs[promoter_element + "H"]) & promoterH_mask, dim=1
        )[promoter_mask].int()
    ).item()

    return {
        "NN": NN_coor,
        "HH": HH_coor,
        "NH": NH_coor,
        "NH2": NH2_coor,
        "NH3": NH3_coor,
        base_element + "N": baseN_coor,
        base_element + "H": baseH_coor,
        promoter_element + "N": promoterN_coor,
        promoter_element + "H": promoterH_coor,
    }
