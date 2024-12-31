"""
Additional tools for the Habor Bosch simulation.
"""

from __future__ import annotations

import ase
import ase.neighborlist as nl
import numpy as np
from ase import Atoms
from ase.data import chemical_symbols, covalent_radii


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
            rotation_matrix = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
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

    # Get radii
    if radius_type == "covalent":
        radii_data = covalent_radii
    else:
        raise ValueError(f"Unsupported radius type: {radius_type}")

    # Calculate diameters for all particle elements
    diameters = {
        element: 2 * radii_data[chemical_symbols.index(element)]
        for element in particle_elements
    }
    radii = {
        element: radii_data[chemical_symbols.index(element)]
        for element in particle_elements
    }

    # Collect surface atom indices
    surface_indices = []

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
