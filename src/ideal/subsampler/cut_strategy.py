# -*- coding: utf-8 -*-
"""
Cut Strategy for obtaining substructure
"""

from __future__ import annotations

import copy
import multiprocessing as mp
import time

import ase.io
import ase.neighborlist
import numpy as np
from ase import Atoms
from pydantic import BaseModel, NonNegativeFloat, PositiveFloat, PositiveInt
from typing_extensions import override

from ideal.utils.habor_bosch import _rotate_to_z

from .unc_module import UncModuleBase

mp.set_start_method("spawn", force=True)


def _project_to_vertical_plane(v1, v2):
    """
    The projection of v1 to the vertical plane of v2
    """
    return v1 - np.dot(v1, v2) / np.dot(v2, v2) * v2


def _get_edge_indices(atoms: Atoms, sub_cutoff: float):
    sent_indices, receive_indices = ase.neighborlist.neighbor_list(
        "ij", atoms, sub_cutoff, self_interaction=True
    )
    return np.array([sent_indices, receive_indices])


def _naive_cut(
    atoms: Atoms,
    center_index: int,
    edge_indices: np.ndarray,
    move2center: bool = True,
):
    """
    Crop off a sphere with a radius of sub_cutoff centered at the center atom
    Place the sphere in the center of parent cell
    """
    scaled_pos = np.array(atoms.get_scaled_positions())
    cell = np.array(atoms.get_cell(complete=True))
    chemical_symbols = atoms.get_chemical_symbols()
    sent_indices, receive_indices = edge_indices[0], edge_indices[1]
    sub_indices = set(receive_indices[sent_indices == center_index])
    sub_indices.discard(center_index)
    sub_indices = [center_index] + list(sub_indices)
    sub_scaled_pos = scaled_pos[sub_indices]
    sub_symbols = [chemical_symbols[i] for i in sub_indices]
    sub = Atoms(scaled_positions=sub_scaled_pos, cell=cell, symbols=sub_symbols)
    sub.info["center_index_in_parent"] = center_index
    sub.info["center_index"] = 0
    sub.info["sub_indices_in_parent"] = np.array(
        sub_indices if len(sub_indices) > 1 else []
    )
    if move2center:
        scaled_pos = np.array(sub.get_scaled_positions())
        scaled_pos = np.mod((scaled_pos - scaled_pos[0].reshape(1, 3) + 0.5), 1)
        sub.set_scaled_positions(scaled_pos)
    return sub


def _single_shrink_cell_and_cell_extend(
    sub: Atoms,
    sub_cutoff: float,
    center_index: int,
    centered_parent_pos: np.ndarray,
    parent_symbols: list[str],
    selected_cell_extend_xneg: float,
    selected_cell_extend_xpos: float,
    selected_cell_extend_yneg: float,
    selected_cell_extend_ypos: float,
    selected_cell_extend_zneg: float,
    selected_cell_extend_zpos: float,
):
    """
    Shrink the cell of the substructure
    Args:
        sub: the substructure.
        sub_cutoff: the cutoff of the substructure.
        center_index: the index of the center atom.
        centered_parent_pos: the positions of the parent atoms centered at the center atom.
        selected_cell_extend_xneg: the selected shrink extend in the x axis negative direction.
        selected_cell_extend_xpos: the selected shrink extend in the x axis positive direction.
        selected_cell_extend_yneg: the selected shrink extend in the y axis negative direction.
        selected_cell_extend_ypos: the selected shrink extend in the y axis positive direction.
        selected_cell_extend_zneg: the selected shrink extend in the z axis negative direction.
        selected_cell_extend_zpos: the selected shrink extend in the z axis positive direction.
    Returns:
        sub: the substructure with shrinked cell.
    """
    ## Shrink the cell of the substructure
    sub_cell = np.array(sub.get_cell(complete=True))
    sub_cell_x_axis = sub_cell[0]
    sub_cell_y_axis = sub_cell[1]
    sub_cell_z_axis = sub_cell[2]
    extended_x_length = min(
        2 * sub_cutoff + selected_cell_extend_xneg + selected_cell_extend_xpos,
        np.linalg.norm(sub_cell_x_axis),
    )
    extended_y_length = min(
        2 * sub_cutoff + selected_cell_extend_yneg + selected_cell_extend_ypos,
        np.linalg.norm(sub_cell_y_axis),
    )
    extended_z_length = min(
        2 * sub_cutoff + selected_cell_extend_zneg + selected_cell_extend_zpos,
        np.linalg.norm(sub_cell_z_axis),
    )
    beta_x = max(
        extended_x_length
        / np.linalg.norm(_project_to_vertical_plane(sub_cell_x_axis, sub_cell_y_axis)),
        extended_x_length
        / np.linalg.norm(_project_to_vertical_plane(sub_cell_x_axis, sub_cell_z_axis)),
    )
    beta_y = max(
        extended_y_length
        / np.linalg.norm(_project_to_vertical_plane(sub_cell_y_axis, sub_cell_x_axis)),
        extended_y_length
        / np.linalg.norm(_project_to_vertical_plane(sub_cell_y_axis, sub_cell_z_axis)),
    )
    beta_z = max(
        extended_z_length
        / np.linalg.norm(_project_to_vertical_plane(sub_cell_z_axis, sub_cell_x_axis)),
        extended_z_length
        / np.linalg.norm(_project_to_vertical_plane(sub_cell_z_axis, sub_cell_y_axis)),
    )
    sub_cell[0] = sub_cell_x_axis * beta_x
    sub_cell[1] = sub_cell_y_axis * beta_y
    sub_cell[2] = sub_cell_z_axis * beta_z
    sub_cell_center_pos = (
        (sub_cutoff + selected_cell_extend_xneg) / extended_x_length * sub_cell[0]
    )
    sub_cell_center_pos += (
        (sub_cutoff + selected_cell_extend_yneg) / extended_y_length * sub_cell[1]
    )
    sub_cell_center_pos += (
        (sub_cutoff + selected_cell_extend_zneg) / extended_z_length * sub_cell[2]
    )
    sub_pos = np.array(sub.get_positions())
    sub_pos = sub_pos - sub_pos[0].reshape(1, 3) + sub_cell_center_pos.reshape(1, 3)
    sub.set_positions(sub_pos)
    sub.set_cell(sub_cell)

    ## Include atoms in the extended cell
    parent_pos = centered_parent_pos + sub_cell_center_pos.reshape(1, 3)
    sub_cell_inv = np.linalg.inv(sub_cell)
    local_coords = parent_pos @ sub_cell_inv
    in_box = np.all((local_coords >= 0) & (local_coords < 1), axis=1)
    sub_indices = set(np.where(in_box)[0].tolist())

    sub_indices = sub_indices | set(sub.info["sub_indices_in_parent"])
    sub_indices.discard(center_index)
    sub_indices = [center_index] + list(sub_indices)
    sub_pos = parent_pos[sub_indices]
    sub_symbols = [parent_symbols[i] for i in sub_indices]
    extended_sub = Atoms(positions=sub_pos, cell=sub_cell, symbols=sub_symbols)
    _extended_new_atom_indices = list(
        set(sub_indices) - set(sub.info["sub_indices_in_parent"])
    )
    extended_new_atom_indices = [
        sub_indices.index(i) for i in _extended_new_atom_indices
    ]
    extended_sub.info["center_index_in_parent"] = center_index
    extended_sub.info["center_index"] = 0
    extended_sub.info["movable_indices"] = (
        extended_new_atom_indices if len(extended_new_atom_indices) > 0 else []
    )
    extended_sub.info["sub_indices_in_parent"] = np.array(
        sub_indices if len(sub_indices) > 1 else []
    )

    return extended_sub


def _try_all_cell_extends(
    sub,
    centered_parent_pos,
    parent_symbols,
    cutoff,
    center_index,
    all_cell_extends,
    unc_model: UncModuleBase,
):
    assert np.all(centered_parent_pos[center_index] == 0)
    sub_i_list = []
    unc_list = []
    for j in range(all_cell_extends.shape[0]):
        sub_i = _single_shrink_cell_and_cell_extend(
            sub=sub,
            sub_cutoff=cutoff,
            center_index=center_index,
            centered_parent_pos=centered_parent_pos,
            parent_symbols=parent_symbols,
            selected_cell_extend_xneg=all_cell_extends[j, 0],
            selected_cell_extend_xpos=all_cell_extends[j, 1],
            selected_cell_extend_yneg=all_cell_extends[j, 2],
            selected_cell_extend_ypos=all_cell_extends[j, 3],
            selected_cell_extend_zneg=all_cell_extends[j, 4],
            selected_cell_extend_zpos=all_cell_extends[j, 5],
        )
        unc, _ = unc_model.unc_predict(
            atoms=sub_i,
            include_grad=False,
        )
        sub_i_list.append(sub_i)
        unc_list.append(unc)
    return sub_i_list, unc_list


class CSCECutStrategyConfig(BaseModel):
    sub_cutoff: PositiveFloat
    cell_extend_min: NonNegativeFloat = 0.0
    cell_extend_max: NonNegativeFloat | None = None
    cell_extend_xneg_min: NonNegativeFloat | None = None
    cell_extend_xneg_max: NonNegativeFloat | None = None
    cell_extend_xpos_min: NonNegativeFloat | None = None
    cell_extend_xpos_max: NonNegativeFloat | None = None
    cell_extend_yneg_min: NonNegativeFloat | None = None
    cell_extend_yneg_max: NonNegativeFloat | None = None
    cell_extend_ypos_min: NonNegativeFloat | None = None
    cell_extend_ypos_max: NonNegativeFloat | None = None
    cell_extend_zneg_min: NonNegativeFloat | None = None
    cell_extend_zneg_max: NonNegativeFloat | None = None
    cell_extend_zpos_min: NonNegativeFloat | None = None
    cell_extend_zpos_max: NonNegativeFloat | None = None
    cut_scan_granularity: NonNegativeFloat = 0.1
    num_process: PositiveInt = 1
    max_num_rs: PositiveInt = 1000

    def create_cut_strategy(self) -> CSCECutStrategy:
        return CSCECutStrategy(
            method="ideal",
            sub_cutoff=self.sub_cutoff,
            cell_extend_min=self.cell_extend_min,
            cell_extend_max=self.cell_extend_max,
            cell_extend_xneg_min=self.cell_extend_xneg_min,
            cell_extend_xneg_max=self.cell_extend_xneg_max,
            cell_extend_xpos_min=self.cell_extend_xpos_min,
            cell_extend_xpos_max=self.cell_extend_xpos_max,
            cell_extend_yneg_min=self.cell_extend_yneg_min,
            cell_extend_yneg_max=self.cell_extend_yneg_max,
            cell_extend_ypos_min=self.cell_extend_ypos_min,
            cell_extend_ypos_max=self.cell_extend_ypos_max,
            cell_extend_zneg_min=self.cell_extend_zneg_min,
            cell_extend_zneg_max=self.cell_extend_zneg_max,
            cell_extend_zpos_min=self.cell_extend_zpos_min,
            cell_extend_zpos_max=self.cell_extend_zpos_max,
            cut_scan_granularity=self.cut_scan_granularity,
            num_process=self.num_process,
            max_num_rs=self.max_num_rs,
        )


class CSCECutStrategyCataystConfig(CSCECutStrategyConfig):
    slab_interval: PositiveFloat = 15.0

    @override
    def create_cut_strategy(self) -> CSCECutStrategy:
        return CSCECutStrategyCatalyst(
            method="ideal",
            sub_cutoff=self.sub_cutoff,
            cell_extend_min=self.cell_extend_min,
            cell_extend_max=self.cell_extend_max,
            cell_extend_xneg_min=self.cell_extend_xneg_min,
            cell_extend_xneg_max=self.cell_extend_xneg_max,
            cell_extend_xpos_min=self.cell_extend_xpos_min,
            cell_extend_xpos_max=self.cell_extend_xpos_max,
            cell_extend_yneg_min=self.cell_extend_yneg_min,
            cell_extend_yneg_max=self.cell_extend_yneg_max,
            cell_extend_ypos_min=self.cell_extend_ypos_min,
            cell_extend_ypos_max=self.cell_extend_ypos_max,
            cell_extend_zneg_min=self.cell_extend_zneg_min,
            cell_extend_zneg_max=self.cell_extend_zneg_max,
            cell_extend_zpos_min=self.cell_extend_zpos_min,
            cell_extend_zpos_max=self.cell_extend_zpos_max,
            cut_scan_granularity=self.cut_scan_granularity,
            num_process=self.num_process,
            max_num_rs=self.max_num_rs,
            slab_interval=self.slab_interval,
        )


class CSCECutStrategy:
    """
    Cut Strategy for obtaining substructure, including Crop, Shrink and Cubic Extend (CSCE)
    """

    def __init__(
        self,
        *,
        method: str = "ideal",
        sub_cutoff: float,
        cell_extend_min: float = 0.0,
        cell_extend_max: float | None = None,
        cell_extend_xneg_min: float | None = None,
        cell_extend_xneg_max: float | None = None,
        cell_extend_xpos_min: float | None = None,
        cell_extend_xpos_max: float | None = None,
        cell_extend_yneg_min: float | None = None,
        cell_extend_yneg_max: float | None = None,
        cell_extend_ypos_min: float | None = None,
        cell_extend_ypos_max: float | None = None,
        cell_extend_zneg_min: float | None = None,
        cell_extend_zneg_max: float | None = None,
        cell_extend_zpos_min: float | None = None,
        cell_extend_zpos_max: float | None = None,
        cut_scan_granularity: float = 0.1,
        num_process: int = 1,
        max_num_rs: int = 1000,
    ):
        assert method in ["naive", "ideal"], "method must be 'naive' or 'ideal'"
        self.method = method
        self.sub_cutoff = sub_cutoff
        self.unc_model = None
        self.cut_scan_granularity = cut_scan_granularity
        self.num_process = min(num_process, mp.cpu_count())
        self.max_num_rs = max_num_rs

        # List of axis and direction suffixes
        axes_directions = ["xneg", "xpos", "yneg", "ypos", "zneg", "zpos"]

        # Dynamically set cell_extend parameters
        for direction in axes_directions:
            min_attr = f"cell_extend_{direction}_min"
            max_attr = f"cell_extend_{direction}_max"

            min_value = locals().get(min_attr, None)
            max_value = locals().get(max_attr, None)

            # Set min value
            setattr(
                self,
                min_attr,
                min_value if min_value is not None else cell_extend_min,
            )

            # Set max value
            setattr(
                self,
                max_attr,
                max_value if max_value is not None else cell_extend_max,
            )

        # Ensure at least one min value is set for all directions
        if any(
            getattr(self, f"cell_extend_{direction}_min") is None
            for direction in axes_directions
        ) or any(
            getattr(self, f"cell_extend_{direction}_max") is None
            for direction in axes_directions
        ):
            raise ValueError("cell_extend not set for all directions")

    def assign_unc_model(self, unc_model: UncModuleBase):
        self.unc_model = unc_model

    def _generate_all_cell_extends(self):
        all_cell_extends = np.array(
            np.meshgrid(
                np.arange(
                    getattr(self, "cell_extend_xneg_min"),
                    getattr(self, "cell_extend_xneg_max") + 1e-6,
                    self.cut_scan_granularity,
                ),
                np.arange(
                    getattr(self, "cell_extend_xpos_min"),
                    getattr(self, "cell_extend_xpos_max") + 1e-6,
                    self.cut_scan_granularity,
                ),
                np.arange(
                    getattr(self, "cell_extend_yneg_min"),
                    getattr(self, "cell_extend_yneg_max") + 1e-6,
                    self.cut_scan_granularity,
                ),
                np.arange(
                    getattr(self, "cell_extend_ypos_min"),
                    getattr(self, "cell_extend_ypos_max") + 1e-6,
                    self.cut_scan_granularity,
                ),
                np.arange(
                    getattr(self, "cell_extend_zneg_min"),
                    getattr(self, "cell_extend_zneg_max") + 1e-6,
                    self.cut_scan_granularity,
                ),
                np.arange(
                    getattr(self, "cell_extend_zpos_min"),
                    getattr(self, "cell_extend_zpos_max") + 1e-6,
                    self.cut_scan_granularity,
                ),
            )
        ).T.reshape(-1, 6)
        chosen_indices = np.random.choice(
            all_cell_extends.shape[0],
            size=int(min(self.max_num_rs, len(all_cell_extends))),
            replace=False,
        )
        all_cell_extends = all_cell_extends[chosen_indices]
        return all_cell_extends

    def _get_centered_parent_pos(
        self,
        atoms: Atoms,
        center_index: int,
    ):
        """
        Get the positions of the parent atoms centered at the center atom.
        """
        scaled_pos = np.array(atoms.get_scaled_positions())
        scaled_pos = np.mod(
            scaled_pos - scaled_pos[center_index].reshape(1, 3) + 0.5, 1
        )
        cell = np.array(atoms.get_cell(complete=True))
        positions = scaled_pos @ cell
        positions = positions - positions[center_index].reshape(1, 3)
        return positions

    def _single_cut(
        self,
        atoms: Atoms,
        center_index: int,
        edge_indices: np.ndarray,
        sub_cutoff: float,
        all_cell_extends: np.ndarray,
        unc_model: UncModuleBase,
        num_process: int,
    ) -> Atoms:
        sub_i_naive = _naive_cut(atoms, center_index, edge_indices)
        if self.method == "naive":
            sub_i_naive.info["movable_indices"] = []
            sub_i_naive.info["center_index"] = 0
            sub_i_naive.info["center_index_in_parent"] = center_index
            return sub_i_naive
        sub_i_list = []
        unc_list = []
        if num_process == 1:
            sub_i_list, unc_list = _try_all_cell_extends(
                sub=sub_i_naive,
                centered_parent_pos=self._get_centered_parent_pos(atoms, center_index),
                parent_symbols=atoms.get_chemical_symbols(),
                cutoff=sub_cutoff,
                center_index=center_index,
                all_cell_extends=all_cell_extends,
                unc_model=unc_model,
            )
        else:
            with mp.Pool(num_process) as pool:
                results = [
                    pool.apply_async(
                        _try_all_cell_extends,
                        args=(
                            sub_i_naive,
                            self._get_centered_parent_pos(atoms, center_index),
                            atoms.get_chemical_symbols(),
                            sub_cutoff,
                            center_index,
                            all_cell_extends,
                            unc_model,
                        ),
                    )
                    for _ in range(num_process)
                ]
                for result in results:
                    sub_i_list.extend(result.get()[0])
                    unc_list.extend(result.get()[1])
        min_unc_index = np.argmin(unc_list)
        return sub_i_list[min_unc_index]

    def _post_single_cut(self, sub: Atoms):
        return sub

    def cut(
        self,
        atoms: Atoms,
        center_indices: list[int],
        pbc: bool = True,
    ):
        """
        Cutting method for obtaining substructure
        Args:
            atoms: the atoms object.
            center_indices: the indices of the center atoms.
            pbc: whether the generated substructures should have periodic boundary conditions.
        Returns:
            subs_list: the list of substructures.
        """
        if self.unc_model is None:
            raise ValueError(
                "Uncertainty model is not assigned, please assign it first using <assign_unc_model()>"
            )
        time1 = time.time()
        subs_list = []
        edge_indices = _get_edge_indices(atoms, self.sub_cutoff)
        for center_index in center_indices:
            sub_i = self._single_cut(
                atoms,
                center_index,
                edge_indices,
                self.sub_cutoff,
                self._generate_all_cell_extends(),
                self.unc_model,
                self.num_process,
            )
            sub_i = self._post_single_cut(sub_i)
            if pbc:
                sub_i.set_pbc(True)
            subs_list.append(sub_i)
            # Print progress
            print(
                "\rCutting: {}/{} in {:.2f}s, Size: {}".format(
                    len(subs_list), len(center_indices), time.time() - time1, len(sub_i)
                ),
                end="",
            )
        print()
        return subs_list


class CSCECutStrategyCatalyst(CSCECutStrategy):
    @override
    def __init__(
        self,
        *args,
        slab_interval: float = 15.0,
        **kwargs,
    ):
        """
        Initialize Catalyst-specific cut strategy with additional parameters.
        """
        # Initialize the parent class with original arguments
        super().__init__(*args, **kwargs)

        # Initialize the new parameter
        self.slab_interval = slab_interval

    @override
    def _get_centered_parent_pos(self, atoms: Atoms, center_index: int):
        rotated_pos = _rotate_to_z(atoms, center_index)
        return rotated_pos - rotated_pos[center_index].reshape(1, 3)

    def _extend_cell_in_zpos(
        self,
        atoms: Atoms,
    ):
        """
        Extend the cell in the z-axis positive direction.
        """
        positions = np.array(atoms.get_positions())
        cell = np.array(atoms.get_cell(complete=True))
        z_axis = cell[2]
        z_length = np.linalg.norm(z_axis)
        beta = (0.5 * z_length + self.slab_interval) / z_length
        cell[2] = z_axis * beta
        atoms.set_cell(cell)
        atoms.set_positions(positions)
        return atoms

    @override
    def _post_single_cut(self, sub: Atoms):
        return self._extend_cell_in_zpos(sub)
