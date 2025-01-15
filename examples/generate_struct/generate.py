from __future__ import annotations

import copy
import random

import numpy as np
from ase import Atoms
from ase.build import bulk, molecule
from ase.cluster import Icosahedron, wulff_construction
from ase.data import chemical_symbols as CHEMICAL_SYMBOLS
from ase.data import covalent_radii
from ase.io import write
from ase.neighborlist import NeighborList, natural_cutoffs, neighbor_list
from tqdm import tqdm

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
    lattice_constant: float | None = None,
    surface_energies: dict | None = None,
    box_extension: float = 10.0,
) -> Atoms:
    """
    Generate a BCC nanoparticle using Wulff construction.

    Parameters:
    - element (str): Element symbol (e.g., 'Fe').
    - size (int): Approximate number of atoms in the nanoparticle.
    - lattice_constant (float): Lattice constant for the BCC structure. If None, use default.
    - surface_energies (dict): Surface energies for specific Miller indices.
                               Default: {'100': 2.5, '110': 2.0, '111': 2.2}.
    - box_extension (float): Extra box space for the periodic boundary conditions.

    Returns:
    - nanoparticle (Atoms): The BCC nanoparticle in a PBC box.
    """
    # Default surface energies for BCC
    if surface_energies is None:
        surface_energies = {"100": 2.5, "110": 2.0, "111": 2.2}

    # Use a default lattice constant if not provided
    if lattice_constant is None:
        lattice_constants = {
            "Fe": 2.866,
            "Cr": 2.885,
            "Mo": 3.147,
        }  # Example values in Å
        lattice_constant = lattice_constants.get(element, 3.0)  # Default to 3.0 Å

    # Convert surface energies to the required format
    surfaces = [(1, 0, 0), (1, 1, 0), (1, 1, 1)]
    energies = [
        surface_energies["100"],
        surface_energies["110"],
        surface_energies["111"],
    ]

    # Generate Wulff nanoparticle with explicit lattice constant
    nanoparticle = wulff_construction(
        symbol=element,
        surfaces=surfaces,
        energies=energies,
        size=size,
        structure="bcc",
        latticeconstant=lattice_constant,  # Pass lattice constant here
    )

    # Define the PBC box dimensions
    positions = nanoparticle.get_positions()
    box_size = np.ptp(positions, axis=0) + box_extension
    nanoparticle.set_cell(box_size)
    nanoparticle.set_pbc(True)

    return nanoparticle


def generate_hcp_Ru_nanoparticle(
    np_size: int, nanoparticle_element: str, box_extension: float
) -> Atoms:
    """
    Generate a nanoparticle with the given size and element inside a PBC box.
    """
    try:
        # 这里用一个简单的经验式，把 np_size 换算成 icosahedron 的壳数
        n_shells = max(3, int(np.ceil(np.float_power(np_size / 3000, 1 / 3))))
        nanoparticle = Icosahedron(nanoparticle_element, noshells=n_shells)
    except NotImplementedError as error:
        # 有些元素 (hcp) 不支持 icosahedron，就用 wulff_construction 替代
        print("Use hcp Wulff construction instead of Icosahedron.")
        if "hcp" in str(error):
            surfaces = [(0, 0, 0, 1), (1, 0, -1, 0), (1, 0, -1, 1)]
            esurf = [2.0, 2.5, 2.8]
            nanoparticle = wulff_construction(
                symbol=nanoparticle_element,
                surfaces=surfaces,
                energies=esurf,
                size=np_size,
                structure="hcp",
                latticeconstant={"a": 2.705, "c/a": 1.7},
                rounding="closest",
            )
        else:
            raise error

    positions = nanoparticle.get_positions()
    box_size = np.ptp(positions, axis=0) + box_extension
    nanoparticle.set_cell(box_size)
    nanoparticle.set_pbc(True)

    return nanoparticle


def convert_pressure_to_num_molecules(
    system: Atoms, pressure_atm: float, nanoparticle_element: str
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

    # 单位转换
    gas_volume_m3 = gas_volume * 1e-30  # Å³ to m³
    pressure_pa = pressure_atm * 101325  # atm to Pa

    # 使用理想气体方程计算
    num_moles = (pressure_pa * gas_volume_m3) / (R * T)

    # 如果压力很高，可以考虑使用范德华方程校正
    if pressure_atm > 100:  # 超过100atm考虑实际气体效应
        print("警告：在高压下理想气体方程可能产生显著误差")

    return int(num_moles * 6.022e23)


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


def compute_pressure_and_num(
    system: Atoms, molecule_element: str, nanoparticle_element: str
):
    atomic_symbols = system.get_chemical_symbols()
    positions = np.array(system.get_positions())
    cell = np.array(system.get_cell(complete=True))

    molecule_indices = [
        i for i, s in enumerate(atomic_symbols) if s == molecule_element
    ]
    assert len(molecule_indices) % 2 == 0, "Molecule indices must be even."
    num_molecule = len(molecule_indices) // 2

    particle_indices = [
        i for i, s in enumerate(atomic_symbols) if s == nanoparticle_element
    ]
    particle_positions = positions[particle_indices]
    center = np.mean(particle_positions, axis=0).reshape(1, 3)
    dist_from_center = np.linalg.norm(particle_positions - center, axis=1)
    approximate_radius = np.max(dist_from_center)

    box_volume = np.linalg.det(cell)
    particle_volume = 4 / 3 * np.pi * approximate_radius**3
    gas_volume = box_volume - particle_volume

    # 单位转换
    gas_volume_m3 = gas_volume * 1e-30  # Å³ to m³

    # 使用理想气体方程计算压力
    pressure_pa = num_molecule * R * T / (gas_volume_m3 * 6.022e23)  # Pa

    # 转换为atm
    pressure_atm = pressure_pa / 101325  # Pa to atm

    # 如果压力很高，提醒用户
    if pressure_atm > 100:
        print("警告：在高压下理想气体方程可能产生显著误差")

    return pressure_atm, num_molecule


def add_molecules(
    system: Atoms,
    molecule_type: str,
    num_molecules: int,
    mol_min_distance: float,
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

    box_size = np.array(system.get_cell()).diagonal()
    existing_positions = system.get_positions()

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
                system += molecule_copy
                pbar.update(1)
                break
    pbar.close()
    return system


def add_promoter(
    system: Atoms,
    nanoparticle_element: str,
    promoter_element: str,
    num_promoters: int,
    method: str = "on_surface",
    del_overlap: bool = False,
    # 下述参数用于均匀分布控制
    promoter_promoter_min_dist: float = 3.0,
) -> Atoms:
    """
    Add promoter atoms by two methods: 'on_surface' and 'semi_embedded'.

    - on_surface:
        Place atoms near the outer surface by generating random positions
        in a spherical shell near the radius, ensuring:
          1) Not too close to existing atoms
          2) Not too close to other promoters (for more uniform distribution)

    - semi_embedded:
        Replace existing surface/near-surface atoms with promoter_element,
        ensuring replaced atoms are also not too close to each other in space.

    Args:
        system: ASE Atoms object representing the system.
        nanoparticle_element: Host metal element (e.g. 'Ru').
        promoter_element: Promoter element (e.g. 'K').
        num_promoters: Number of promoters to add/replace.
        method: 'on_surface' or 'semi_embedded'.
        promoter_promoter_min_dist: Minimum distance between newly added Promoter atoms (on_surface).
        surface_uniform_min_dist: Minimum distance among replaced atoms (semi_embedded).

    Returns:
        Updated ASE Atoms object with added or replaced promoter atoms.
    """
    if num_promoters == 0:
        return system

    method = method.lower()
    if method not in ["on_surface", "semi_embedded", "semi_embedded_deep"]:
        raise ValueError(
            "Method must be 'on_surface', 'semi_embedded', or 'semi_embedded_deep'."
        )

    # 读取原子半径 (可选，如果需要用在后续逻辑里)
    element_radius = {
        s: covalent_radii[CHEMICAL_SYMBOLS.index(s) - 1]
        for s in [nanoparticle_element, promoter_element]
    }

    # 计算系统中所有原子的空间中心 & 纳米颗粒的近似半径
    positions = np.array(system.get_positions())
    cell = np.array(system.get_cell(complete=True))
    center = np.mean(positions, axis=0)
    atomic_symbols = system.get_chemical_symbols()

    if method == "on_surface":
        surface_indices = get_surface_indices(system, [nanoparticle_element], [])
        radius_data = covalent_radii
        nano_promoter_diameter = (
            radius_data[CHEMICAL_SYMBOLS.index(nanoparticle_element)]
            + radius_data[CHEMICAL_SYMBOLS.index(promoter_element)]
        )

        promoter_positions = []
        existing_positions = np.array(positions)
        for _ in range(num_promoters):
            trial_count = 0
            random.shuffle(surface_indices)
            for surface_index in surface_indices:
                surface_pos = positions[surface_index]
                center_to_surface = surface_pos - center
                center_to_surface = center_to_surface / np.linalg.norm(
                    center_to_surface
                )
                offset = center_to_surface * nano_promoter_diameter
                new_promoter_pos = surface_pos + offset
                # 与当前已添加的 Promoter 的距离
                if len(promoter_positions) > 0:
                    dist_to_promoters = np.linalg.norm(
                        np.array(promoter_positions) - new_promoter_pos.reshape(1, 3),
                        axis=1,
                    )
                    if np.any(dist_to_promoters < promoter_promoter_min_dist):
                        continue
                # 通过检查，接受这次放置
                promoter_positions.append(new_promoter_pos)
                break
            else:
                print("Warning: Could not place a promoter satisfying all constraints.")
                # 也可以 break 或者直接跳过

        # 放置完后，把 promoter 塞进 system
        promoter_positions = np.array(promoter_positions).reshape(-1, 3)
        for pos in promoter_positions:
            atomic_symbols.append(promoter_element)
        final_positions = np.vstack([positions, promoter_positions])
        system_with_promoters = Atoms(
            atomic_symbols, positions=final_positions, cell=cell, pbc=True
        )

    elif "semi_embedded" in method:
        # semi_embedded: 通过“替换”方式，将若干表面/近表面原子换成 promoter
        cutoffs = natural_cutoffs(system)
        nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
        nl.update(system)

        surface_indices = []
        for i in range(len(system)):
            indices, _ = nl.get_neighbors(i)
            coord_number = len(indices)
            # 简单判定：配位数较低(比如 < 10)的原子视为表面/近表面原子
            if method == "semi_embedded":
                if coord_number > 8 and coord_number <= 10:
                    surface_indices.append(i)
            elif method == "semi_embedded_deep":
                if coord_number > 10 and coord_number <= 13:
                    surface_indices.append(i)
            else:
                raise ValueError("Invalid method.")

        if len(surface_indices) < num_promoters:
            print(
                "Warning: Not enough surface atoms to replace. "
                f"Need {num_promoters} but only {len(surface_indices)} available."
            )

        # 为了均匀分布，先打乱 surface_indices 的顺序，再按“彼此距离排斥”的思路选取
        random.shuffle(surface_indices)

        chosen_indices = []
        chosen_positions = []
        for idx in surface_indices:
            if len(chosen_indices) >= num_promoters:
                break
            candidate_pos = positions[idx]
            # 检查与已经选定的表面原子在空间上是否太近
            if len(chosen_positions) == 0:
                chosen_indices.append(idx)
                chosen_positions.append(candidate_pos)
            else:
                dists = np.linalg.norm(
                    np.array(chosen_positions) - candidate_pos, axis=1
                )
                if np.all(dists > promoter_promoter_min_dist):
                    chosen_indices.append(idx)
                    chosen_positions.append(candidate_pos)

        # 如果最后还是选不满 num_promoters，可能说明限制太严格
        if len(chosen_indices) < num_promoters:
            print(
                f"Warning: Only able to choose {len(chosen_indices)} surface atoms "
                f"under 'surface_uniform_min_dist={promoter_promoter_min_dist}'."
            )

        # 真正执行替换
        for idx in chosen_indices:
            atomic_symbols[idx] = promoter_element

        # 删除重叠原子
        if del_overlap:
            promoter_positions = np.array([positions[idx] for idx in chosen_indices])
            covalent_radius_promoter = element_radius[promoter_element]
            covalent_radius_host = element_radius[nanoparticle_element]
            deletion_indices = []

            for i, pos in enumerate(positions):
                # 如果当前位置是 promoter，自然跳过
                if i in chosen_indices:
                    continue
                # 计算到所有 promoter 的距离
                distances_to_promoters = np.linalg.norm(
                    promoter_positions - pos.reshape(1, 3), axis=1
                ).flatten()
                if np.any(
                    distances_to_promoters
                    < covalent_radius_promoter + covalent_radius_host
                ):
                    deletion_indices.append(i)

            # 删除重叠原子
            remaining_positions = [
                pos for i, pos in enumerate(positions) if i not in deletion_indices
            ]
            remaining_symbols = [
                sym for i, sym in enumerate(atomic_symbols) if i not in deletion_indices
            ]
        else:
            remaining_positions = positions
            remaining_symbols = atomic_symbols

        system_with_promoters = Atoms(
            remaining_symbols, positions=remaining_positions, cell=cell, pbc=True
        )
    else:
        raise ValueError("Invalid method.")

    return system_with_promoters


# ------------------------ 测试与使用示例 ------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--np_size", type=int, default=1500)
    parser.add_argument("--nanoparticle_element", type=str, default="Fe")
    parser.add_argument("--box_extension", type=float, default=15.0)
    parser.add_argument("--promoter_element", type=str, default="K")
    parser.add_argument("--promoters_ratio", type=float, default=0.025)
    parser.add_argument("--add_promoter_method", type=str, default="on_surface")
    parser.add_argument("--delete_overlap", type=bool, default=True)
    parser.add_argument("--promoter_promoter_min_dist", type=float, default=12.0)
    parser.add_argument("--H2_density", type=float, default=0.00402 * 2)
    parser.add_argument("--N2_density", type=float, default=0.00402 * 2)
    parser.add_argument("--mol_min_distance", type=float, default=2.5)
    args = parser.parse_args()
    args_dict = vars(args)

    # ==================== 参数设置 ====================
    np_size = args_dict["np_size"]
    nanoparticle_element = args_dict["nanoparticle_element"]
    box_extension = args_dict["box_extension"]

    promoter_element = args_dict["promoter_element"]
    promoters_ratio = args_dict["promoters_ratio"]
    num_promoters = int(np_size * promoters_ratio)
    add_promoter_method = args_dict["add_promoter_method"]
    delete_overlap = args_dict["delete_overlap"]
    promoter_promoter_min_dist = args_dict["promoter_promoter_min_dist"]

    # H2_pressure = 100  ## 0.0402 N^2/Å^3
    H2_density = args_dict["H2_density"]
    # N2_pressure = 100  ## 0.0402 N^2/Å^3
    N2_density = args_dict["N2_density"]
    mol_min_distance = args_dict["mol_min_distance"]
    # ==================== 参数设置 ====================

    # 1. 生成纳米粒子
    if nanoparticle_element == "Fe":
        cluster = generate_bcc_Fe_nanoparticle(
            element="Fe",
            size=np_size,
            surface_energies={"100": 2.4, "110": 2.1, "111": 2.3},
            box_extension=box_extension,
        )
    else:
        cluster = generate_hcp_Ru_nanoparticle(
            np_size, nanoparticle_element, box_extension
        )
    cluster = pos_normalize(cluster, nanoparticle_element)

    # 2. 添加 promoter
    #    这里演示如何使 Promoter 尽量分布均匀 (on_surface 和 semi_embedded 都加了逻辑)
    cluster = add_promoter(
        cluster,
        nanoparticle_element=nanoparticle_element,
        promoter_element=promoter_element,
        num_promoters=num_promoters,
        method=add_promoter_method,
        del_overlap=delete_overlap,
        promoter_promoter_min_dist=promoter_promoter_min_dist,
    )

    # symbols = cluster.get_chemical_symbols()
    # Fe_indices = [i for i, s in enumerate(symbols) if s == nanoparticle_element]
    # for i in Fe_indices:
    #     symbols[i] = "H"
    # cluster.set_chemical_symbols(symbols)
    # write("./test.xyz", cluster)
    # exit()

    # 3. 添加气体分子
    # num_H2 = convert_pressure_to_num_molecules(
    #     cluster, H2_pressure, nanoparticle_element
    # )
    # num_N2 = convert_pressure_to_num_molecules(
    #     cluster, N2_pressure, nanoparticle_element
    # )
    num_H2 = convert_density_to_num_molecules(cluster, H2_density, nanoparticle_element)
    num_N2 = convert_density_to_num_molecules(cluster, N2_density, nanoparticle_element)
    print(f"Adding {num_H2:.2e} H2 molecules.")
    print(f"Adding {num_N2:.2e} N2 molecules.")
    cluster = add_molecules(cluster, "N2", num_N2, mol_min_distance)
    cluster = add_molecules(cluster, "H2", num_H2, mol_min_distance)

    atomic_symbols = cluster.get_chemical_symbols()
    final_np_size = len([s for s in atomic_symbols if s == nanoparticle_element])
    final_num_promoters = len([s for s in atomic_symbols if s == promoter_element])
    final_H2_pressure, final_H2_num = compute_pressure_and_num(
        cluster, "H", nanoparticle_element
    )
    final_N2_pressure, final_N2_num = compute_pressure_and_num(
        cluster, "N", nanoparticle_element
    )
    final_H2_pressure = int(final_H2_pressure)
    final_N2_pressure = int(final_N2_pressure)

    print(f"Final nanoparticle size: {final_np_size}")
    print(f"Final number of promoters: {final_num_promoters}")
    print(f"Final H2 pressure: {final_H2_pressure} atm")
    print(f"Final N2 pressure: {final_N2_pressure} atm")
    print(f"Final H2 molecules: {final_H2_num}")
    print(f"Final N2 molecules: {final_N2_num}")

    final_name = f"../../contents/habor-bosch/particles/{nanoparticle_element}{final_np_size}-{promoter_element}{final_num_promoters}-{add_promoter_method}-{final_H2_pressure}atmH2-{final_N2_pressure}atmN2.xyz"
    write(final_name, cluster)
