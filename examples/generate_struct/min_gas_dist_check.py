from __future__ import annotations

import ase.neighborlist as nl
import numpy as np  # 导入 NumPy
from ase import Atoms
from ase.io import read, write

# 读取分子结构
atoms = read("./particles/Fe949-K40-semi_embedded-326atmH2-292atmN2.xyz")
symbols = atoms.get_chemical_symbols()
positions = atoms.get_positions()
cell = atoms.get_cell()

# 筛选出 N 和 H 原子
NH_indices = [i for i, s in enumerate(symbols) if s == "N" or s == "H"]
positions_NH = positions[NH_indices]
symbols_NH = [symbols[i] for i in NH_indices]

# 将 symbols_NH 转换为 NumPy 数组
symbols_NH = np.array(symbols_NH)

# 构建新的 Atoms 对象
gas_struct = Atoms(symbols=symbols_NH, positions=positions_NH, cell=cell, pbc=True)

# 设置截断距离
cutoff = 1.5

# 生成邻居列表
src_indices, dst_indices, dist = nl.neighbor_list(
    "ijd", gas_struct, cutoff, self_interaction=False
)

# 计算最小 H-H 距离
hh_mask = (symbols_NH[src_indices] == "H") & (symbols_NH[dst_indices] == "H")
if np.any(hh_mask):
    min_HH_dist = dist[hh_mask].min()
    print(f"Minimum H-H distance: {min_HH_dist:.4f} Å")
else:
    print("No H-H pairs found within the cutoff distance.")

# 计算最小 N-N 距离
nn_mask = (symbols_NH[src_indices] == "N") & (symbols_NH[dst_indices] == "N")
if np.any(nn_mask):
    min_NN_dist = dist[nn_mask].min()
    print(f"Minimum N-N distance: {min_NN_dist:.4f} Å")
else:
    print("No N-N pairs found within the cutoff distance.")

# 计算最小 H-N 距离
hn_mask = ((symbols_NH[src_indices] == "H") & (symbols_NH[dst_indices] == "N")) | (
    (symbols_NH[src_indices] == "N") & (symbols_NH[dst_indices] == "H")
)
if np.any(hn_mask):
    min_HN_dist = dist[hn_mask].min()
    print(f"Minimum H-N distance: {min_HN_dist:.4f} Å")
else:
    print("No H-N pairs found within the cutoff distance.")
