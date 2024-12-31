# -*- coding: utf-8 -*-
from __future__ import annotations

import copy
import hashlib
import os
import os.path as osp
import random
from functools import lru_cache

import numpy as np
import torch
from ase import Atoms
from ase.io import read, write
from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
)
from typing_extensions import override


@torch.jit.script
def convert_to_single_emb(x, offset: int = 512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x


class AtomCalDataset:
    def __init__(
        self,
        atom_list: list[Atoms],
        energies: list[float],
        forces: list[np.ndarray],
        stresses: list[np.ndarray],
        finetune_task_label: list,
    ):
        self.data = self._preprocess(
            atom_list,
            energies,
            forces,
            stresses,
            finetune_task_label,
        )

    def _preprocess(
        self,
        atom_list,
        energies: list[float],
        forces: list[np.ndarray],
        stresses: list[np.ndarray],
        finetune_task_label: list,
        use_ase_energy: bool = False,
        use_ase_force: bool = False,
        use_ase_stress: bool = False,
    ):
        data_list = []
        for i, (atom, energy, force, stress) in enumerate(
            zip(atom_list, energies, forces, stresses)
        ):
            item_dict = atom.todict()
            item_dict["info"] = {}
            if energy is None:
                energy = 0
            if force is None:
                force = np.zeros([len(atom), 3])
            if stress is None:
                stress = np.zeros([3, 3])
            try:
                energy = atom.get_total_energy() if use_ase_energy else energy
                force = (
                    atom.get_forces(apply_constraint=False) if use_ase_force else force
                )
                stress = atom.get_stress(voigt=False) if use_ase_stress else stress
            except Exception as e:
                RuntimeError(f"Error in {i}th data: {e}")

            if finetune_task_label is not None:
                item_dict["finetune_task_label"] = finetune_task_label[i]
            else:
                item_dict["finetune_task_label"] = 0

            item_dict["info"]["energy"] = energy
            item_dict["info"]["stress"] = stress  # * 160.2176621
            item_dict["forces"] = force
            data_list.append(item_dict)

        return data_list

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        item = self.data[idx]
        return preprocess_atom_item(item, idx)

    def __len__(self):
        return len(self.data)


def preprocess_atom_item(item, idx):
    # numbers = item.pop("numbers")
    numbers = item["numbers"]
    item["x"] = torch.tensor(numbers, dtype=torch.long).unsqueeze(-1)
    # positions = item.pop("positions")
    positions = item["positions"]
    item["pos"] = torch.tensor(positions, dtype=torch.float64)
    item["cell"] = torch.tensor(item["cell"], dtype=torch.float64)
    item["pbc"] = torch.tensor(item["pbc"], dtype=torch.bool)
    item["idx"] = idx
    item["y"] = torch.tensor([item["finetune_task_label"]])
    item["total_energy"] = torch.tensor([item["info"]["energy"]], dtype=torch.float64)
    item["stress"] = torch.tensor(item["info"]["stress"], dtype=torch.float64)
    item["forces"] = torch.tensor(item["forces"], dtype=torch.float64)

    item = Data(**item)

    x = item.x

    item.x = convert_to_single_emb(x)

    return item


class IceWaterDataloader(InMemoryDataset):
    url = [
        "https://github.com/BingqingCheng/ice-in-water/raw/master/liquid-1000/dataset_1000_eVAng.xyz"
    ]

    def __init__(
        self,
        root: str = "contents/",
        dataset_seed: int = 42,
        type: str = "multi-dataset",
        train_size: int = 900,
    ):
        """
        Args:
            root (string): Root directory where the dataset should be saved.
            dataset_seed (int): Random seed for dataset split.
            type (string):
                - "default": refers to that the dataset is split into train, val, test set.
                - "multi-dataset": refers to that the dataset is specifically split into train, val set for multi-dataset potential training.
        """
        self.dataset_seed = dataset_seed
        self.type = type
        obj = hashlib.sha1()
        obj.update(str((self.dataset_seed, self.type, train_size)).encode("utf-8"))
        self.hash = obj.hexdigest()
        self.train_size = train_size
        assert self.train_size <= 900, "train_size should be less than 900"
        super().__init__(root)
        if not osp.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
        processed = osp.exists(self.processed_paths[0])

        if not processed:
            self.process_()

        self.data = (
            read(self.processed_paths[0], format="extxyz", index=":"),
            read(self.processed_paths[1], format="extxyz", index=":"),
            None,
            # read(self.processed_paths[2], format="extxyz", index=":"),
        )

    @property
    @override
    def raw_file_names(self) -> list[str]:
        return ["dataset_1000_eVAng.xyz"]

    @property
    @override
    def raw_dir(self) -> str:
        return osp.join(self.root, "IceWater", "raw")

    @property
    @override
    def processed_dir(self) -> str:
        return osp.join(self.root, "IceWater", "processed{}".format(self.hash))

    @property
    @override
    def processed_file_names(self) -> list[str]:
        return ["train.extxyz", "val.extxyz", "test.extxyz"]

    @override
    def download(self):
        download_url(self.url[0], self.raw_dir)

    def process_(self):
        data = read(self.raw_paths[0], format="extxyz", index=":")
        sorted(data, key=lambda x: x.info["TotEnergy"] / len(x))  # type: ignore
        val_indices = np.linspace(0, len(data) - 1, num=int(0.1 * len(data)), dtype=int)
        train_indices = [i for i in range(len(data)) if i not in val_indices]
        train_indices = random.Random(self.dataset_seed).sample(
            train_indices, self.train_size
        )
        new_train_indices = []
        for i in range(((int)(0.9 * len(data)) // self.train_size)):
            new_train_indices.extend(copy.deepcopy(train_indices))
        if len(new_train_indices) != (int)(0.9 * len(data)):
            res = random.Random(self.dataset_seed).sample(
                train_indices, (int)(0.9 * len(data)) - len(new_train_indices)
            )
            new_train_indices.extend(res)

        write(
            osp.join(self.processed_paths[0]),
            [data[idx] for idx in new_train_indices],  # type: ignore
            format="extxyz",
        )
        write(
            osp.join(self.processed_paths[1]),
            [data[idx] for idx in val_indices],  # type: ignore
            format="extxyz",
        )

    @override
    def __repr__(self) -> str:
        return "IceWater"


if __name__ == "__main__":
    train, val, _ = IceWaterDataloader("contents/", train_size=900).data
    print(len(train), len(val))
    train, val, _ = IceWaterDataloader("contents/", train_size=20).data
    print(len(train), len(val))
