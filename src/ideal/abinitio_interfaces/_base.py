from __future__ import annotations

from abc import ABC, abstractmethod

from ase import Atoms
from pydantic import BaseModel


class AbinitioInterfaceConfigBase(BaseModel):
    @abstractmethod
    def create_interface(self) -> AbinitioInterfaceBase:
        pass


class AbinitioInterfaceBase(ABC):
    @abstractmethod
    def run(self, atoms: Atoms) -> Atoms | None:
        """
        Run an ab initio calculation on the given atoms.
        """
        pass
