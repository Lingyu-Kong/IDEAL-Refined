"""
Provides interfaces to VASP.
Given a list of atoms, the interface will generate the input files for VASP and run the calculation.
"""

from __future__ import annotations

import os

import numpy as np
from ase import Atoms
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.sets import MPRelaxSet
from typing_extensions import override

from ._base import AbinitioInterfaceBase


class VaspInterface(AbinitioInterfaceBase):
    def __init__(
        self,
        user_incar_settings: dict,
        user_kpoints_settings: dict,
        vasp_cmd: str,
        vasp_run_dir: str = "./vasp_run",
    ):
        self.user_incar_settings = user_incar_settings
        self.user_kpoints_settings = user_kpoints_settings
        self.vasp_cmd = vasp_cmd
        self.vasp_run_dir = vasp_run_dir

    @override
    def run(
        self,
        atoms: Atoms,
    ):
        """
        Run a VASP calculation on the given atoms.
        """
        # Convert atoms to pymatgen structure
        structure = AseAtomsAdaptor.get_structure(atoms)  # type: ignore

        # Set up the VASP input files
        vasp_input_set = MPRelaxSet(
            structure,
            user_incar_settings=self.user_incar_settings,
            user_kpoints_settings=self.user_kpoints_settings,
        )
        os.system(f"rm -rf {self.vasp_run_dir}")
        vasp_input_set.write_input(self.vasp_run_dir)

        # Run the VASP calculation
        original_dir = os.getcwd()
        os.chdir(self.vasp_run_dir)
        os.system(self.vasp_cmd)
        os.chdir(original_dir)

        # Read the output
        try:
            labeled_atoms: Atoms = read(f"{self.vasp_run_dir}/vasprun.xml", index=-1)  # type: ignore
            energy = labeled_atoms.get_potential_energy()
            forces = labeled_atoms.get_forces()
            stress = labeled_atoms.get_stress()
            return labeled_atoms
        except Exception as e:
            return None


from ase.calculators.singlepoint import SinglePointCalculator


class VaspFakeInterface(AbinitioInterfaceBase):
    """
    TODO: This class is just for quick testing. Do not use it in production.
    This class will assign zero values to the energy, forces, and stress.
    """

    def __init__(
        self,
        vasp_cmd: str,
        user_incar_settings: dict = {},
        user_kpoints_settings: dict = {},
        vasp_run_dir: str = "./vasp_run",
    ):
        self.user_incar_settings = user_incar_settings
        self.user_kpoints_settings = user_kpoints_settings
        self.vasp_cmd = vasp_cmd
        self.vasp_run_dir = vasp_run_dir

    @override
    def run(
        self,
        atoms: Atoms,
    ):
        """
        Run a VASP calculation on the given atoms.
        """
        # Convert atoms to pymatgen structure
        structure = AseAtomsAdaptor.get_structure(atoms)  # type: ignore

        # Set up the VASP input files
        vasp_input_set = MPRelaxSet(
            structure,
            user_incar_settings=self.user_incar_settings,
            user_kpoints_settings=self.user_kpoints_settings,
        )
        # vasp_input_set.write_input(self.vasp_run_dir)

        # Assign zero values to the energy, forces, and stress
        energy = 0.0
        forces = np.zeros((len(atoms), 3))
        stress = np.zeros((6,))
        calc = SinglePointCalculator(atoms, energy=energy, forces=forces, stress=stress)
        atoms.set_calculator(calc)
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        stress = atoms.get_stress()
        return atoms
