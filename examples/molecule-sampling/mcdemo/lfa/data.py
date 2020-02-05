"""Specific implementations of data stores"""

from typing import Tuple, Union

from ase import Atoms
from pymatgen import Molecule
from pymatgen.io.ase import AseAtomsAdaptor

from proxima.data import InMemoryDataStorage


class ASEDataStore(InMemoryDataStorage):
    """Stores data from an ASE simulation as Pymatgen objects because they work better with matminer

    Allows for simpler implementations of the other LFA components"""

    # TODO (wardlt): Should probably build in the "conversion" format into the LFAEngine
    #  The `AseAtomsAdaptor` class is used several different places in this code base

    def __init__(self, convert_to_pmg=True):
        """
        Args:
             convert_to_pmg (bool): Whether to convert the Atoms object to PMG before storing
        """
        self.convert_to_pmg = convert_to_pmg
        super().__init__()

    def transform(self, args: Tuple[Atoms]) -> Union[Atoms, Molecule]:
        atoms = args[0]
        if self.convert_to_pmg:
            return AseAtomsAdaptor.get_molecule(atoms)
        return atoms

    def add_pair(self, inputs, outputs):
        inputs = self.transform(inputs)
        super().add_pair(inputs, outputs)

    def add_pairs(self, inputs, outputs):
        inputs = [self.transform(i) for i in inputs]
        return super().add_pairs(inputs, outputs)
