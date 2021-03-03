from pathlib import Path
from typing import List,Optional

import numpy as np
from ase import Atoms
from maml.apps.pes import GAPotential
from quippy.potential import Potential
from pymatgen.io.ase import AseAtomsAdaptor


def make_periodic(atoms: Atoms, vacuum_buffer: float = 10) -> Atoms:
    """Make an atoms object periodic
    
    Required for fitting with ``fit_gap``
    
    Args:
        atoms: Atoms object to be adjusted
        vacuum_buffer: Amount of space to buffer on either side of the molecule
    Returns:
        Atoms object made periodic
    """

    # Give the cell periodic boundary conditions in each direction
    atoms.pbc = [True] * 3
    
    # Compute the size of the cell in each direction
    mol_size = np.max(atoms.positions, axis=0) - np.min(atoms.positions, axis=0)
    
    # Give a cell that is big enough to have the desired buffer on each size
    atoms.cell = np.diag(mol_size + vacuum_buffer * 2)
    
    # Center the atoms in the middle of it
    atoms.center()
    return atoms


def fit_gap(atoms: List[Atoms], energies: List[float], forces: Optional[np.ndarray] = None, **kwargs) -> Potential:
    """Fit a GAP potential using MAL and return a ready-to-use QUIPPy object
    
    Args:
        atoms: List of molecules to use for training
        energies: List of energies for each structure
        forces: If available, forces acting on each atom
        output_dir: Directory in which to store potential
        kwargs: Passed to the :meth:`GAPotential.train` method
    Returns:
        ase Potential object instantiated with a model fit to the data
    """

    # Convert all of the structures to periodic
    atoms = [make_periodic(a.copy()) for a in atoms]

    # Conver them to PyMatgen objects
    conv = AseAtomsAdaptor()
    strcs = [conv.get_structure(a) for a in atoms]

    # Fit using maml
    gap = GAPotential()
    gap.train(strcs, energies, forces, **kwargs)

    # Save to disk and return the QUIPy object
    gap.write_param()
    return Potential(param_filename="gap.xml")
