"""Simple Monte Carlo sampling application"""
from mcdemo.lfa.surrogates import CoulombMatrixKNNSurrogate
from mcdemo.lfa.uq import DistanceBasedUQWithFeaturization
from mcdemo.lfa.data import ASEDataStore
from mcdemo.utils import get_qm9_path, get_platform_info

from ase.calculators.psi4 import Psi4
from ase.io.xyz import read_xyz
from proxima.decorator import LFAEngine
from proxima.training import TrainingEngine
from tqdm import tqdm

from argparse import ArgumentParser
from datetime import datetime
from time import perf_counter
from csv import DictWriter
from io import StringIO

import numpy as np
import gzip
import json
import os


if __name__ == "__main__":
    # Parse the arguments
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--mol', '-m', help='Index of the molecule to use for sampling', default=1, type=int)
    arg_parser.add_argument('--temp', '-T', help='Temperature at which to sample (K).'
                                                 ' Default is 298 (room temperature)', default=298, type=float)
    arg_parser.add_argument('--nsteps', '-n', help='Number of Monte Carlo steps', default=64, type=int)
    arg_parser.add_argument('--random', '-S', help='Random seed', default=None, type=int)

    # Parse the arguments
    args = arg_parser.parse_args()
    run_params = args.__dict__
    kT = 3.1668115635e-6 * args.temp

    # Get the system information
    host_info = get_platform_info()

    # Set the random seed
    np.random.seed(args.random)

    # Download the QM9 dataset and get the molecule of interest
    qm9_path = get_qm9_path()
    with gzip.open(qm9_path, 'rt') as fp:
        for _, d in zip(range(args.mol), fp):
            pass
        mol_info = json.loads(d)

    # Parse the molecule coordinates into an ASE object
    atoms = next(read_xyz(StringIO(mol_info['xyz'])))

    # Open an experiment directory
    start_time = datetime.utcnow()
    out_dir = f'{start_time.strftime("%d%b%y-%H%M%S")}'
    os.makedirs(out_dir)

    # Save the parameters and host information
    with open(os.path.join(out_dir, 'run_params.json'), 'w') as fp:
        json.dump(run_params, fp, indent=2)
    with open(os.path.join(out_dir, 'host_info.json'), 'w') as fp:
        json.dump(host_info, fp, indent=2)

    # Initialize the ASE calculator
    calc = Psi4(atoms=atoms, method='b3lyp', memory='500MB', basis='6-311g_d_p_')

    # Make the LFA wrapper
    lfa_func = LFAEngine(calc.get_potential_energy, CoulombMatrixKNNSurrogate(),
                         DistanceBasedUQWithFeaturization(0.1),
                         ASEDataStore(), TrainingEngine())
    calc.get_potential_energy = lfa_func

    # Compute a starting energy
    energy = calc.get_potential_energy(atoms)

    # Start the Monte Carlo loop
    with open(os.path.join(out_dir, 'run_data.csv'), 'w') as fp:
        log_file = DictWriter(fp, fieldnames=['step', 'energy', 'new_energy', 'time', 'accept'])
        log_file.writeheader()

        for step in tqdm(range(args.nsteps)):
            # Make a copy of the structure with rattled positions
            new_atoms = atoms.copy()
            new_atoms.rattle()

            # Compute the energy
            calc.reset()
            new_energy = calc.get_potential_energy(new_atoms)

            # Determine whether we should accept this state
            prob = np.exp((new_energy - energy) / kT)
            accept = prob < np.random.rand()

            # Store the results
            log_file.writerow({'step': step, 'energy': energy, 'new_energy': new_energy,
                               'accept': accept, 'time': perf_counter()})
            if accept:
                energy = new_energy
                atoms = new_atoms

    # Drop out the LFA statistics
    with open(os.path.join(out_dir, 'lfa_stats.json'), 'w') as fp:
        json.dump(lfa_func.get_performance_info()._asdict(), fp, indent=2)
