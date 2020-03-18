"""Simple Monte Carlo sampling application"""
from ase import Atoms
from mcdemo.lfa.surrogates import CoulombMatrixKNNSurrogate, GAPSurrogate
from mcdemo.lfa.uq import DistanceBasedUQWithFeaturization
from mcdemo.lfa.data import ASEDataStore
from mcdemo.utils import get_qm9_path, get_platform_info

from ase.calculators.psi4 import Psi4
from ase.io.xyz import read_xyz
from scipy.stats import bayes_mvs

from proxima.decorator import LFAEngine
from proxima.training import PeriodicRetrain
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


# Hard code the fidelity options
_fidelity = {
    'low': {'method': 'HF', 'basis': 'sto-3g'},
    'medium': {'method': 'b3lyp', 'basis': '6-311g_d_p_'},
    'high': {'method': 'ccsd(t)', 'basis': 'cc-pVTZ'}
}


if __name__ == "__main__":
    # Parse the arguments
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--mol', '-m', help='Index of the molecule to use for sampling', default=1, type=int)
    arg_parser.add_argument('--temp', '-T', help='Temperature at which to sample (K).'
                                                 ' Default is 298 (room temperature)', default=298, type=float)
    arg_parser.add_argument('--nsteps', '-n', help='Number of Monte Carlo steps', default=64, type=int)
    arg_parser.add_argument('--random', '-S', help='Random seed', default=None, type=int)
    arg_parser.add_argument('--perturb', '-p', help='Perturbation size', default=0.001, type=float)
    arg_parser.add_argument('--fidelity', '-f', help='Controls the accuracy/cost of the quantum chemistry code',
                            default='low', choices=['low', 'medium', 'high'], type=str)
    arg_parser.add_argument('--max-model-size', '-s', help='Maximum number of points to use in GAP surrogate model',
                            default=None, type=int)
    arg_parser.add_argument('--uq-tolerance', '-u', help='Larger tolerance values will use surrogates more often',
                            default=0.1, type=float)
    arg_parser.add_argument('--retrain-interval', '-t', help='How often to retrain the model. Controls how many new '
                                                             'data points are acquired before the model is retrained',
                            default=1, type=int)

    # Parse the arguments
    args = arg_parser.parse_args()
    run_params = args.__dict__
    kT = 3.1668115635e-6 * args.temp

    # Get the system information
    host_info = get_platform_info()

    # Set the random seed
    np.random.seed(args.random)
    rng = np.random.RandomState(args.random)

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
    out_dir = os.path.join('runs', f'{start_time.strftime("%d%b%y-%H%M%S")}')
    os.makedirs(out_dir)

    # Save the parameters and host information
    with open(os.path.join(out_dir, 'run_params.json'), 'w') as fp:
        json.dump(run_params, fp, indent=2)
    with open(os.path.join(out_dir, 'host_info.json'), 'w') as fp:
        json.dump(host_info, fp, indent=2)

    # Initialize the ASE calculator
    calc = Psi4(memory='500MB', **_fidelity[args.fidelity])

    # Make the LFA wrapper
    lfa_func = LFAEngine(calc.get_potential_energy, GAPSurrogate(args.max_model_size),
                         DistanceBasedUQWithFeaturization(args.uq_tolerance),
                         ASEDataStore(convert_to_pmg=False),
                         PeriodicRetrain(args.retrain_interval))
    calc.get_potential_energy = lfa_func

    # Compute a starting energy
    energy = calc.get_potential_energy(atoms)

    # Function to compute the radius of gyration
    def radius_of_gyration(atoms: Atoms):
        """Compute the radius of gyration of a molecule

        Method: http://www.charmm-gui.org/?doc=lecture&module=scientific&lesson=10
        """
        cm = atoms.get_center_of_mass()
        disp = np.linalg.norm(atoms.get_positions() - cm, 2, axis=1)
        m = atoms.get_masses()
        return np.dot(m, disp) / np.sum(m)

    # Start the Monte Carlo loop
    r_g = []
    energies = []
    with open(os.path.join(out_dir, 'run_data.csv'), 'w') as fp:
        log_file = DictWriter(fp, fieldnames=['step', 'energy', 'new_energy', 'true_new_energy',
                                              'time', 'accept', 'surrogate'])
        log_file.writeheader()

        for step in tqdm(range(args.nsteps)):
            # Make a copy of the structure with rattled positions
            new_atoms = atoms.copy()
            new_atoms.rattle(stdev=args.perturb, rng=rng)

            # Compute the energy
            calc.reset()
            new_energy = calc.get_potential_energy(new_atoms)
            if lfa_func.did_last_call_use_surrogate():
                true_new_energy = lfa_func.target_function(new_atoms)
            else:
                true_new_energy = new_energy

            # Determine whether we should accept this state
            delta_E = new_energy - energy
            prob = np.exp(-delta_E / kT)
            accept = rng.random() < prob

            # Act on decision
            if accept:
                energy = new_energy
                atoms = new_atoms

            # Compute the radius of gyration
            r_g.append(radius_of_gyration(atoms))

            # Store the measured energy
            energies.append(energy)

            # Store the results
            log_file.writerow({'step': step, 'energy': energy, 'new_energy': new_energy,
                               'true_new_energy': true_new_energy,
                               'accept': accept, 'time': perf_counter(),
                               'surrogate': lfa_func.did_last_call_use_surrogate()})

    # Drop out the LFA statistics
    with open(os.path.join(out_dir, 'lfa_stats.json'), 'w') as fp:
        json.dump(lfa_func.get_performance_info()._asdict(), fp, indent=2)

    # Print out the radius of gyration statistics on the last 50% of the steps
    #  Note: One should really check for when the "burn in" period ends but accuracy isn't the point of this demo app
    r_g = r_g[len(r_g) // 2:]
    with open(os.path.join(out_dir, 'r_g.json'), 'w') as fp:
        json.dump(r_g, fp)

    if max(r_g) - min(r_g) < 1e-6:
        # Special Case: The R_g does not change during the whole test
        with open(os.path.join(out_dir, 'result.json'), 'w') as fp:
            json.dump({
                'r_g': {'statistic': r_g[0], 'minmax': [np.nan, np.nan]}
            }, fp, indent=2)
    else:
        stats = bayes_mvs(r_g)
        with open(os.path.join(out_dir, 'result.json'), 'w') as fp:
            json.dump({
                'r_g': stats[0]._asdict()
            }, fp, indent=2)
