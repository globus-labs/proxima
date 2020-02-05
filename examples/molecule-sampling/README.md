# Monte Carlo Demo Application

This demo application models a simple type of atomic scale simulation: Monte Carlo sampling.

## Application Description

Monte Carlo sampling works by computing the average property of an atomic system
at a certain temperature. 
It works by making small perturbations to an atomic system, choosing
whether to accept the perturbation as a new starting point based on 
with a probability related to the energy change, and then 
repeating for many timesteps.
With some [careful consideration of how probabilities 
are chosen](https://en.wikipedia.org/wiki/Monte_Carlo_method_in_statistical_physics),
the average over the timesteps represents a property
for a simulation at the set temperature. 

Assessing the energy of a system is main computational cost
of the energy evaluation.
This implementation uses a quantum chemistry code, [Psi4](http://www.psicode.org/),
to evaluate the energy.
It is currently hard-coded to use Density Functional Theory (DFT) for the energy
evaluation.

## Running the Application and Adjust Simulation Parameters

The application is a serial application launched through the CLI:

```bash
python monte_carlo.py
``` 

There are several command line arguments for the code which can be
written to screen with ``python monte_carlo.py --help``.

There are two ways to vary the problem size:

1. _Changing the molecule size._ The `--mol` flag controls which
molecule from the QM9 database is used for the simulation. There are
133k molecules and they are roughly ordered by increasing molecular size.
So, larger numbers in the ``--mol`` flag will lead to longer run times 
2. _Adjusting the number of steps._ The ``--nsteps`` flag determines how many times 
the main loop is iterated through.

There are a few ways for adjusting the sensitivity to surrogate model errors:

1. _Changing the temperature_. The ``--temp`` flag. Logan's hypothesizes that lower 
temperatures will be more sensitive to errors because the error variance will be lower.
2. _Changing the number of steps_. Logan also thinks that the impact of 
approximation error will be smaller the more steps you average over.

Ways to control how often the LFA is used:
1. _Increasing the temperature_: (Indirect) Will cause perturbations to be accepted more often,
leading to a greater drift in the molecule position.

Logan intends the ability to add in control over the number of threads
used by Psi4 as a simple way to control the runtime of the energy step.

## Logging Capabilities 

Thea application creates a new directory with runtime information
each time you invoke it.
The directories are named after the start time and contain several files:

- `run_params.json`: A file with the configuration of the application
- `run_data.csv`: Information from the trajectory, with columns:
    - `step`: Step index
    - `energy`: Energy of the current structure
    - `new_energy`: Energy of the new structure
    - `time`: Time step was completed
    - `surrogate`: Whether the surrogate was used
- `host_info.json`: Information about what host was run on
- `lfa_json.json`: Summary statistics of LFA performance

## Implementation Details

We use a few different open-source libraries to build this application
and a custom ``mcdemo`` module to hold utility operations.

The demo app relies heavily on the [`ase`](https://gitlab.com/ase/ase)
module for running the physics portion of the codes.
``ase`` interfaces with Psi4 and is used to manipulate the atomic structures.

The ``mcdemo`` application should be installed with `pip install -e .`
to put the code in devlopement mode and reduce the chance PYTHONPATH-related
chaos. "Development mode" adds the library's current location to the Python path
so that whenever you open a Python interpreter it will import the
latest version of each file.

The ``mcdemo`` package contains the LFA components specific to this application.
