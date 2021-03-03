# Monte Carlo Demo Application

This demo application models a simple type of atomic scale simulation: Monte Carlo sampling.

## Installation

First, install the environment in the base directory of this repo:

`conda env create --file environment.yml`

Then update the environment with the YAML file in this folder:

`conda env update --file update_environment.yml`

Next, install `quippy` in the `proxima` environment by:

1. Activating the environment
1. Downloading QUIP from GitHub (see [`libAtoms` docs](https://libatoms.github.io/GAP/installation.html))
1. Get a copy of the GAP source code [from libAtoms](http://www.libatoms.org/gap/gap_download.html)
1. Unpack the `GAP.tar` archive in the `./src/` directory of QUIP
1. Compile and install the package following [`libAtoms`'s documentation](https://libatoms.github.io/GAP/installation.html#quick-start)
    1. Ignore the "--user" flag for the `QUIPPY_INSTALL_OPTS`. You do not need to install in user mode because we have an isolated enviornment with Anaconda.
    1. You will need to run `make install-quippy` to install the Python module
1. Once complete, add `build/[your build directory]/fit_gap` to your path

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

Our application computes the average radius of gyration of a molecule, 
which should increase with temperature.

Assessing the energy of a system is main computational cost
of the energy evaluation.
This implementation uses a quantum chemistry code, [Psi4](http://www.psicode.org/),
to evaluate the energy.
It is currently hard-coded to use Density Functional Theory (DFT) for the energy
evaluation.

## Running the Application and Adjusting Simulation Parameters

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
1. _Adjusting the number of steps._ The ``--nsteps`` flag determines how many times 
the main loop is iterated through.
1. _Adjusting fidelity_: The `--fidelity` flag controls how expensive of
a quantum chemistry method is used.

There are a few ways for adjusting the sensitivity to surrogate model errors:

1. _Changing the temperature_. The ``--temp`` flag. Logan's hypothesizes that lower 
temperatures will be more sensitive to errors because the error variance will be lower.
1. _Changing the number of steps_. Logan also thinks that the impact of 
approximation error will be smaller the more steps you average over.
1. _Increasing the molecule size_: Picking a larger number for `--mol` will get a larger
molecule will have a larger $R_g$, which might expose errors in the 
surrogate more easily. 

Ways to control how often the LFA is used:
1. _Increasing the UQ tolerance_: Larger values of the `--uq-tolernace` parameter will
allow the surrogate to be used more often. 
1. _Increasing the temperature_: (Indirect) Will cause perturbations to be accepted more often,
leading to a greater drift in the molecule position and more likely the the inputs will
be outside of the domain of applicability.
1. _Increasing the perturbation size_: Use the ``--perturb`` flag to increase the amount
the structure is changed at each step. Larger perturbations will be less likely to use
the surrogate model, as they will be farther from training points. Default is 0.01

Ways to adjust accuracy of surrogate model:
1. _Changing the maximum observation points_. Our model is build on Kernel methods, which
means the model runtime increases linearly with the number of "observation points."
The maximum number of observation points is fixed by ``--max-model-size``. 
Increasing the numebr of points increases accuracy of the model at the expense 
of longer runtimes.
1. _Increasing the time between retraining_: The `--retrain-interval` sections controls how
often the model is retrained. Increase it from the default of 1 (every time new data is 
acquired) to make the model less accurate.

## Logging Capabilities 

The application creates a new directory with runtime information
each time you invoke it.
The directories are named after the start time and contain several files:

- `run_params.json`: A file with the configuration of the application
- `run_data.csv`: Information from the trajectory, with columns:
    - `step`: Step index
    - `energy`: Energy of the current structure
    - `new_energy`: Energy of the new structure (could be surrogate or target function)
    - `true_new_energy`: Energy computed using the target function
    - `time`: Time step was completed
    - `surrogate`: Whether the surrogate was used
- `host_info.json`: Information about what host was run on
- `lfa_json.json`: Summary statistics of LFA performance
    - `lfa_runs`: Number of times the surrogate model was invoked
    - `lfa_time`: Total time spent running the surrogate model
    - `uq_time`: Time spent assessing the surrogate model is appropriate
    - `train_time`: Time spent re-training the surrogate model
    - `target_runs`: Number of times the function being replaced was run
    - `target_time`: Time spent running the target function
- `result.json`: Output of the simulation code. Gives both the mean and 90%
  confidence intervals of the property (radius of gyration) so that you 
  can assess if differences are significant. 
- `r_g.json`: Raw values of the radius of gyration. Useful if you want to do
  analysis beyond the coarse statistics in `result.json`
  
 ## Measuring Simulation Quality

There are a few routes to measuring the quality of the simulation from the logging data:

1. _Comparing True and Predicted Energies_: The `run_data.csv` model stores the output of the surrogate
model and the actual function result. While not the target observable of the simulation, 
you can compare these two values on-the-fly or afterwards
to assess the quality of the surrogate. 
2. _Comparing Output Observable_: The `result.json` file contains the final observable of the simulation.
The quality of the simulation can be quantified by assessing how the "R_g" changes.
Make sure to use the confidence intervals. The R_g is determined based on an average over
the timesteps in the simulation and can have a large uncertainty.


## Recommended Parameters

The `example-run.sh` performs a "baseline" run of the sampling code.
The code does not use the surrogate model (as `--uq-tolerance` is less than zero)
and samples enough steps to get a good measurement of the radius of gyration.
The measurements are performed with "molecule #1" in our dataset at
1K and 500K. 
The code requires 40 minutes to run on an Intel 2550K and 
produces the following values with a random seed of 1:

- R<sub>g</sub> @ 1 K: 0.27256 &#x212b;
- R<sub>g</sub> @ 500 K: 0.27369 &#x212b;


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
