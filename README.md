# proxima
System for learning approximations for and replacing expensive function 
calls on-the-fly

## Installation

The environment needed to run proxima and the examples is contained
within the ``environment.yml`` file. Install it using Anaconda:

``conda env create --file environment.yml --force``

At present, the environment will only install on Mac and Linux due to the 
Psi4 dependency. Psi4 is only needed for one of the demo applications,
so you can remove it if needed.
