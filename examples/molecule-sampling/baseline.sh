#! /bin/bash

for temp in 1 500; do
  python monte_carlo.py --temp $temp -n 1000 --fidelity low --uq-tolerance -1 --mol 1 --perturb 0.003 --retrain-interval 10000 -S 1
done
