#!/bin/bash
sbatch --output="mysimulation.log" --open-mode=truncate -n 4 --time=05:00 --wrap="mpiexec python mysimulation.py"
