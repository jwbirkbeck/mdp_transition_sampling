#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH -p batch
#SBATCH --job-name=benchmarking_w1_calc
#SBATCH --time=60:00:00
#SBATCH --output=benchmarking_w1_calc.out
#SBATCH --error=benchmarking_w1_calc.err
#SBATCH --mail-type=ALL

module load conda/py3-latest
source activate mdp_transition_sampling

cd /home/$USER/mdp_transition_sampling
pip install -e .

staskfarm hpc/benchmarking_w1_calc.txt
