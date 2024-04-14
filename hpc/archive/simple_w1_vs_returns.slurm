#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH -p batch
#SBATCH --job-name=simple_w1_vs_returns
#SBATCH --time=60:00:00
#SBATCH --output=simple_w1_vs_returns.out
#SBATCH --error=simple_w1_vs_returns.err
#SBATCH --mail-type=ALL

module load conda/py3-latest
source activate mdp_transition_sampling

cd /home/$USER/mdp_transition_sampling
pip install -e .

staskfarm hpc/simple_w1_vs_returns.txt
