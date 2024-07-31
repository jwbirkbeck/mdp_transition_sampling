#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH -p batch
#SBATCH --job-name=mw_w1_vs_sopr
#SBATCH --time=60:00:00
#SBATCH --output=mw_w1_vs_sopr.out
#SBATCH --error=mw_w1_vs_sopr.err
#SBATCH --mail-type=ALL

module load conda/py3-latest
source activate mdp_transition_sampling

cd /home/$USER/mdp_transition_sampling
pip install -e .

staskfarm hpc/mw_w1_vs_sopr.txt
