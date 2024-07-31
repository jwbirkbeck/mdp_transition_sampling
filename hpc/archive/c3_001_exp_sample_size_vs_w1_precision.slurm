#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH -p batch
#SBATCH --job-name=c3_001_exp_sample_size_vs_w1_precision
#SBATCH --time=60:00:00
#SBATCH --output=c3_001_exp_sample_size_vs_w1_precision.out
#SBATCH --error=c3_001_exp_sample_size_vs_w1_precision.err
#SBATCH --mail-type=ALL

module load conda/py3-latest
source activate mdp_transition_sampling

cd /home/$USER/mdp_transition_sampling
pip install -e .

staskfarm hpc/c3_001_exp_sample_size_vs_w1_precision.txt
