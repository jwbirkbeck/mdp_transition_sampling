#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH -p batch
#SBATCH --job-name=c3_002_exp_random_vs_reward_based_sampling
#SBATCH --time=60:00:00
#SBATCH --output=c3_002_exp_random_vs_reward_based_sampling.out
#SBATCH --error=c3_002_exp_random_vs_reward_based_sampling.err
#SBATCH --mail-type=ALL

module load conda/py3-latest
source activate mdp_transition_sampling

cd /home/$USER/mdp_transition_sampling
pip install -e .

staskfarm hpc/c3_002_exp_random_vs_reward_based_sampling.txt
