#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH -p batch
#SBATCH --job-name=bandit_policy_selection
#SBATCH --time=60:00:00
#SBATCH --output=bandit_policy_selection.out
#SBATCH --error=bandit_policy_selection.err
#SBATCH --mail-type=ALL

module load conda/py3-latest
source activate mdp_transition_sampling

cd /home/$USER/mdp_transition_sampling
pip install -e .

staskfarm hpc/bandit_policy_selection.txt
