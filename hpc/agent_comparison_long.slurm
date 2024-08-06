#!/bin/bash
#SBATCH --ntasks=600
#SBATCH --ntasks-per-node=40
#SBATCH --nodes=15
#SBATCH -p batch
#SBATCH --job-name=agent_comparison_long
#SBATCH --time=60:00:00
#SBATCH --output=agent_comparison_long.out
#SBATCH --error=agent_comparison_long.err
#SBATCH --mail-type=ALL

module load conda/py3-latest
source activate mdp_transition_sampling

cd /home/$USER/mdp_transition_sampling
pip install -e .

staskfarm hpc/agent_comparison_long.txt
