#!/bin/bash
#SBATCH --ntasks=20
#SBATCH --nodes=1
#SBATCH -p batch
#SBATCH --job-name=agent_comparison_long_small
#SBATCH --time=60:00:00
#SBATCH --output=agent_comparison_long_small.out
#SBATCH --error=agent_comparison_long_small.err
#SBATCH --mail-type=ALL

module load conda/py3-latest
source activate mdp_transition_sampling

cd /home/$USER/mdp_transition_sampling
pip install -e .

staskfarm hpc/agent_comparison_long_small.txt
