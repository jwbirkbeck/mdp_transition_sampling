#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH -p batch
#SBATCH --job-name=agent_comparison
#SBATCH --time=60:00:00
#SBATCH --output=agent_comparison.out
#SBATCH --error=agent_comparison.err
#SBATCH --mail-type=ALL

module load conda/py3-latest
source activate mdp_transition_sampling

cd /home/$USER/mdp_transition_sampling
pip install -e .

staskfarm hpc/agent_comparison.txt
