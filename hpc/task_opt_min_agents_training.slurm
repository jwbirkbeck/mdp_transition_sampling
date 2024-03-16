#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=14
#SBATCH -p batch
#SBATCH --job-name=task_opt_min_agents_training
#SBATCH --time=60:00:00
#SBATCH --output=task_opt_min_agents_training.out
#SBATCH --error=task_opt_min_agents_training.err
#SBATCH --mail-type=ALL

module load conda/py3-latest
source activate mdp_transition_sampling

cd /home/$USER/mdp_transition_sampling
pip install -e .

staskfarm hpc/task_opt_min_agents_training.txt
