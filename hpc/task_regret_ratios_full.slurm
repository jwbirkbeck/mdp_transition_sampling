#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=17
#SBATCH -p batch
#SBATCH --job-name=task_regret_ratios_full
#SBATCH --time=60:00:00
#SBATCH --output=task_regret_ratios_full.out
#SBATCH --error=task_regret_ratios_full.err
#SBATCH --mail-type=ALL

source ~/venvs/mdp_transition_sampling/bin/activate

cd /home/$USER/mdp_transition_sampling
pip install -e .

staskfarm hpc/task_regret_ratios_full.txt
