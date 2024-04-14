#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH -p batch
#SBATCH --job-name=sac_dynalike
#SBATCH --time=60:00:00
#SBATCH --output=sac_dynalike.out
#SBATCH --error=sac_dynalike.err
#SBATCH --mail-type=ALL

module load conda/py3-latest
source activate mdp_transition_sampling

cd /home/$USER/mdp_transition_sampling
pip install -e .

staskfarm hpc/sac_dynalike.txt
