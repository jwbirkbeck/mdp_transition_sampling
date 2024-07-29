#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH -p batch
#SBATCH --job-name=simplegrid_sampling_comp
#SBATCH --time=60:00:00
#SBATCH --output=simplegrid_sampling_comp.out
#SBATCH --error=simplegrid_sampling_comp.err
#SBATCH --mail-type=ALL

module load conda/py3-latest
source activate mdp_transition_sampling

cd /home/$USER/mdp_transition_sampling
pip install -e .

staskfarm hpc/simplegrid_sampling_comp.txt
