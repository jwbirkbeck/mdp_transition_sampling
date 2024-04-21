#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH -p batch
#SBATCH --job-name=sopr_on_cont_nonstat
#SBATCH --time=60:00:00
#SBATCH --output=sopr_on_cont_nonstat.out
#SBATCH --error=sopr_on_cont_nonstat.err
#SBATCH --mail-type=ALL

module load conda/py3-latest
source activate mdp_transition_sampling

cd /home/$USER/mdp_transition_sampling
pip install -e .

staskfarm hpc/sopr_on_cont_nonstat.txt
