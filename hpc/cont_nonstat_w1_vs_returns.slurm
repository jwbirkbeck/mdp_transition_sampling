#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH -p batch
#SBATCH --job-name=cont_nonstat
#SBATCH --time=60:00:00
#SBATCH --output=cont_nonstat.out
#SBATCH --error=cont_nonstat.err
#SBATCH --mail-type=ALL

module load conda/py3-latest
source activate mdp_transition_sampling

cd /home/$USER/mdp_transition_sampling
pip install -e .

staskfarm hpc/cont_nonstat_w1_vs_returns.txt
