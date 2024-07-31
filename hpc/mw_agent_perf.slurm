#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH -p batch
#SBATCH --job-name=mw_agent_perf
#SBATCH --time=60:00:00
#SBATCH --output=mw_agent_perf.out
#SBATCH --error=mw_agent_perf.err
#SBATCH --mail-type=ALL

module load conda/py3-latest
source activate mdp_transition_sampling

cd /home/$USER/mdp_transition_sampling
pip install -e .

staskfarm hpc/mw_agent_perf.txt
