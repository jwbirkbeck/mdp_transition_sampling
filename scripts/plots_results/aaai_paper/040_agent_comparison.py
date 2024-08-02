import scipy.stats
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import os, glob, sys, re
from src.utils.filepaths import results_path_local


results_dir = os.path.join(results_path_local, 'agent_comparison/')
all_filenames = glob.glob(results_dir + "results_[0-9]*.csv")
results = pd.DataFrame()
for filename in all_filenames:
    tmp = pd.read_csv(filename)
    tmp['run'] = int(re.search(r'[0-9]+', filename)[0])
    results = pd.concat((results, tmp))
results = results.reset_index()

minmax_epi = results.groupby('run')['episode'].max().min()
n_steps = minmax_epi * 500

results2 = results.query(f"episode <={minmax_epi}")
# Note that the masking approach does not use cluster size, unlike all other runs
results2.groupby(['cluster_size'])[['lpr_reward', 'wlpr_reward', 'wlpr2_reward', 'mask_reward', 'lpg_reward']].mean().transpose()

results2.groupby('cluster_size')[['lpr_reward', 'wlpr_reward', 'wlpr2_reward', 'mask_reward', 'lpg_reward']].std().transpose()

results2.groupby('run')['episode'].max().min()
