import scipy.stats
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import os, glob, sys, re
from src.utils.filepaths import results_path_local


results_dir = os.path.join(results_path_local, 'agent_comp2/')
all_filenames = glob.glob(results_dir + "results_[0-9]*.csv")
results = pd.DataFrame()
for filename in all_filenames:
    tmp = pd.read_csv(filename)
    tmp['run'] = int(re.search(r'[0-9*]+(?=.csv)', filename)[0])
    results = pd.concat((results, tmp))
results = results.reset_index()


results.groupby('cluster_size')[['lpr_reward', 'wlpr_reward', 'wlpr2_reward', 'mask_reward', 'lpg_reward']].mean()

mean_data = results.groupby(['episode'])[['lpr_reward', 'wlpr_reward', 'wlpr2_reward', 'mask_reward', 'lpg_reward']].mean().rolling(100).mean()
std_data = results.groupby(['episode'])[['lpr_reward', 'wlpr_reward', 'wlpr2_reward', 'mask_reward', 'lpg_reward']].mean().rolling(100).std()

mean_data.index *= 500
std_data.index *= 500

plt.plot(mean_data.index, mean_data.lpr_reward, label='LPR')
plt.fill_between(mean_data.index, mean_data.lpr_reward-std_data.lpr_reward, mean_data.lpr_reward+std_data.lpr_reward, alpha=0.1)
plt.plot(mean_data.index, mean_data.wlpr_reward, label='CPR')
plt.fill_between(mean_data.index, mean_data.wlpr_reward-std_data.wlpr_reward, mean_data.wlpr_reward+std_data.wlpr_reward, alpha=0.1)
plt.plot(mean_data.index, mean_data.wlpr2_reward, label='Bandit CPR')
plt.fill_between(mean_data.index, mean_data.wlpr2_reward-std_data.wlpr2_reward, mean_data.wlpr2_reward+std_data.wlpr2_reward, alpha=0.1)
plt.plot(mean_data.index, mean_data.mask_reward, label='Mask LRL')
plt.fill_between(mean_data.index, mean_data.mask_reward-std_data.mask_reward, mean_data.mask_reward+std_data.mask_reward, alpha=0.1)
plt.plot(mean_data.index, mean_data.lpg_reward, label='LPG-FTW')
plt.fill_between(mean_data.index, mean_data.lpg_reward-std_data.lpg_reward, mean_data.lpg_reward+std_data.lpg_reward, alpha=0.1)
plt.legend()
plt.xlabel('Steps')
plt.ylabel('Return')
plt.title('Smoothed lifetime average returns, k = 6')
plt.tight_layout()
plt.savefig('agent_comp2.png', dpi=300)
plt.show()

results.groupby(['episode'])[['lpr_reward', 'wlpr_reward', 'wlpr2_reward', 'mask_reward', 'lpg_reward']].mean().rolling(100).mean().plot()

from src.utils.plot_funcs import *

for method in ['lpr_reward', 'wlpr_reward', 'wlpr2_reward', 'mask_reward', 'lpg_reward']:
    results_list = []
    for run in range(0,20):
        tmp = results.query(f'run=={run}')
        results_list.append(list(tmp[method].values))
    create_avg_avg_plot(results_list, window=100, label=method)
plt.legend()
plt.show()