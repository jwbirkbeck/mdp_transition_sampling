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
    tmp['run'] = int(re.search(r'[0-9]+', filename)[0])
    results = pd.concat((results, tmp))
results = results.reset_index()


results.groupby('cluster_size')[['lpr_reward', 'wlpr_reward', 'wlpr2_reward', 'mask_reward', 'lpg_reward']].mean()



# plotdata = results.query('cluster_size == 6')
# sum(plotdata.lpr_reward) / plotdata.shape[0]
# sum(plotdata.wlpr_reward) / plotdata.shape[0]
# sum(plotdata.wlpr2_reward) / plotdata.shape[0]
# sum(plotdata.mask_reward) / plotdata.shape[0]
# sum(plotdata.lpg_reward) / plotdata.shape[0]
