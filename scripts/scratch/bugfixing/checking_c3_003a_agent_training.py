import glob, sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.utils.consts import task_pool_10
import pickle

results_dir = "/opt/project/results/c3_003a_train_agents/"

all_filenames = glob.glob(os.path.join(results_dir, "eval_[0-9]*.csv"))
agent_results = pd.DataFrame()

tmp_ind = 0
for filename in all_filenames:
    tmp = pd.read_csv(filename)
    tmp['run'] = tmp_ind
    tmp_ind += 1
    agent_results = pd.concat((agent_results, tmp))

run = 0
agent_results.query(f"run == {run}").groupby('task')['rewards'].plot()
plt.show()
