import os, sys, re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob


results_dir = "/opt/project/results/sac_dynalike/"
all_filenames = glob.glob(results_dir + "results_[0-9]*.csv")
results = pd.DataFrame()
for filename in all_filenames:
    tmp = pd.read_csv(filename)
    tmp['run'] = int(re.findall('\d+', filename)[0])
    results = pd.concat((results, tmp))

del results['Unnamed: 0']

running_mean = lambda run, agent, window: np.convolve(results.query('run == ' + str(run))[agent], np.ones(window)/window, mode='valid')

run = 0
window = 100
plt.plot(running_mean(run=run, agent='sac', window=window), label='sac')
plt.plot(running_mean(run=run,agent='sac_dynalike', window=window), label='sac_dynalike')
plt.legend()
plt.show()

for run in results.run.unique():
    plt.plot(running_mean(run=run, agent='sac', window=window), label='sac')
plt.show()

for run in results.run.unique():
    plt.plot(running_mean(run=run, agent='sac_dynalike', window=window), label='sac_dynalike')
plt.show()

def plot_with_err(results, agent, window=100, run=None):
    if run is None:
        plotdata = results.groupby(['episode'])[agent]
        mean = np.convolve(plotdata.mean(), np.ones(window)/window, mode='valid')
        std = np.convolve(plotdata.std(), np.ones(window)/window, mode='valid')
        plt.fill_between(x = range(len(mean)), y1 = mean - std, y2 = mean + std, alpha=0.2)
        plt.plot(mean, label=agent)
    else:
        plotdata = results.query('run == ' + str(run))[agent]
        mean = plotdata.rolling(100).mean()
        std = plotdata.rolling(100).std()
        plt.fill_between(x=range(len(mean)), y1=mean - std, y2=mean + std, alpha=0.2)
        plt.plot(mean, label=agent)

plot_with_err(results, 'sac')
plot_with_err(results, 'sac_dynalike')
plt.xlabel("Training episode")
plt.ylabel("Episode reward")
plt.legend()
plt.tight_layout()
plt.savefig("SAC_vs_SAC_Dynalike.png")
plt.show()

run = 0
run += 1
plot_with_err(results, 'sac', run=run)
plot_with_err(results, 'sac_dynalike', run=run)
plt.legend()
plt.show()


plt.plot(np.convolve(results.groupby(['episode'])['sac'].mean(), np.ones(window)/window, mode='valid'), label='sac')
plt.fill_between()
plt.plot(np.convolve(results.groupby(['episode'])['sac_dynalike'].mean(), np.ones(window)/window, mode='valid'), label='sac_dynalike')
plt.legend()
plt.show()

results.groupby(['episode'])['sac'].mean().plot()
results.groupby(['episode'])['sac_dynalike'].mean().plot()
plt.show()
