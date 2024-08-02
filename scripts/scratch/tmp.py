import pickle
import os

with open('/opt/project/scripts/local_experiments/mw_w1_dists.pkl', 'rb') as file:
    tmp = pickle.load(file)

import matplotlib.pyplot as plt

plt.scatter(tmp.ns_test_ind, tmp.w1)
plt.show()