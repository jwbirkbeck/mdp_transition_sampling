import numpy as np
import matplotlib.pyplot as plt


def create_avg_plot(results, window=10, label=None):
    averages = []
    for ind in range(len(results) - window + 1):
        averages.append(np.mean(results[ind:ind + window]))
    plt.plot(averages, label=label)
    plt.figtext(0.05, 0.01, "Average across last " + str(window) + " episodes")


def create_avg_avg_plot(results_nested_list, window=100, label=None, x_axis_multiplier=None):
    rolling_averages = []
    num_runs = len(results_nested_list)
    run_length = len(results_nested_list[0])  # all run lengths equal the first run length
    for i in range(num_runs):
        run = results_nested_list[i]
        roll_avg = []
        for ind in range(1, run_length - window + 1):
            start_ind = max(0, ind - window)
            roll_avg.append(np.mean(run[start_ind:ind]))
        rolling_averages.append(roll_avg)

    averages = []
    stds = []
    run_length = len(rolling_averages[0])
    for i in range(run_length):
        avg = 0
        for j in range(num_runs):
            avg += rolling_averages[j][i] / float(num_runs)
        averages.append(avg)
        avgs = []
        for j in range(num_runs):
            avgs.append(rolling_averages[j][i])
        stds.append(np.std(avgs))

    std_high = [a + b for a, b in zip(averages, stds)]
    std_low = [a - b for a, b in zip(averages, stds)]
    if x_axis_multiplier is not None:
        x_range = range(0, x_axis_multiplier * len(averages), x_axis_multiplier)
    else:
        x_range = range(0, len(averages))

    plt.plot(x_range, averages, label=label)
    plt.fill_between(x_range, std_low, std_high, alpha=0.2)
    # plt.figtext(0.05, 0.01, "Average across " + str(num_runs) + " runs' average total reward of last " + str(window) + " episodes")


