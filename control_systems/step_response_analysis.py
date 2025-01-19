import numpy as np
import control as ct
import matplotlib.pyplot as plt


def apply_step_response(sys, time, step_magnitude=1):
    U = np.ones_like(time) * step_magnitude
    _, output = ct.forced_response(sys, time, U)
    return output, U


def plot_step_response(output, time, U, color=None):
    plt.figure(figsize=(8, 3))
    plt.plot(time, U, 'r-.', label='Reference Signal')
    if color:
        plt.plot(time, output, color=color, label='Output Signal')
    else:
        plt.plot(time, output, 'b', label='Output Signal')

    plt.xlabel('$t$ (s)')
    plt.ylabel('$y(t)$')
    plt.xlim([time[0], time[-1]])
    plt.grid()

    rise_time, overshoot, _, _ = compute_step_response_performance(output, time)
    rise_start_index = np.where(output >= 0.1 * U[0])[0][0]
    rise_end_index = np.where(output >= 0.9 * U[0])[0][0]
    rise_start = time[rise_start_index]
    rise_end = time[rise_end_index]
    plt.axvline(x=rise_start, color='k', linestyle=':', label=f'Rise Time = {rise_time:.3f}s', linewidth=1)
    plt.axvline(x=rise_end, color='k', linestyle=':', linewidth=1)
    plt.axhline(y=U[-1] * (1 + overshoot / 100), color='k', linestyle='--',
                label=f'Percent Overshoot = {overshoot:.2f}%', linewidth=1)

    plt.legend()


def compute_rise_time(output, time):
    final_value = output[-1]
    rise_start_value = 0.1 * final_value
    rise_end_value = 0.9 * final_value
    start_index = np.where(output >= rise_start_value)[0][0]
    end_index = np.where(output >= rise_end_value)[0][0]
    rise_time = time[end_index] - time[start_index]
    return rise_time


def compute_percent_overshoot(output):
    final_value = output[-1]
    peak_value = np.max(output)
    percent_overshoot = ((peak_value - final_value) / final_value) * 100
    return percent_overshoot


def compute_peak_time(output, time):
    peak_index = np.argmax(output)
    peak_time = time[peak_index]
    return peak_time


def compute_settling_time(output, time):
    final_value = output[-1]

    lower_bound = final_value * 0.95
    upper_bound = final_value * 1.05

    settle_index = np.where((output >= lower_bound) & (output <= upper_bound))[0]

    for i in range(len(settle_index) - 1):
        if settle_index[i + 1] - settle_index[i] > 1:
            settle_index = settle_index[i + 1:]
            break

    if not settle_index.any():
        settling_time = None
    else:
        settling_time = time[settle_index[0]]

    return settling_time


def compute_step_response_performance(output, time, print_results=False):
    rise_time = compute_rise_time(output, time)
    overshoot = compute_percent_overshoot(output)
    peak_time = compute_peak_time(output, time)
    settling_time = compute_settling_time(output, time)

    if print_results:
        print(f'Rise Time:      {rise_time:.3f} s')
        print(f'Overshoot:      {overshoot:.3f} %')
        print(f'Peak Time:      {peak_time:.3f} s')
        print(f'Settling Time:  {rise_time:.3f} s')

    return rise_time, overshoot, peak_time, settling_time
