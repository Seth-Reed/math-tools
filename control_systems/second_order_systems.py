import numpy as np
import control as ct
import matplotlib.pyplot as plt
import matplotlib

from dragonfly.functions.control_systems.step_response_analysis import *


def create_second_order_system(wn, zeta):
    num = [wn ** 2]
    den = [1, 2 * zeta * wn, wn ** 2]
    sys = ct.TransferFunction(num, den)
    return sys


t_0 = 0
t_f = 30
dt = 0.005
time = np.arange(t_0, t_f, dt)

wn = 1.0
zeta = 0.25

sys = create_second_order_system(wn, zeta)
output, U = apply_step_response(sys, time)

rise_time, overshoot, peak_time, settling_time = compute_step_response_performance(output, time, print_results=True)

plot_step_response(output, time, U)
plt.title('Step Response of a Second Order System')
# plt.show()

# FREQUENCY RESPONSE
gain, phase, frq = ct.bode(sys, plot=False)
mag = 20*np.log10(gain)
phase = np.degrees(phase)

fig, ax = plt.subplots(2, 1, figsize=(8, 5))
ax[0].plot(frq, mag, 'b')
ax[0].axhline(-3, color='k', linestyle='--', linewidth=1, label='-3 dB line')
ax[0].axvline(wn, color='r', linestyle='-.', linewidth=1, label=f'$\\omega_n=${wn:.3f} rad/s')
bw_label = '$\\omega_{bw}$'
wb = ct.bandwidth(sys)
ax[0].axvline(wb, color='g', linestyle='-.', linewidth=1, label=f'{bw_label}={wb:.3f} rad/s')
ax[0].set_ylabel('Magnitude (dB)')
ax[0].legend()

ax[1].plot(frq, phase, 'b')
ax[1].axvline(wb, color='g', linestyle='-.', linewidth=1)
ax[1].axvline(wn, color='r', linestyle='-.', linewidth=1)
ax[1].set_ylabel('Phase (deg)')
ax[1].set_xlabel('Frequency (rad/s)')

for a in ax:
    a.grid()
    a.set_xscale('log')

ax[0].set_title('Bode Plot of a Second Order System')
plt.tight_layout()


# Generate frequency range
frq = np.logspace(-2, 2, 5000)
data = ct.nyquist_response(sys, frq, plot=False)
real, imag = np.real(data.response), np.imag(data.response)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(real, imag, 'b', label='Nyquist Curve')

ax.axhline(0, color='k', linestyle='-', linewidth=0.5)
ax.axvline(0, color='k', linestyle='-', linewidth=0.5)
ax.scatter([-1], [0], color='red', s=50, zorder=5, label='-1 Point')
unit_circle = matplotlib.patches.Circle((0, 0), 1, edgecolor='black', linestyle='--', linewidth=1, fill=False)
ax.add_patch(unit_circle)

ax.set_title('Nyquist Plot')
ax.set_xlabel('Real Part')
ax.set_ylabel('Imaginary Part')
ax.set_aspect('equal', 'box')
ax.legend(loc='upper right')
ax.grid()

plt.tight_layout()


plt.show()
