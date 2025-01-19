import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from dragonfly.simulator.aircraft_configuration import AircraftConfiguration

aircraft = AircraftConfiguration()
gimbal = aircraft.gimbal
l_0 = gimbal.get_config()['l_0']
l_1 = gimbal.get_config()['l_1']
l_2 = gimbal.get_config()['l_2']
l_3 = gimbal.get_config()['l_3']


def calculate_positions(gamma):
    alpha = gimbal.solve_four_bar(angle=gamma)

    L_0, L_1, L_2, _ = gimbal.compute_link_vectors(gamma)
    A = np.zeros(3)
    _, B, C, D, _ = gimbal.compute_relative_link_points(L_0, L_1, L_2, A)
    x0, y0 = A[:2]
    x1, y1 = B[:2]
    x2, y2 = C[:2]
    x3, y3 = D[:2]

    return (x0, y0), (x1, y1), (x2, y2), (x3, y3)


fig, ax = plt.subplots()
ax.set_xlim(-.025, .125)
ax.set_ylim(-.02, .040)
ax.set_xticks(np.empty(0))
ax.set_yticks(np.empty(0))
ax.set_aspect('equal')
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)


def init():
    line.set_data([], [])
    return line,


def animate(i):
    gamma = gamma_values[i]
    positions = calculate_positions(gamma)
    x_vals = [pos[0] for pos in positions]
    y_vals = [pos[1] for pos in positions]
    line.set_data(x_vals, y_vals)
    return line,


# gamma_values = np.radians(np.concatenate([np.linspace(-12.5, 12.5, 100), np.linspace(12.5, -12.5, 100)]))
gamma_values = np.array([np.radians(-10)])
ani = animation.FuncAnimation(fig, animate, init_func=init,
                              frames=len(gamma_values), interval=25, blit=True)
ani.save('four_bar_mechanism.gif', writer='pillow')
plt.show()
