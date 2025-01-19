import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from dragonfly.simulator.aircraft_configuration import AircraftConfiguration

# Initialize aircraft configuration and gimbal
aircraft = AircraftConfiguration()
gimbal = aircraft.gimbal
l_0 = gimbal.get_config()['l_0']
l_1 = gimbal.get_config()['l_1']
l_2 = gimbal.get_config()['l_2']
l_3 = gimbal.get_config()['l_3']


def calculate_positions_3d(gamma_pitch, gamma_yaw):
    L_0, L_1, L_2, _ = gimbal.compute_link_vectors(gamma_yaw)
    A = np.zeros(3)
    B, C, D, E = gimbal.compute_relative_link_points(A, L_0, L_1, L_2)
    x0, y0, z0 = A
    x1, y1, z1 = B
    x2, y2, z2 = C
    x3, y3, z3 = D

    xE, yE, zE = E
    _, L_1_0, L_2_0, _ = gimbal.compute_link_vectors(0)
    R_Apsi_Epsi = np.array([[np.cos(gamma_yaw), -np.sin(gamma_yaw), 0], [np.sin(gamma_yaw), np.cos(gamma_yaw), 0], [0, 0, 1]])
    R_Epsi_Atheta = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
    R = R_Apsi_Epsi @ R_Epsi_Atheta
    A = E - R @ (0.5 * L_2_0 + L_1_0)

    L_0, L_1, L_2, _ = gimbal.compute_link_vectors(gamma_pitch)
    B, C, D, E = gimbal.compute_relative_link_points(A, R @ L_0, R @ L_1, R @ L_2)

    x4, y4, z4 = A
    x5, y5, z5 = B
    x6, y6, z6 = C
    x7, y7, z7 = D

    R_Atheta_Etheta = np.array([[np.cos(gamma_pitch), -np.sin(gamma_pitch), 0], [np.sin(gamma_pitch), np.cos(gamma_pitch), 0], [0, 0, 1]])
    x8, y8, z8 = E
    x9, y9, z9 = E + R @ R_Atheta_Etheta @ np.array([0, 0.025, 0])

    return (x0, y0, z0), (x1, y1, z1), (x2, y2, z2), (x3, y3, z3), (x4, y4, z4), (x5, y5, z5), (x6, y6, z6), (x7, y7, z7), (x8, y8, z8), (x9, y9, z9)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-.025, .125])
ax.set_zlim([-.02, .05])
ax.set_ylim([-.075, .075])
ax.set_xticks(np.empty(0))
ax.set_yticks(np.empty(0))
ax.set_zticks(np.empty(0))
ax.set_aspect('equal')

line_yaw, = ax.plot([], [], [], 'o-', lw=2, color='g', label='Yaw Links')
line_pitch, = ax.plot([], [], [], 'o-', lw=2, color='b', label='Pitch Links')
line_thrust, = ax.plot([], [], [], 'x-', lw=2, color='mediumpurple', label='Thrust Vector')
ax.legend()


def init():
    line_pitch.set_data([], [])
    line_pitch.set_3d_properties([])
    line_yaw.set_data([], [])
    line_yaw.set_3d_properties([])
    line_thrust.set_data([], [])
    line_thrust.set_3d_properties([])
    return line_yaw, line_pitch, line_thrust


def animate(i):
    gamma_pitch = gamma_values_pitch[i]
    gamma_yaw = gamma_values_yaw[i]
    positions = calculate_positions_3d(gamma_pitch, gamma_yaw)

    x_vals_yaw = [positions[0][0], positions[1][0], positions[2][0], positions[3][0]]
    z_vals_yaw = [positions[0][1], positions[1][1], positions[2][1], positions[3][1]]
    y_vals_yaw = [positions[0][2], positions[1][2], positions[2][2], positions[3][2]]

    line_yaw.set_data(x_vals_yaw, y_vals_yaw)
    line_yaw.set_3d_properties(z_vals_yaw)

    x_vals_pitch = [positions[4][0], positions[5][0], positions[6][0], positions[7][0]]
    z_vals_pitch = [positions[4][1], positions[5][1], positions[6][1], positions[7][1]]
    y_vals_pitch = [positions[4][2], positions[5][2], positions[6][2], positions[7][2]]

    line_pitch.set_data(x_vals_pitch, y_vals_pitch)
    line_pitch.set_3d_properties(z_vals_pitch)

    x_vals_thrust = [positions[8][0], positions[9][0]]
    z_vals_thrust = [positions[8][1], positions[9][1]]
    y_vals_thrust = [positions[8][2], positions[9][2]]

    line_thrust.set_data(x_vals_thrust, y_vals_thrust)
    line_thrust.set_3d_properties(z_vals_thrust)

    return line_yaw, line_pitch, line_thrust


# DANCING WITH THE BARS
gamma_sweep = np.concatenate([np.linspace(0, 12.5, 25),
                              np.linspace(12.5, 0, 25),
                              np.linspace(0, -12.5, 25),
                              np.linspace(-12.5, 0, 25)])
gamma_sweep2 = np.concatenate([np.linspace(12.5, -12.5, 50),
                              np.linspace(-12.5, 12.5, 50)])

gamma_hold = np.linspace(0, 0, 25)

gamma_values_yaw = np.radians(np.concatenate([gamma_sweep,
                                              gamma_hold, gamma_hold, gamma_hold, gamma_hold,
                                              gamma_hold,
                                              gamma_sweep,
                                              gamma_sweep,
                                              gamma_hold,
                                              gamma_hold]))

gamma_values_pitch = np.radians(np.concatenate([gamma_hold, gamma_hold, gamma_hold, gamma_hold,
                                                gamma_sweep,
                                                np.linspace(0, 12.5, 25),
                                                gamma_sweep2,
                                                gamma_sweep2,
                                                np.linspace(12.5, 0, 25),
                                                gamma_hold]))

# gamma_values_yaw = np.radians(np.array([-12.5]))
# gamma_values_pitch = np.radians(np.array([0]))
ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(gamma_values_pitch), interval=20, blit=False)

plt.show()
ani.save('thrust_vectoring.gif', writer='pillow')
