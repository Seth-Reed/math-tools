import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from matplotlib.cm import coolwarm

from dragonfly.functions.math.quaternion import normalize, rotate
import dragonfly.aircraft_model.forces_and_moments as fam


def rotation_animation(quaternions, controls, time, config=None, fps=60., scale=1.):
    total_frames = len(time)
    max_time = max(time)
    downsample_factor = round(total_frames / (fps * max_time))
    ms_per_frame = 1000 / fps
    quaternions = np.array(quaternions[::downsample_factor])
    controls = np.array(controls[::downsample_factor])

    if config:
        fam.set_config(config)

    fig, ax = create_plot(scale)

    vectors = scale * np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # X, Y, Z body frame
    axis_labels = ['$\\hat{x}^b$', '$\\hat{y}^b$', '$\\hat{z}^b$']
    colors = ['r', 'b', 'g']
    lines = [ax.plot([0, v[0]], [0, v[1]], [0, v[2]], color=colors[i], label=f"{axis_labels[i]}")[0] for i, v in enumerate(vectors)]

    # thrust_line = ax.plot([0, 0], [0, 0], [0, 0], '-', color='grey', label='$\\vec{F}_D$ (Color$\\propto\\tau$)')[0]
    thrust_line = ax.plot([0, 0], [0, 0], [0, 0], '-', color='grey')[0]

    norm = Normalize(vmin=-3, vmax=3)

    def update(num, quaternions, lines, controls, thrust_line, norm):
        # Normalize and apply rotation to vectors
        q_normalized = normalize(quaternions[num])
        rotated_vectors = np.array([rotate(vector, q_normalized, to_inertial=True) for vector in vectors])

        # Update lines
        for line, vec in zip(lines, rotated_vectors):
            line.set_data([0, vec[0]], [0, vec[1]])
            line.set_3d_properties([0, vec[2]])

        # Update thrust vector
        thrust_vector = compute_thrust_vector(quaternions[num], controls[num])
        thrust_line.set_data([0, thrust_vector[0]], [0, thrust_vector[1]])
        thrust_line.set_3d_properties([0, thrust_vector[2]])
        # print("Thrust Vector:", thrust_vector)  # Debugging output

        # Update color based on tau
        tau_value = controls[num][1]
        color = coolwarm(norm(tau_value))
        thrust_line.set_color(color)

    animation = FuncAnimation(fig,
                              update,
                              frames=len(quaternions),
                              fargs=(quaternions, lines, controls, thrust_line, norm),
                              interval=ms_per_frame)
    plt.legend(loc='upper right')
    plt.show()


def create_plot(scale):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-scale, scale])
    ax.set_ylim([-scale, scale])
    ax.set_zlim([-scale, scale])
    ax.set_xticks(np.linspace(-scale, scale, 3))
    ax.set_yticks(np.linspace(-scale, scale, 3))
    ax.set_zticks(np.linspace(-scale, scale, 3))
    ax.set_xlabel('$\\hat{x}^v$')
    ax.set_ylabel('$\\hat{y}^v$')
    ax.set_zlabel('$\\hat{z}^v$')
    ax.invert_yaxis()
    ax.invert_zaxis()
    return fig, ax


def compute_thrust_vector(quaternion, controls):
    q = quaternion
    T, tau, gamma_theta, gamma_psi = controls
    mg = fam.get_config()['m'] * fam.get_config()['g']

    F_D_B, _ = fam.propulsion_effects(T, tau, gamma_theta, gamma_psi)
    F_D_V = rotate(F_D_B, q, to_inertial=True)
    F_D_V_norm = F_D_V / mg

    return F_D_V_norm

