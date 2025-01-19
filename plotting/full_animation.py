import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from dragonfly.functions.math.quaternion import rotate


def full_animation(time, positions_north, positions_east, altitudes, quaternions, scale=1., fps=60., speed=1):
    # Timing Calculations
    total_frames = len(time)
    max_time = max(time)
    downsample_factor = round(total_frames / (fps * max_time)) * speed
    ms_per_frame = 1000 / fps

    # Downsample data
    time = time[::downsample_factor]
    positions_north = np.array(positions_north[::downsample_factor])
    positions_east = np.array(positions_east[::downsample_factor])
    altitudes = np.array(altitudes[::downsample_factor])
    quaternions = np.array(quaternions[::downsample_factor])

    # Initialize figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Set plot limits and labels
    buffer = scale
    ax.set_xlim([min(positions_north) - buffer, max(positions_north) + buffer])
    ax.set_ylim([min(positions_east) - buffer, max(positions_east) + buffer])
    ax.set_zlim([min(altitudes) - buffer, max(altitudes) + buffer])
    ax.set_box_aspect([np.ptp(ax.get_xlim()), np.ptp(ax.get_ylim()), np.ptp(ax.get_zlim())])
    ax.invert_yaxis()
    ax.set_xlabel('North [m]')
    ax.set_ylabel('East [m]')
    ax.set_zlabel('Altitude [m]')
    # ax.grid(False)
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_zticks([])

    # Plot entire path
    ax.plot(positions_north, positions_east, altitudes, '--', color='k', linewidth=1.5, label='Trajectory')

    # Initialize the position marker and trajectory
    position_marker, = ax.plot([], [], [], 'bo', label='Current Position')
    trajectory_line, = ax.plot([], [], [], 'v-', linewidth=1.5)

    # Body frame axes initialization
    vectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    axis_labels = ['$\\hat{x}^b$', '$\\hat{y}^b$', '$\\hat{z}^b$']
    colors = ['r', 'b', 'g']
    axis_lines = [ax.plot([0, v[0]], [0, v[1]], [0, v[2]], color=colors[i],  label=f"{axis_labels[i]}")[0] for i, v in enumerate(vectors)]
    # plt.legend()
    # Animation update function
    def update(frame):
        # Update vehicle position and trajectory
        p_n = positions_north[frame]
        p_e = positions_east[frame]
        h = altitudes[frame]
        position_marker.set_data(p_n, p_e)
        position_marker.set_3d_properties(h)
        # trajectory_line.set_data(positions_north[:frame], positions_east[:frame])
        # trajectory_line.set_3d_properties(altitudes[:frame])

        # Update body frame axes based on quaternion
        q = quaternions[frame]
        for i, line in enumerate(axis_lines):
            rotated_vector = rotate(vectors[i], q, to_inertial=True)
            end_point = rotated_vector * scale + np.array([p_n, p_e, -h])
            line.set_data([p_n, end_point[0]], [p_e, end_point[1]])
            line.set_3d_properties([h, -end_point[2]])

        return [position_marker, trajectory_line] + axis_lines

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(time), blit=False, interval=ms_per_frame)
    return ani






