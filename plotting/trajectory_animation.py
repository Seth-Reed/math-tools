import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def trajectory_animation(positions_north, positions_east, altitudes, time, fps=60., axis_limit_buffer=0.5):
    total_frames = len(time)
    max_time = max(time)
    downsample_factor = round(total_frames / (fps * max_time))
    ms_per_frame = 1000 / fps
    positions_north = np.array(positions_north[::downsample_factor])
    positions_east = np.array(positions_east[::downsample_factor])
    altitudes = np.array(altitudes[::downsample_factor])

    # Initialize the figure and axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the entire path with a thin dashed line
    ax.plot(positions_north, positions_east, altitudes, 'k--', linewidth=0.5, label='Trajectory')

    # Initialize a line plot for the animated segment
    position, = ax.plot([], [], [], 'bo', label='Current Position')
    trajectory, = ax.plot([], [], [], 'b-', linewidth=1.5)

    # Setting the limits for the axes
    ax.set_xlim(min(positions_north) - axis_limit_buffer, max(positions_north) + axis_limit_buffer)
    ax.set_ylim(min(positions_east) - axis_limit_buffer, max(positions_east) + axis_limit_buffer)
    ax.set_zlim(min(altitudes) - axis_limit_buffer, max(altitudes) + axis_limit_buffer)
    ax.set_box_aspect([np.ptp(ax.get_xlim()), np.ptp(ax.get_ylim()), np.ptp(ax.get_zlim())])
    ax.invert_yaxis()
    ax.set_xlabel('Position North [m]')
    ax.set_ylabel('Position East [m]')
    ax.set_zlabel('Altitude [m]')
    ax.set_title('3D Vehicle Trajectory')
    ax.legend()
    plt.tight_layout()

    def init():
        position.set_data([], [])
        position.set_3d_properties([])
        trajectory.set_data([], [])
        trajectory.set_3d_properties([])
        return position, trajectory

    def update(frame):
        """Update function for the animation, called for each frame."""
        p_n = positions_north[frame-1]
        p_e = positions_east[frame-1]
        p_h = altitudes[frame-1]

        position.set_data(p_n, p_e)
        position.set_3d_properties(p_h)

        trajectory.set_data(positions_north[:frame], positions_east[:frame])
        trajectory.set_3d_properties(altitudes[:frame])

        return position, trajectory

    # Create the animation

    anim = FuncAnimation(fig, update,
                         frames=len(positions_north),
                         init_func=init,
                         blit=False,
                         interval=ms_per_frame)

    # plt.show()


if __name__ == '__main__':
    if __name__ == '__main__':
        # Helical Path
        t = np.arange(0, 10, 0.001)
        x = np.sin(t)
        y = np.cos(t)
        z = t

        trajectory_animation(x, y, z, t)
