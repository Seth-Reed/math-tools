import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation


def rotate_xyz(q, x, y, z):
    quaternion = np.array([q[1], q[2], q[3], q[0]])
    rotation = Rotation.from_quat(quaternion)
    rotated_x = rotation.apply(x)
    rotated_y = rotation.apply(y)
    rotated_z = rotation.apply(z)
    return rotated_x, rotated_y, rotated_z


if __name__ == "__main__":
    q0 = np.array([np.sqrt(2)/2., 0., -np.sqrt(2)/2., 0.])
    qf = np.array([0.5, -0.5, -0.5, 0.5])
    quaternion_data = [q0, qf]
    quaternion_df = pd.DataFrame(quaternion_data, columns=['q0', 'q1', 'q2', 'q3'])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    initial_x, initial_y, initial_z = rotate_xyz(q0, np.array([1., 0., 0.]), np.array([0., 1., 0.]), np.array([0., 0., 1.]))
    x_quiver = ax.quiver(0, 0, 0, initial_x[0], initial_x[1], initial_x[2], color='r', label='X-axis')
    y_quiver = ax.quiver(0, 0, 0, initial_y[0], initial_y[1], initial_y[2], color='g', label='Y-axis')
    z_quiver = ax.quiver(0, 0, 0, initial_z[0], initial_z[1], initial_z[2], color='b', label='Z-axis')
    frame_indicator = ax.text(0.0, 0.0, 0.0, '', transform=ax.transAxes, fontsize=12, color='Black')

    print(initial_x)
    print(initial_y)
    print(initial_z)

    def update_frame(frame):
        quaternion = quaternion_df.iloc[frame].to_numpy()
        rotated_x, rotated_y, rotated_z = rotate_xyz(quaternion, np.array([1., 0., 0.]), np.array([0., 1., 0.]), np.array([0., 0., 1.]))
        x_quiver.set_segments([[np.zeros(3), rotated_x]])
        y_quiver.set_segments([[np.zeros(3), rotated_y]])
        z_quiver.set_segments([[np.zeros(3), rotated_z]])
        frame_indicator.set_text(f'Frame {frame}/{len(quaternion_df) - 1}')
        return x_quiver, y_quiver, z_quiver

    num_frames = len(quaternion_df)
    animation = FuncAnimation(fig, update_frame, frames=num_frames, blit=False, interval=2000)

    ax.set_box_aspect([np.ptp(ax.get_xlim()), np.ptp(ax.get_ylim()), np.ptp(ax.get_zlim())])
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.invert_zaxis()
    ax.invert_yaxis()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    plt.show()
