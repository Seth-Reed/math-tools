import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from functools import partial

import dragonfly.aircraft_model.forces_and_moments as fam
import dragonfly.functions.math.quaternion as quat

from dragonfly.simulator.aircraft_configuration import AircraftConfiguration

aircraft = AircraftConfiguration()
config = aircraft.config
fam.set_config(config)
m = fam.get_config()['m']
g = fam.get_config()['g']


def initialize_diagram(scale=2.):
    fig, ax = plt.subplots(figsize=(9, 6))
    plt.subplots_adjust(left=0.1, right=0.6, bottom=0.2)
    axis_settings(ax, scale)
    return fig, ax


def axis_settings(ax, scale):
    unit_circle = plt.Circle((0, 0), 1, fill=False, color='k', linestyle='--')
    ax.add_patch(unit_circle)
    ax.set_xlim(-scale, scale)
    ax.set_ylim(-scale, scale)
    ax.set_xticks(np.linspace(-scale, scale, 5))
    ax.set_yticks(np.linspace(-scale, scale, 5))
    ax.invert_yaxis()
    ax.set_xlabel('$\\hat{x}^v$')
    ax.set_ylabel('$\\hat{z}^v$')
    ax.set_title('Forces (X-Z Plane)')
    ax.grid()
    ax.set_aspect('equal', adjustable='box')


def body_to_vehicle_rotation(v, q):
    return quat.rotate(v, q, to_inertial=True)


def add_body_frame_axes(ax, q):
    axes_B = np.array([[1., 0., 0.],
                       [0., 0., 0.],
                       [0., 0., 0.5]])
    for axis_B in axes_B:
        axis_V = body_to_vehicle_rotation(axis_B, q)
        ax.quiver(0, 0, axis_V[0], axis_V[2], color='blue', angles='xy', scale_units='xy', scale=1, width=0.01)


def add_force_vector(ax, F, label=None, scale=1, color='grey'):
    ax.quiver(0, 0, F[0], F[2], angles='xy', scale_units='xy', scale=scale, label=label, color=color)
    if label:
        ax.legend()


def add_velocity_vector(ax, v_inf, alpha, theta):
    q = quat.eul2quat(pitch=theta-alpha, degrees=True)
    v_S = np.array([v_inf, 0., 0.])
    v_V = body_to_vehicle_rotation(v_S, q)
    v_n = v_V[0]
    v_d = v_V[2]

    if v_inf != 0.:
        v_n /= v_inf
        v_d /= v_inf

    x_start = 2*v_n
    y_start = 2*v_d
    x_end = -0.5*v_n
    y_end = -0.5*v_d

    # print("Plotting vector at:", (x_start, y_start, x_end, y_end))
    ax.quiver(x_start, y_start, x_end, y_end, color='black', angles='xy', scale_units='xy', scale=1, width=0.01, label='$v_\\infty$')

    ax.legend()


def create_sliders(fig, ax, x, force_func):
    sliders = {}
    position_offset = 0.8

    # Parameter Sliders
    for param, settings in x.iterrows():
        ax_slider = fig.add_axes([0.7, position_offset, 0.25, 0.05])
        sliders[param] = Slider(ax=ax_slider,
                                label=param,
                                valmin=settings['min'],
                                valmax=settings['max'],
                                valinit=settings['default'])
        sliders[param].on_changed(partial(update_plot, ax=ax, force_func=force_func, sliders=sliders))
        position_offset -= 0.05

    # Scaling Slider
    ax_slider_scale = fig.add_axes([0.25, 0.05, 0.65, 0.03])
    sliders['force_scale'] = Slider(ax=ax_slider_scale,
                                    label='Force Scale',
                                    valmin=1,
                                    valmax=1000,
                                    valinit=m*g)
    sliders['force_scale'].on_changed(partial(update_plot, ax=ax, force_func=force_func, sliders=sliders))

    return sliders


def plot_force_vectors(ax, F, scale=1, colors=None):
    if colors is None:
        colors = {}
    for label, vectors in F.items():
        color = colors.get(label, 'gray')
        for vector in vectors:
            add_force_vector(ax, vector, label=label, scale=scale, color=color)
    ax.legend()


def update_plot(val, ax, force_func, sliders):
    ax.clear()
    axis_settings(ax, scale=2.)
    params = {param: slider.val for param, slider in sliders.items() if param != 'force_scale'}

    theta = params['Pitch Angle']
    q = quat.eul2quat(pitch=theta, degrees=True)
    add_body_frame_axes(ax, q)

    add_velocity_vector(ax, params['Airspeed'], params['Angle of Attack'], params['Pitch Angle'])

    F, colors = force_func(params)
    force_scale = sliders['force_scale'].val
    plot_force_vectors(ax, F, scale=force_scale, colors=colors)


def compute_forces_example(params):
    theta = params['Pitch Angle']
    T = params['Drive Thrust']

    q = quat.eul2quat(pitch=theta, degrees=True)

    F_G_B = fam.gravitational_effects(q=q)
    F_D_B, _ = fam.propulsion_effects(T=T, tau=0., gamma_theta=0., gamma_psi=0.)

    F_G_V = body_to_vehicle_rotation(F_G_B, q)
    F_D_V = body_to_vehicle_rotation(F_D_B, q)
    F_V = F_G_V + F_D_V

    F = pd.DataFrame({
        'Gravity': [F_G_V],
        'Drive': [F_D_V],
        'Sum': [F_V]
    })

    color_map = get_colors(F)
    colors = {
        'Gravity': color_map[0],
        'Drive': color_map[1],
        'Sum': color_map[2]
    }

    return F, colors


def get_colors(F):
    colormap = plt.cm.plasma
    _, num_forces = np.shape(F)
    colors = [colormap(i / num_forces) for i in range(num_forces)]
    return colors


if __name__ == '__main__':
    # Define Input Space
    x = pd.DataFrame({
        'Pitch Angle': [-180, 180, 0],
        'Drive Thrust': [-450., 450., m*g]
    }, index=['min', 'max', 'default']).T

    # Create Plot
    fig, ax = initialize_diagram()
    sliders = create_sliders(fig, ax, x, force_func=compute_forces_example)

    # Plot Body Frame
    q = quat.eul2quat(pitch=x['default']['Pitch Angle'], degrees=True)
    add_body_frame_axes(ax, q)

    # Plot Default Forces
    F, colors = compute_forces_example(x['default'])
    plot_force_vectors(ax, F, scale=int(sliders['force_scale'].val), colors=colors)

    plt.show()





