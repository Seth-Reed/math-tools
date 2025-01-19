import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd

import dragonfly.aircraft_model.gimbal as gimbal
from dragonfly.simulator.aircraft_configuration import AircraftConfiguration

aircraft = AircraftConfiguration()
config = aircraft.config
gimbal.set_config(config)

alpha_0 = config['GEOMETRIC PROPERTIES']['gimbal_neutral_angle']

gamma_range = np.radians(np.linspace(-12.5, 12.5, 51))
alpha_range = np.zeros_like(gamma_range)
x_range = np.zeros_like(gamma_range)
y_range = np.zeros_like(gamma_range)
z_range = np.zeros_like(gamma_range)

for i, gamma in enumerate(gamma_range):
    alpha = gimbal.solve_four_bar(gamma)
    alpha_range[i] = alpha

    (x, y, z) = gimbal.gimbal_arm(gamma_theta=gamma,
                                  gamma_psi=0.,
                                  alpha_theta=alpha,
                                  alpha_psi=alpha_0)

    m_to_in = 39.3701
    x_range[i] = x * m_to_in - 39.6
    y_range[i] = (y + 0.05) * m_to_in
    z_range[i] = z * m_to_in

df = pd.DataFrame({
    'Deflection Angle (deg)': np.degrees(gamma_range),
    'Servo Angle (deg)': np.degrees(alpha_range),
    'X Position (in)': x_range,
    'Y Position (in)': y_range,
    'Z Position (in)': z_range
})
# df.to_csv('output.dat', sep='\t', index=False)


fig, ax = plt.subplots()
cmap = cm.coolwarm
sc = ax.scatter(x_range, z_range, c=np.degrees(gamma_range), cmap=cmap)
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('Deflection Angle (deg)')
ax.set_xlabel('X Position (in)')
ax.set_ylabel('Z Position (in)')
ax.grid()
ax.invert_yaxis()
ax.set_aspect('equal', adjustable='box')


plt.show()