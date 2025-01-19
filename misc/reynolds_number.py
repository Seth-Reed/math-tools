from dragonfly.aircraft_model.aero import atmospheric_properties


def compute_reynolds_number(v, rho, L, mu=1.716e-5):
    """
    Computes the reynolds number of a flow given velocity, air density, characteristic length, and dynamic viscosity
    Assumes consistent units for proper dimensional analysis.

    Parameters:
        v (float): freestream velocity (m/s)
        rho (float): air density (kg/m^3)
        L (float): characteristic length, often the airfoil chord length (m)
        mu (float): dynamic fluid viscosity (N*s/m^2), defaults to 1.716e-5 for air at 273 K
    """
    return rho * v * L / mu


def compute_dynamic_viscosity(T):
    """
    Computes the dynamic velocity of air given the current temperature in (K) using the Sutherland formula.

    Parameter:
        T (float): air temperature (K)

    Return:
        (float): dynamic viscosity (N*s/m^2)
    """
    T_0 = 273.          # standard temperature (K)
    mu_0 = 1.716e-5     # standard dynamic viscosity (N*s/m^2)
    S_mu = 111.         # effective temperature / Sutherland constant (K)
    return mu_0 * (T / T_0) ** 1.5 * ((T_0 + S_mu) / (T + S_mu))


if __name__ == '__main__':
    # Low Reynolds number computations
    # design inputs
    v = 26.         # freestream velocity (m/s)
    h = 0           # altitude (m)
    L = 0.387       # chord (m)

    # intermediate variables
    rho, T, _ = atmospheric_properties(h=h)
    T += 273.  # convert to Kelvin
    mu = compute_dynamic_viscosity(T)

    # Reynolds number
    Re = compute_reynolds_number(v, rho, L, mu)

    print(f'Freestream velocity = {v} m/s')
    print(f'Altitude = {h} m')
    print(f'Characteristic Length = {L} m')
    print()
    print(f'Air density = {rho} kg/m^3')
    print(f'Air temperature = {T} K')
    print(f'Air dynamic viscosity = {mu} N*s/m^2')
    print()
    print(f'Reynolds Number = {Re}')