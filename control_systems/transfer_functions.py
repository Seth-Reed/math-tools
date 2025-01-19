import control as ct


def loop_transfer_function(G, C, print_minreal_settings=False):
    return ct.minreal(G * C, verbose=print_minreal_settings)


def closed_loop_transfer_function(L, print_minreal_settings=False):
    return ct.minreal(L / (1 + L), verbose=print_minreal_settings)


def pid_controller(k_p, k_i, k_d, tau=0.):
    C_p = ct.tf([k_p], [1])
    C_i = ct.tf([k_i], [1, 0])
    C_d = ct.tf([k_d, 0], [tau, 1])
    # print(C_p, C_i, C_d)
    # print(C_p + C_i + C_d)
    return C_p + C_i + C_d


def error_reference_transfer_function(L, print_minreal_settings=False):
    return ct.minreal(1 / (1 + L), verbose=print_minreal_settings)


def error_noise_transfer_function(L, print_minreal_settings=False):
    return ct.minreal(L / (1 + L), verbose=print_minreal_settings)


def error_disturbance_transfer_function(G, L, print_minreal_settings=False):
    return ct.minreal(G / (1 + L), verbose=print_minreal_settings)


def error_transfer_functions(G, C, print_minreal_settings=False):
    L = loop_transfer_function(G, C, print_minreal_settings)

    E_R = error_reference_transfer_function(L, print_minreal_settings)
    E_N = error_noise_transfer_function(L, print_minreal_settings)
    E_D = error_disturbance_transfer_function(L, print_minreal_settings)

    return E_R, E_N, E_D


