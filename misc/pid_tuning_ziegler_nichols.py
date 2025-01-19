def ziegler_nichols(k_u, t_u):
    k_p = 0.2 * k_u
    t_i = t_u / 2.
    t_d = t_u / 3.

    if t_i != 0:
        k_i = k_p / t_i
    else:
        k_i = 0.
    k_d = k_p * t_d

    return k_p, k_i, k_d


if __name__ == '__main__':
    k_p, k_i, k_d = ziegler_nichols(k_u=100., t_u=6.78)

    print('k_p = ', k_p)
    print('k_i = ', k_i)
    print('k_d = ', k_d)

