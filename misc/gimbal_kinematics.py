import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import time


class Gimbal:
    def __init__(self, alpha=None, links=None):
        if links is None:
            self.l1, self.l2, self.l3, self.l0 = [25., 85., 25., 100.]
        else:
            self.l1, self.l2, self.l3, self.l0 = links

        self.neutral_angle = np.arccos((self.l0 - self.l2) / (2 * self.l3))
        if alpha is None:
            self.alpha = self.neutral_angle
        else:
            self.alpha = alpha

        self.alpha_max = np.arccos((self.l0 ** 2 + self.l1 ** 2 - (self.l2 + self.l3) ** 2) /
                                   (2 * self.l0 * self.l1))
        self.alpha_min = np.arccos((self.l0 ** 2 + (self.l1 + self.l2) ** 2 - self.l3 ** 2) /
                                   (2 * self.l0 * (self.l1 + self.l2)))

    def vector_method(self, X):
        x_eq = self.l1 * np.cos(self.alpha) + self.l2 * np.cos(X[0]) - self.l3 * np.cos(X[1]) - self.l0
        y_eq = self.l1 * np.sin(self.alpha) + self.l2 * np.sin(X[0]) - self.l3 * np.sin(X[1])
        return [x_eq, y_eq]

    def projection_method(self):
        # Calculate the expression
        numerator = (self.l3 * np.sqrt(1 - ((2 * self.l0 * self.l1 * np.cos(
            self.alpha) - self.l1 ** 2 + self.l2 ** 2 + self.l3 ** 2 - self.l0 ** 2) ** 2) / (
                                               4 * self.l2 ** 2 * self.l3 ** 2)) * (
                                 self.l1 * np.cos(self.alpha) - self.l0) + (self.l1 * np.sin(self.alpha) * (
                -2 * self.l0 * self.l1 * np.cos(
            self.alpha) + self.l1 ** 2 + self.l2 ** 2 - self.l3 ** 2 + self.l0 ** 2)) / (2 * self.l2))
        denominator = (self.l1 * self.l3 * np.sin(self.alpha) * np.sqrt(
            1 - ((2 * self.l0 * self.l1 * np.cos(
                self.alpha) - self.l1 ** 2 + self.l2 ** 2 + self.l3 ** 2 - self.l0 ** 2) ** 2) / (
                    4 * self.l2 ** 2 * self.l3 ** 2)) - (self.l1 * np.cos(self.alpha) - self.l0) * (
                               -2 * self.l0 * self.l1 * np.cos(
                           self.alpha) + self.l1 ** 2 + self.l2 ** 2 - self.l3 ** 2 + self.l0 ** 2) / (2 * self.l2))

        return -np.arctan2((self.l3 * np.sqrt(1 - ((2 * self.l0 * self.l1 * np.cos(
            self.alpha) - self.l1 ** 2 + self.l2 ** 2 + self.l3 ** 2 - self.l0 ** 2) ** 2) / (
                                               4 * self.l2 ** 2 * self.l3 ** 2)) * (
                                 self.l1 * np.cos(self.alpha) - self.l0) + (self.l1 * np.sin(self.alpha) * (
                -2 * self.l0 * self.l1 * np.cos(
            self.alpha) + self.l1 ** 2 + self.l2 ** 2 - self.l3 ** 2 + self.l0 ** 2)) / (2 * self.l2)), (self.l1 * self.l3 * np.sin(self.alpha) * np.sqrt(
            1 - ((2 * self.l0 * self.l1 * np.cos(
                self.alpha) - self.l1 ** 2 + self.l2 ** 2 + self.l3 ** 2 - self.l0 ** 2) ** 2) / (
                    4 * self.l2 ** 2 * self.l3 ** 2)) - (self.l1 * np.cos(self.alpha) - self.l0) * (
                               -2 * self.l0 * self.l1 * np.cos(
                           self.alpha) + self.l1 ** 2 + self.l2 ** 2 - self.l3 ** 2 + self.l0 ** 2) / (2 * self.l2)))


if __name__ == '__main__':
    g = Gimbal()
    alphas = np.arange(g.alpha_min, g.alpha_max, 0.01*np.pi/180)
    gammas_vector = np.empty_like(alphas)
    gammas_project = np.empty_like(alphas)

    start_time = time.time()
    for i, alpha in enumerate(alphas):
        g.alpha = alpha
        gammas_vector[i] = fsolve(g.vector_method, np.array([0., np.pi - g.neutral_angle]))[0]
    end_time = time.time()
    vector_time = end_time-start_time
    start_time = time.time()
    for i, alpha in enumerate(alphas):
        g.alpha = alpha
        gammas_project[i] = g.projection_method()
    end_time = time.time()
    project_time = end_time - start_time

    print('vector method time: ', vector_time)
    print('projection method time: ', project_time)
    print('speed improvement:', vector_time/project_time)

    plt.figure()
    plt.plot((alphas-g.neutral_angle) * 180. / np.pi, gammas_vector * 180 / np.pi, label='vector method')
    plt.plot((alphas-g.neutral_angle) * 180. / np.pi, gammas_project * 180 / np.pi, '--', label='projection method')
    plt.legend()
    plt.ylabel('gamma (deg)')
    plt.xlabel('alpha - alpha* (deg)')
    plt.show()