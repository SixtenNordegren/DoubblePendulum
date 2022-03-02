import matplotlib.pyplot as plt
import numpy as np
import os
from numpy import sin, cos

# Problem parameters
g = 9.81
l_1 = 1
l_2 = l_1
m_1 = 1
m_2 = m_1

# Numerical parameters
a = 0.0
b = 1.0
h = 0.1

# For bounded region.
# N = 1000
# h = (b - a) / N


def alpha_1(phi, theta):
    return (l_2 / l_1) * (m_2 / (m_1 + m_2)) * cos(phi - theta)


def alpha_2(phi, theta):
    return (l_1 / l_2) * cos(phi - theta)


def f_1(phi, theta, phi_dot, theta_dot):
    return (
        -(l_2 / l_1) * (m_2 / (m_1 + m_2)) * theta_dot ** 2 * sin(phi - theta)
        - g * sin(phi) / l_1
    )


def f_2(phi, theta, phi_dot, theta_dot):
    return (l_1 / l_2) * phi_dot ** 2 * sin(phi - theta) - g * sin(theta) / l_2


def g_1(phi, theta, phi_dot, theta_dot):
    return (
        f_1(phi, theta, phi_dot, theta_dot)
        - alpha_1(phi, theta) * f_2(phi, theta, phi_dot, theta_dot)
    ) / (1 - alpha_1(phi, theta) * alpha_2(phi, theta))


def g_2(phi, theta, phi_dot, theta_dot):
    return (
        -alpha_2(phi, theta) * f_1(phi, theta, phi_dot, theta_dot)
        + f_2(phi, theta, phi_dot, theta_dot)
    ) / (1 - alpha_1(phi, theta) * alpha_2(phi, theta))


def f(r, t):
    """
    r(omega_1, omega_2, omega_1_dot, omega_2_dot, t)
    g(phi, theta, phi_dot, omega_dot)
    """
    omega_1 = r[2]
    omega_2 = r[3]

    omega_1_dot = g_1(r[0], r[1], r[2], r[3])
    omega_2_dot = g_2(r[0], r[1], r[2], r[3])

    return np.array([omega_1, omega_2, omega_1_dot, omega_2_dot])


def RK(r, t):
    """This is the runge-kutta procdure with
    the standard 4 iterations."""

    k_1 = h * f(r, t)
    k_2 = h * f(r + 0.5 * k_1, t + 0.5 * h)
    k_3 = h * f(r + 0.5 * k_2, t + 0.5 * h)
    k_4 = h * f(r + k_3, t + h)

    r += (k_1 + 2 * k_2 + 2 * k_3 + k_4) / 6
    t = t
    return r, t


def plot(data):
    plt.plot(data)
    plt.show()


def pendulum(r, N=1000, a=0, b=100):
    res = []

    for i in np.linspace(a, b, num=N):

        res.append(np.array(r))
        r, t = RK(r, i)
    if os.path.isfile("results.csv"):
        os.remove("results.csv")
    np.savetxt("results.csv", np.array(res), delimiter=",")
    return res



def main():
    r = np.array([0.5, 0.5, 0, 0])
    epsilon = 0.5
    res = np.array(pendulum(r))

    ress = np.array(pendulum(r + epsilon))

    res = np.array(res)
    plt.scatter(res[:,0], res[:, 2])
    plt.show()

    plt.scatter(ress[:,0], res[:, 2])
    plt.show()

if __name__ == "__main__":
    main()
