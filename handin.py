import matplotlib.pyplot as plt
import numpy as np
import os
from numpy import sin, cos

# Problem parameters
g = 1
l_1 = 2
l_2 = 1
m_1 = 3
m_2 = 1

# Numerical parameters
a = 0.0
b = 1.0
h = 0.1

# For bounded region.
# N = 1000
# h = (b - a) / N


def alpha_1(phi, theta, P):
    l_1 = P[0]
    l_2 = P[1]
    m_1 = P[2]
    m_2 = P[3]
    return (l_2 / l_1) * (m_2 / (m_1 + m_2)) * cos(phi - theta)


def alpha_2(phi, theta, P):
    l_1 = P[0]
    l_2 = P[1]
    m_1 = P[2]
    m_2 = P[3]
    return (l_1 / l_2) * cos(phi - theta)


def f_1(phi, theta, phi_dot, theta_dot, P):
    l_1 = P[0]
    l_2 = P[1]
    m_1 = P[2]
    m_2 = P[3]
    return (
        -(l_2 / l_1) * (m_2 / (m_1 + m_2)) * theta_dot ** 2 * sin(phi - theta)
        - g * sin(phi) / l_1
    )


def f_2(phi, theta, phi_dot, theta_dot, P):
    l_1 = P[0]
    l_2 = P[1]
    m_1 = P[2]
    m_2 = P[3]
    return (l_1 / l_2) * phi_dot ** 2 * sin(phi - theta) - g * sin(theta) / l_2


def g_1(phi, theta, phi_dot, theta_dot, P):
    return (
        f_1(phi, theta, phi_dot, theta_dot, P)
        - alpha_1(phi, theta, P) * f_2(phi, theta, phi_dot, theta_dot, P)
    ) / (1 - alpha_1(phi, theta, P) * alpha_2(phi, theta, P))


def g_2(phi, theta, phi_dot, theta_dot, P):
    return (
        -alpha_2(phi, theta, P) * f_1(phi, theta, phi_dot, theta_dot, P)
        + f_2(phi, theta, phi_dot, theta_dot, P)
    ) / (1 - alpha_1(phi, theta, P) * alpha_2(phi, theta, P))


def f(r, t, P=[1, 1, 1, 1]):
    """
    r(omega_1, omega_2, omega_1_dot, omega_2_dot, t)
    g(phi, theta, phi_dot, omega_dot)
    """
    omega_1 = r[2]
    omega_2 = r[3]

    omega_1_dot = g_1(r[0], r[1], r[2], r[3], P)
    omega_2_dot = g_2(r[0], r[1], r[2], r[3], P)

    return np.array([omega_1, omega_2, omega_1_dot, omega_2_dot])


def RK(r, t, P):
    """This is the runge-kutta procdure with
    the standard 4 iterations."""

    k_1 = h * f(r, t, P)
    k_2 = h * f(r + 0.5 * k_1, t + 0.5 * h, P)
    k_3 = h * f(r + 0.5 * k_2, t + 0.5 * h, P)
    k_4 = h * f(r + k_3, t + h, P)

    r += (k_1 + 2 * k_2 + 2 * k_3 + k_4) / 6
    t = t
    return r, t


def plot(data):
    plt.plot(data)
    plt.show()


def pendulum(r, P, N=3000, a=0, b=100):
    res = []

    for i in np.linspace(a, b, num=N):

        res.append(np.array(r))
        r, t = RK(r, i, P)
    if os.path.isfile("results.csv"):
        os.remove("results.csv")
    np.savetxt("results.csv", np.array(res), delimiter=",")
    return res



def main():
    name = "poincare_1"
    P = np.array([l_1, l_2, m_1, m_2])
    r = np.array([0.01, 0.01, 0, 0])
    epsilon = 0.001
    res = np.array(pendulum(r, P))

    ress = np.array(pendulum(r, P + epsilon))

    res = np.array(res)
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    fig.suptitle('Poincare maps ( \u03b5 = {0} )'.format(epsilon))
    ax1.plot(res[:,3], res[:, 1], '.')
    ax1.set(xlabel='\u03c9 2', ylabel="\u03c6 2")
    ax1.set_title("Initial conditions = {0}".format(P))
    ax2.set_title("")
    ax2.plot(ress[:,3], res[:, 1],'.')
    ax2.set_title("Initial conditions = {0} + \u03b5".format(P))
    ax2.set(xlabel='\u03c9 2')
    plt.savefig(str(name) + ".pdf", format="pdf")

    plt.show()

if __name__ == "__main__":
    main()
