import numpy as np
from matplotlib.pyplot import plot, show, figure


def linear_regression(m):
    m += 1
    MP_arr = np.ones((N, m))
    for i in range(1, m):
        MP_arr[:, i] = MP_arr[:, i-1]*x

    w = np.linalg.pinv(MP_arr) @ t

    # MP_arr_T = MP_arr.T
    # w = np.linalg.inv(MP_arr_T @ MP_arr) @ MP_arr_T @ t

    model = MP_arr @ w
    return model


def error_graph(m_range):
    M = np.arange(1, m_range + 1)
    E = np.zeros(M.size)

    for m in M:
        model = linear_regression(m)
        e = (t - model)**2
        E[m - 1] = np.sum(e) / 2

    figure()
    plot(M, E, color='green')
    show()


def graph(r):
    figure()
    plot(x, t, color='blue', linestyle=' ', marker='o', markersize='0.1')
    plot(x, z, color='red')
    plot(x, r, color='green')
    show()


N = 10000
x = np.linspace(0, 1, N)

z = 20*np.sin(2 * np.pi * 3 * x) + 100*np.exp(x)
error = 10 * np.random.randn(N)
t = z + error

graph(linear_regression(1))
graph(linear_regression(8))
graph(linear_regression(100))
error_graph(100)
