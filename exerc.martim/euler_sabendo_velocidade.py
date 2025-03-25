import numpy as np
import matplotlib.pyplot as plt

g = 9.8  # m/s^2

def MetodoEuler(x0, v0, t0, tf, dt):
    N = int((tf - t0) / dt + 0.1) + 1
    x = np.zeros(N)
    v = np.zeros(N)
    t = np.zeros(N)

    x[0] = x0
    v[0] = v0
    t[0] = t0

    for i in range(N - 1):
        a = g
        x[i + 1] = x[i] + dt * v[i]
        v[i + 1] = v[i] + dt * a
        t[i + 1] = t[i] + dt

    return t, x, v

x0 = 1.0
v0 = 0.0
t0 = 0.0
tf = 2.0
dt = 0.1

t, x, v = MetodoEuler(x0, v0, t0, tf, dt)
x_th = x0 + v0 * t + 0.5 * g * t**2

plt.subplot(2, 1, 1)
plt.plot(t, x, '.')
plt.plot(t, x_th)

plt.subplot(2, 1, 2)
plt.plot(t, v, '.')

plt.show()