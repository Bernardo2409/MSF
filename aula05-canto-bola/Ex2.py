import numpy as np
import matplotlib.pyplot as plt

m = 0.45
r = 0.11
A = np.pi * r**2
p_ar = 1.225
g = 9.8
w = np.array([0, 390, 0])

dt = 0.001
t0 = 0
tf = 0.5
x0 = np.array([0, 0, 23.8])
v0 = np.array([25, 5, -50])

def accel(v):
    F_magnus = 1/2 * A * p_ar * r * np.cross(w, v)
    return np.array([0, -g, 0]) + F_magnus/m

n = int((tf-t0) / dt)  #nº passos
matriz = (n + 1, 3)

t = np.zeros(n + 1)
x = np.zeros(matriz)
v = np.zeros(matriz)
a = np.zeros(matriz)

a[0] = accel(v0)
v[0] = v0
t[0] = t0
x[0] = x0

for i in range(n):
    a[i + 1] = accel(v[i])
    v[i + 1] = v[i] + a[i] * dt
    x[i + 1] = x[i] + v[i] * dt
    t[i + 1] = t[i] + dt

#2D
plt.figure()
plt.plot(t, x[:, 0], label="X")
plt.plot(t, x[:, 1], label="Y")
plt.plot(t, x[:, 2], label="Z")
plt.xlabel("t (s)")
plt.ylabel("x(t) (m)")
plt.legend(loc="upper left")
plt.title("Trajetória em 2D")
plt.show()

#3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x[:, 0], x[:, 1], x[:, 2], label="Trajetória 3D")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title("Trajetória em 3D")
ax.legend()
ax.view_init(0, -30, 90)
plt.show()
