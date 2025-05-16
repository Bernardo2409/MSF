import numpy as np
import matplotlib.pyplot as plt

t0 = 0.0 # condição inicial, tempo [ano]
tf = 10.0 # limite do domínio, tempo final [ano]
dt = 0.001 # passo [ano]

v0 = 2.0 * np.pi # condição inicial, módulo da velocidade inicial [AU/ano]
x0 = 1.0 # condição inicial, coordenada x da posição inicial [AU]
G = 4.0 * np.pi ** 2 # constante de gravitação [AU^3 / M ano^2]

# inicializar domínio [ano]
t = np.arange(t0, tf, dt)

# inicializar solução, aceleração a 2D [AU / ano^2]
a = np.zeros([np.size(t), 2])

# inicializar solução, velocidade [m / s]
v = np.zeros([np.size(t), 2])
v[0, :] = np.array([0, v0]) # velocidade [AU / ano] para t = 0 ano

# inicializar solução, posição [m]
r = np.zeros([np.size(t), 2])
r[0, :] = np.array([x0, 0.0]) # posição [AU] para t = 0 ano

for i in range(np.size(t) - 1):
    a[i, :] = - G * r[i, :] / np.linalg.norm(r[i, :]) ** 3 # aceleração instantânea
    v[i + 1, :] = v[i, :] + a[i, :] * dt
    r[i + 1, :] = r[i, :] + v[i, :] * dt


plt.plot(r[:, 0], r[:, 1], 'b-')
plt.xlabel("r_x [AU]")
plt.ylabel("r_y [AU]")
plt.show()