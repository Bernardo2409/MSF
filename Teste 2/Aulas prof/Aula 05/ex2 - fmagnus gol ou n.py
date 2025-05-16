import numpy as np
import matplotlib.pyplot as plt

t0 = 0.0 # condição inicial, tempo [s]
tf = 0.5 # limite do domínio, tempo final [s]
dt = 0.001 # passo [s]

r0 = np.array([0.0, 0.0, 23.8]) # condição inicial, velocidade inicial [m/s]
v0 = np.array([25.0, 5.0, -50.0]) # condição inicial, velocidade inicial [m/s]

w = 390.0 # condição inicial, velocidade angular CONSTANTE [ra
g = 9.8 # aceleração gravítica [m/s^2]
R = 0.11 # raio da bola [m]
A = np.pi * R ** 2 # área da secção da bola
m = 0.45 # massa da bola [kg]
rho = 1.225 # densidade do ar [kg/m^3]
v_T = 100 * 1000 / 3600 # velocidade terminal [m/s]
D = g / v_T ** 2

# inicializar domínio [s]
t = np.arange(t0, tf, dt)

# inicializar solução, aceleração [m/s^2]
a = np.zeros([3, np.size(t)])

# inicializar solução, velocidade [m/s]
v = np.zeros([3, np.size(t)])
v[:,0] = v0

# inicializar solução, posição [m]
r = np.zeros([3, np.size(t)])
r[:,0] = r0

for i in range(np.size(t) - 1):
    # Caclular norma da velocidade, |v|
    v_norm = np.linalg.norm(v[:, i])
    a[0, i] = - D * v[0, i] * v_norm + A * rho * R * w * v[2,i] / (2 * m)
    a[1, i] = - g - D * v[1, i] * v_norm
    a[2, i] = - D * v[2, i] * v_norm - A * rho * R * w * v[0,i] / (2 * m)
    v[:, i + 1] = v[:, i] + a[:, i] * dt
    r[:, i + 1] = r[:, i] + v[:, i] * dt

plt.plot(r[0,:], r[1,:], 'b-')
plt.xlabel("Posição horizontal, r_x [m]")
plt.ylabel("Posição vertical, r_y [m]")
plt.show()