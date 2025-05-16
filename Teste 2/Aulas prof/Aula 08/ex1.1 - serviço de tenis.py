# Um jogador de ténis treina o serviço, sacando a bola da linha de base diagonalmente para a
# sua frente, como ilustrado no diagrama
# O jogador saca a bola do ponto (x, y, z) = (0, 2m, 3m) com velocidade (vx, vy, vz) = (160km/h,20km/h,-20km/h).
# 1. Determine o movimento da bola usando o metodo de Euler a 3D.
# Considere a força de gravidade e a resistência do ar, com velocidade terminal 𝑣𝑇 = 120km/h.

import numpy as np
import matplotlib.pyplot as plt

t0 = 0.0 # condição inicial, tempo [s]
tf = 0.5 # limite do domínio, tempo final [s]
dt = 0.001 # passo [s]

r0 = np.array([0.0, 2.0, 3.0]) # condição inicial, posição inicial [m]
v0 = np.array([160.0, 20.0, -20.0]) * 1000 / 3600 # condição inicial, velocidade ini
g = 9.8 # aceleração gravítica [m/s^2]
v_T = 120.0 # velocidade terminal [m/s]
M = 57 / 1000 # massa da bola [kg]
D = g / v_T ** 2 # coeficiente de resistência do ar [m^-1]

# inicializar domínio [s]
N = int((tf - t0) / dt + 1)
t = np.linspace(t0, tf, num=N)

# inicializar solução, aceleração [m/s^2]
a = np.zeros([np.size(t), 3]) # aceleração resultante
a_res = np.zeros([np.size(t), 3]) # aceleração devido à resistência do ar
a_grv = np.zeros([np.size(t), 3]) # aceleração devido à gravidade

# inicializar solução, velocidade [m/s]
v = np.zeros([np.size(t), 3])
v[0, :] = v0

# inicializar solução, posição [m]
r = np.zeros([np.size(t), 3])
r[0, :] = r0

for i in range(np.size(t) - 1):
    # calcular aceleração
    a_grv[i, :] = - g * np.array([0.0, 0.0, 1.0])
    a_res[i, :] = - D * np.linalg.norm(v[i, :]) * v[i, :]
    a[i, :] = a_grv[i, :] + a_res[i, :]
    v[i + 1, :] = v[i, :] + a[i, :] * dt
    r[i + 1, :] = r[i, :] + v[i, :] * dt

# 2. Faça um gráfico da trajetória da bola de ténis. Em que ponto a bola cai no solo?
fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection='3d')
# Coordenadas das linhas de campo e rede
x0 = 0
x1 = 5.5
x2 = 11.9
x3 = 18.3
x4 = 2 * 11.9
y0 = 0
y1 = 4.1
y2 = 8.2
z0 = 0.0
z1 = 1.0
ax.plot3D(np.array([x0, x4]), np.array([y0, y0]), np.array([z0, z0]), 'r-')
ax.plot3D(np.array([x1, x3]), np.array([y1, y1]), np.array([z0, z0]), 'r-')
ax.plot3D(np.array([x0, x4]), np.array([y2, y2]), np.array([z0, z0]), 'r-')
ax.plot3D(np.array([x0, x0]), np.array([y0, y2]), np.array([z0, z0]), 'r-')
ax.plot3D(np.array([x1, x1]), np.array([y0, y2]), np.array([z0, z0]), 'r-')
ax.plot3D(np.array([x2, x2]), np.array([y0, y2]), np.array([z0, z0]), 'r-')
ax.plot3D(np.array([x3, x3]), np.array([y0, y2]), np.array([z0, z0]), 'r-')
ax.plot3D(np.array([x4, x4]), np.array([y0, y2]), np.array([z0, z0]), 'r-')
ax.plot3D(np.array([x2, x2]), np.array([y0, y2]), np.array([z1, z1]), 'k-')
ax.plot3D(np.array([x2, x2]), np.array([y0, y0]), np.array([z0, z1]), 'k-')
ax.plot3D(np.array([x2, x2]), np.array([y2, y2]), np.array([z0, z1]), 'k-')
# Trajetória da bola
ax.plot3D(r[:, 0], r[:, 1], r[:, 2], 'b-')
ax.set_title('Trajetória da bola de ténis em 3D')
ax.set_box_aspect(aspect = (14, 8, 3))
plt.show()

plt.plot(r[:, 0], r[:, 1], 'b-')
plt.axis([-0.5, 24, -0.5, 8.5])
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.plot(np.array([x0, x4]), np.array([y0, y0]), 'r-')
plt.plot(np.array([x1, x3]), np.array([y1, y1]), 'r-')
plt.plot(np.array([x0, x4]), np.array([y2, y2]), 'r-')
plt.plot(np.array([x0, x0]), np.array([y0, y2]), 'r-')
plt.plot(np.array([x1, x1]), np.array([y0, y2]), 'r-')
plt.plot(np.array([x3, x3]), np.array([y0, y2]), 'r-')
plt.plot(np.array([x4, x4]), np.array([y0, y2]), 'r-')
plt.plot(np.array([x2, x2]), np.array([y0, y2]), 'k-')
xtk = [x0, x1, x2, x3, x4]
xlbl = ['0', '5.5', '11.9', '18.3', '23.8']
plt.xticks(xtk, xlbl)
plt.title('Projeção da trajetória em 0xy')
plt.show()

print("Coordenadas da bola quando passa a rede")
i = np.size(r[ r[:,0] > 11.9 , 0])
print("t = {0:.2f} s".format(t[i]))
print("x(tr) = {0:.2f} m".format(r[i, 0]))
print("y(tr) = {0:.2f} m".format(r[i, 1]))
print("z(tr) = {0:.2f} m".format(r[i, 2]))
print("Coordenadas da bola quando atinge o solo")
i = np.size(r[ r[:,2] > 0 , 2])
print("t = {0:.2f} s".format(t[i]))
print("x(tr) = {0:.2f} m".format(r[i, 0]))
print("y(tr) = {0:.2f} m".format(r[i, 1]))
print("z(tr) = {0:.2f} m".format(r[i, 2]))

plt.plot(r[:, 0], r[:, 2], 'b-')
plt.axis([-0.5, 24, -0.5, 12])
plt.xlabel("x [m]")
plt.ylabel("z [m]")
plt.plot(np.array([x0, x4]), np.array([z0, z0]), 'r-')
plt.plot(np.array([x2, x2]), np.array([z0, z1]), 'k-')
xtk = [x0, x1, x2, x3, x4]
xlbl = ['0', '5.5', '11.9', '18.3', '23.8']
plt.xticks(xtk, xlbl)
plt.title('Projeção da trajetória em 0xz')
plt.show()