import numpy as np
import matplotlib.pyplot as plt

sigma = 10
b = 8/3
r = 28

x0 = 0
y0 = 1
z0 = 0

dt = 0.01
t_max = 50
n_steps = int(t_max / dt)

t = np.linspace(0, t_max, n_steps)
x = np.zeros(n_steps)
y = np.zeros(n_steps)
z = np.zeros(n_steps)

x[0] = x0
y[0] = y0
z[0] = z0

for i in range(n_steps - 1):
    dx = sigma * (y[i] - x[i]) * dt
    dy = (x[i] * (r - z[i]) - y[i]) * dt
    dz = (x[i] * y[i] - b * z[i]) * dt
    
    x[i + 1] = x[i] + dx
    y[i + 1] = y[i] + dy
    z[i + 1] = z[i] + dz

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(t, x, label='x(t)', color='blue')
plt.plot(t, y, label='y(t)', color='orange')
plt.plot(t, z, label='z(t)', color='green')
plt.title('Sistemas de Lorenz')
plt.xlabel('Tempo')
plt.ylabel('Valores')
plt.legend()
plt.grid()

plt.subplot(2, 2, 2)
plt.plot(x, y, label='x vs y', color='purple')
plt.title('Diagrama de Fase: x vs y')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()

plt.subplot(2, 2, 3)
plt.plot(x, z, label='x vs z', color='red')
plt.title('Diagrama de Fase: x vs z')
plt.xlabel('x')
plt.ylabel('z')
plt.title("Projeção de x e z")
plt.grid()

plt.subplot(2, 2, 4)
plt.plot(y, z, label='y vs z', color='brown')
plt.xlabel('y')
plt.ylabel('z')
plt.title("Projeção de y e z")
plt.grid()

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
plt.plot(x, z, color='blue', linewidth=0.5)
plt.xlabel('x')
plt.ylabel('z')
plt.title('Atrator de Lorenz (projecao x-z)')
plt.grid()
plt.show()