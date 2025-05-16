k = 1
m = 1
x0 = 4
v0x = 0
#energia potencial elatica inicial
Ep = 0.5 * k * x0 ** 2
#energia cinetica inicial
Ec = 0.5 * m * v0x ** 2
#energia mecanica inicial
Em = Ep + Ec
print(f"Energia total: {Em}J")

import numpy as np
import matplotlib.pyplot as plt
k = 1
m = 1
x0 = 4
v0 = 0
dt = 0.01
tf = 20

n = int(tf / dt + 0.1)

x_euler = np.zeros(n)
v_euler = np.zeros(n)
E_euler = np.zeros(n)

x_ec = np.zeros(n)
v_ec = np.zeros(n)
E_ec = np.zeros(n)

x_euler[0] = x0
v_euler[0] = v0
E_euler[0] = Em

x_ec[0] = x0
v_ec[0] = v0
E_ec[0] = Em
t = np.linspace(0, tf, n)
#metodo de euler
for i in range(n - 1):
    a = - k * x_euler[i] / m
    v_euler[i + 1] = v_euler[i] + a * dt
    x_euler[i + 1] = x_euler[i] + v_euler[i] * dt
    E_euler[i + 1] = (0.5 * m * v_euler[i + 1] ** 2) + (0.5 * k * x_euler[i + 1] ** 2)
for i in range(n - 1):
    a = - k * x_ec[i] / m
    v_ec[i + 1] = v_ec[i] + a * dt
    x_ec[i + 1] = x_ec[i] + v_ec[i + 1] * dt
    E_ec[i + 1] = (0.5 * m * v_ec[i + 1] ** 2) + (0.5 * k * x_ec[i + 1] ** 2)
    
plt.plot(t, E_euler, "--r", label="Metodo de Euler")
plt.plot(t, E_ec, "--b", label="Metodo de Euler-Cromer")
plt.plot(Em, "--g", label="Valor real")
plt.xlabel("Tempo [s]")
plt.ylabel("Energia Mecanica [J]")
plt.title("Metodo de Euler vs Metodo Euler-Cromer")
plt.legend()
plt.axhline(Em, color='green', linestyle=':', label='Energia Teórica')
plt.ylim(7.8,)
plt.show()
