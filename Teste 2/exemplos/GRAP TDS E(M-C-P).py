import numpy as np
import matplotlib.pyplot as plt

m = 0.5
k = 2 / m**4
dt = 0.01
g = 9.80

t = np.arange(0, 100, dt)
x = np.zeros(t.size)
v = np.zeros(t.size)

Et = np.zeros(t.size)
Ec = np.zeros(t.size)
Ep = np.zeros(t.size)
Fx = np.zeros(t.size)

for i in range(0, t.size - 1):
    v[i+1] = v[i] + g * dt
    x[i+1] = x[i] + v[i] * dt

    Ep[i] = k * ((x[i] - 0.5)**2) * (x[i] + 0.5)**2 
    Fx[i] = -4 * k * (x[i])**3 + k * x[i]
    Ec[i] = 0.5 * m * v[i]**2

    Et[i] = Ec[i] + Ep[i]


plt.plot(t[:-1], Ep[:-1], label = "Energia Potencial")
plt.plot(t[:-1], Ec[:-1], label = "Energia Cinética")
plt.plot(t[:-1], Et[:-1], label = "Energia Total")

plt.title("Energias em função do tempo")
plt.xlabel("t (s)")
plt.ylabel("Energia (J)")
plt.legend()
plt.show()