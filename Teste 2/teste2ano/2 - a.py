import numpy as np
import matplotlib.pyplot as plt

m = 0.5
k = 2 / m**4
dt = 0.01
g = 9.80

t = np.arange(0, 100, dt)
x = np.zeros(t.size)
v = np.zeros(t.size)

Ep = np.zeros(t.size)
Fx = np.zeros(t.size)

for i in range(0, t.size - 1):
    v[i+1] = v[i] + g * dt
    x[i+1] = x[i] + v[i] * dt

    Ep[i] = k * ((x[i] - 0.5)**2) * (x[i] + 0.5)**2 
    Fx[i] = -4 * k * (x[i])**3 + k * x[i]


plt.plot(t[:-1], Ep[:-1])
plt.title("Diagrama da Energia Potencial")
plt.xlabel("t (s)")
plt.ylabel("Energia (J)")
plt.show()

print("Pelo gráfico, quando a energia for menor do que 0.25 J, o o movimento vai-se tornar num movimento acelerado, uma vez que a energia aumenta exponencialmente")





