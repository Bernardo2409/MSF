import numpy as np
import matplotlib.pyplot as plt

t0 = 0.0
tf = 4.0
dt = 0.01
v0 = 2 * np.pi
x0 = 1.0

G = 4 * np.pi**2

t = np.arange(t0, tf, dt)
x = np.zeros(np.size(t))
y = np.zeros(np.size(t))
vx = np.zeros(np.size(t))
vy = np.zeros(np.size(t))
ax = np.zeros(np.size(t))
ay = np.zeros(np.size(t))

x[0] = x0
y[0] = 0
vx[0] = 0
vy[0] = v0
N=len(t)
 
for i in range(N-1):
    r = np.sqrt(x[i]**2 + y[i]**2)
    ax = -G * x[i] / r**3
    ay = -G * y[i] / r**3
    
    vx[i+1] = vx[i] + ax * dt
    vy[i+1] = vy[i] + ay * dt
    x[i+1] = x[i] + vx[i+1] * dt
    y[i+1] = y[i] + vy[i+1] * dt


plt.figure(figsize=(6,6))
plt.plot(x, y, label="Órbita da Terra", color='b')
plt.scatter(0, 0, color='orange', label="Sol")  
plt.xlabel("Posição em X (AU)")
plt.ylabel("Posição em Y (AU)")

plt.show()

print("Sim, é possivel obter elipses, e que concordam com as leis de Kepler.")