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

gm = G

for i in range(np.size(t) - 1):
    t[i+1] = t[i]+dt
    r=np.sqrt(x[i]**2+y[i]**2)
    ax[i] = -gm/r**3*x[i]
    ay[i] = -gm/r**3*y[i]
    vx[i+1] = vx[i] + ax[i]*dt
    vy[i+1] = vy[i] + ay[i]*dt
    x[i+1] = x[i] + vx[i]*dt
    y[i+1] = y[i] + vy[i]*dt

plt.plot(x, y, label="Órbita da Terra")
plt.xlabel("Posição em X (AU)")
plt.ylabel("Posição em Y (AU)")
plt.title("Órbita da Terra em torno do Sol")
plt.show()