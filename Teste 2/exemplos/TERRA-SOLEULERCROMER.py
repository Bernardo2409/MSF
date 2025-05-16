import numpy as np
import matplotlib.pyplot as plt

gm = 4*np.pi**2
x0 = 1
y0 = 0
vx0 = 0
vy0 = 2*np.pi
dt = 0.001

n = int(1/dt*10)

t = np.zeros(n+1)
x = np.zeros(n+1)
y = np.zeros(n+1)
vy = np.zeros(n+1)
vx = np.zeros(n+1)
ay = np.zeros(n+1)
ax = np.zeros(n+1)

x[0] = x0
y[0] = y0
vx[0] = vx0
vy[0] = vy0

for i in range(n):
    t[i+1]=t[i]+dt
    r=np.sqrt(x[i]**2+y[i]**2)
    ax[i]=-gm/r**3*x[i]
    ay[i]=-gm/r**3*y[i]
    vx[i+1]=vx[i]+ax[i]*dt
    vy[i+1]=vy[i]+ay[i]*dt
    x[i+1]=x[i]+vx[i+1]*dt
    y[i+1]=y[i]+vy[i+1]*dt

plt.figure(figsize=(8, 8))
plt.plot(x, y, label="Trajetória", color="blue")
plt.scatter(0, 0, color="orange", label="Centro (Sol)", s=100)
plt.xlabel("Posição x (UA)")
plt.ylabel("Posição y (UA)")
plt.title("Trajetória do objeto no plano x-y")
plt.legend()
plt.grid()
plt.axis("equal")
plt.show()