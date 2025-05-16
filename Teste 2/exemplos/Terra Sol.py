import matplotlib.pyplot as plt
import numpy as np

G = 4*np.pi**2

m = 5.978 * 10**24
M = 1 ## 1.989 * 10**30

d = 1

dt = 0.01
t = np.arange(0,100,dt)

Rx = np.zeros(t.size)
Ry = np.zeros(t.size)

Vx = np.zeros(t.size)
Vy = np.zeros(t.size)

Rx[0] = 1
Vy[0] = 2 * np.pi

##Metodo euler-cromer
for i in range(0, t.size-1):
    d = np.sqrt(Rx[i]**2 + Ry[i]**2)
    ax = (-G * M * Rx[i])/d**3
    ay = (-G * M * Ry[i])/d**3
    
    Vx[i+1] = Vx[i] + ax * dt
    Vy[i+1] = Vy[i] + ay * dt
    
    Rx[i+1] = Rx[i] + Vx[i+1] * dt
    Ry[i+1] = Ry[i] + Vy[i+1] * dt
    
plt.plot(Rx,Ry, label="Euler-Cromer")

##Metodo euler
for i in range(0, t.size-1):
    d = np.sqrt(Rx[i]**2 + Ry[i]**2)
    ax = (-G * M * Rx[i])/d**3
    ay = (-G * M * Ry[i])/d**3
    
    Vx[i+1] = Vx[i] + ax * dt
    Vy[i+1] = Vy[i] + ay * dt
    
    Rx[i+1] = Rx[i] + Vx[i] * dt
    Ry[i+1] = Ry[i] + Vy[i] * dt
    
plt.plot(Rx,Ry, label="Euler")

plt.legend()
plt.show()