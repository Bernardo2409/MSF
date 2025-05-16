import numpy as np
import matplotlib.pyplot as plt

g = 9.8
m = 0.45
r = 0.11
PAr = 1.225
dt = 0.01

t = np.arange(0, 5 + dt, dt)
v0 = 100/3.6
ang = np.radians(16)
A = np.pi * (r) ** 2
D = 0.0127

mag = 0.5 * A * PAr *r

rx = np.zeros(t.size)
ry = np.zeros(t.size)
rz = np.zeros(t.size)

vx = np.zeros(t.size)
vy = np.zeros(t.size)
vz = np.zeros(t.size)

rx[0] = 0
ry[0] = 0
rz[0] = 0

vx[0] = v0 * np.cos(ang)
vy[0] = v0 * np.sin(ang)
vz[0] = 0

wx = 0
wy = 0
wz = -10

for i in range(0, t.size-1):
    v = np.sqrt(vx[i]**2 + vy[i]**2 + vz[i]**2)
    
    amagx = mag * np.cross([wx, wy, wz], [vx[i], vy[i], vz[i]])[0] / m 
    amagy = mag * np.cross([wx, wy, wz], [vx[i], vy[i], vz[i]])[1] / m 
    amagz = mag * np.cross([wx,wy,wz], [vx[i],vy[i],vz[i]])[2] / m 
    
    ax = - D * vx[i] * abs(v) + amagx
    ay = - g - D * vy[i] * abs(v) + amagy
    az = - D * vz[i] * abs(v) + amagz
    
    vx[i+1] = vx[i] + ax * dt
    vy[i+1] = vy[i] + ay * dt
    vz[i+1] = vz[i] + az * dt
    
    rx[i+1] = rx[i] + vx[i] *dt
    ry[i+1] = ry[i] + vy[i] * dt
    rz[i+1] = rz[i] + vz[i] * dt

    t[i+1] = t[i] + dt

t = t[:i]
vx = vx[:i]
vy = vy[:i]
vz = vz[:i]

rx = rx[:i]
ry = ry[:i]
rz = rz[:i]

plt.plot(rx, ry)
plt.xlabel("x(t) (m)")
plt.ylabel("y(t) (m)")
plt.title("Trajetória em xOy")
plt.show()

for i in range(len(rx)):
    if rx[i+1] > 20:
        index = i
        break

if 0 < ry[index] < 2.4 and -3.75 < rz[index] < 3.75:
    print("A bola entrou na baliza, marcaste!")
else:
    print("A bola não entrou na baliza")


