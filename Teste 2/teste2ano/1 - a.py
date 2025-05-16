import numpy as np
import matplotlib.pyplot as plt

v0 = 100/3.6            #Passar a velocidade para m/s
angle = np.radians(16)
v0x = v0*np.cos(angle)
v0y = v0*np.sin(angle)
v0z = 0
g = -9.8
x0 = 0
y0 = 0
z0 = 0
dt = 0.01


t = np.arange(0,100, dt)
y = np.zeros(t.size)
x = np.zeros(t.size)
z = np.zeros(t.size)
vx = np.zeros(t.size)
vy = np.zeros(t.size)
vz = np.zeros(t.size)

x[0] = x0
y[0] = y0
z[0] = z0
vx[0] = v0x
vy[0] = v0y
vz[0] = v0z
D = 0.0127

for i in range(0, t.size-1):
    v = np.sqrt(vx[i]**2 + vy[i]**2 + vz[i] ** 2)

    ax = - D * vx[i] * abs(v)
    ay = g - D * vy[i] * abs(v)
    az = - D * vz[i] * abs(v)

    vx[i+1] = vx[i] + ax * dt
    vy[i+1] = vy[i] + ay * dt
    vz[i+1] = vz[i] + az * dt

    x[i+1] = x[i] + vx[i] * dt 
    y[i+1] = y[i] + vy[i] * dt
    z[i+1] = z[i] + vz[i] * dt

    if y[i+1] < 0:
        break

t = t[:i+2]
x = x[:i+2]
y = y[:i+2]
z = z[:i+2]
vx = vx[:i+2]
vy = vy[:i+2]
vz = vz[:i+2]

plt.plot(x, y)
plt.xlabel("x(t) (m)")
plt.ylabel("y(t) (m)")
plt.title("Trajetória em xOy")
plt.show()

for i in range(len(x)):
    if x[i+1] > 20:
        index = i
        break

if 0 < y[index] < 2.4 and -3.75 < z[index] < 3.75:
    print("A bola entrou na baliza, marcaste!")
else:
    print("A bola não entrou na baliza")

