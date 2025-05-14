import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

t0 = 0
tf = 10
dt = 0.001  
v0 = 2 * np.pi
x0 = 1.0

G = 4.0 * np.pi ** 2

t = np.arange(t0, tf, dt)           
a = np.zeros([np.size(t), 2])

v = np.zeros([np.size(t), 2])
v[0, :] = np.array([0, v0])

r = np.zeros([np.size(t), 2])       
r[0, :] = np.array([x0, 0.0])

for i in range(np.size(t) - 1):
    a[i, :] = -G * r[i, :] / np.linalg.norm(r[i, :]) ** 3
    v[i + 1, :] = v[i, :] + a[i, :] * dt
    r[i + 1, :] = r[i, :] + v[i, :] * dt

# Ex3
fig, ax = plt.subplots()

ax.scatter(0, 0, color='yellow', s=300, label='Sol')

terra, = ax.plot(r[0,0], r[0,1], 'o', label='Terra')
ax.set(xlim=[-2, 2], ylim=[-2, 2])

def update(frame):
    # terra.set_data([r[frame, 0], r[frame, 1]])
    terra.set_xdata([r[frame, 0]])
    terra.set_ydata([r[frame, 1]])
    return terra,

# ani = FuncAnimation(fig, update, frames=np.arange(0, len(t), 10), interval=20)
ani = FuncAnimation(fig, update, frames= 600, interval=5)


ax.plot(r[:, 0], r[:, 1], 'b--', )    
plt.xlabel("Posição em X (AU)")
plt.ylabel("Posição em Y (AU)")
plt.legend()
plt.pause(0.01)
plt.show()

print("A volta da Terra ao Sol não é fechada")
print("Sim")