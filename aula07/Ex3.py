import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

g = 9.8
dt = 0.001
x_max = 2.5
acel = 2.0

x0 = 0.0
vx0 = 0.0


def f3(x):
    if 0 <= x < 2:
        return 0.3 * (x - 2) ** 2
    else:
        return 0

def f3_prime(x):
    if 0 <= x < 2:
        return 0.6 * (x - 2)
    else:
        return 0


x = x0
vx = vx0
t = 0.0

x_list = [x0]
vx_list = [vx0]
y_list = [f3(x)]
t_list = [t]

while x < x_max:
    ax = -g * f3_prime(x) * acel
    vx += ax * dt
    x += vx * dt
    t += dt

    x_list.append(x)
    vx_list.append(vx)
    y_list.append(f3(x))
    t_list.append(t)



print(f"Pista parabólica - tempo até x = 2.5 m: {t:.3f} s, velocidade final: {vx:.3f} m/s")

fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlim(0, 2.6)

#mudei a altura 0.10 altura dos outros
ax.set_ylim(-0.05, 1)
ax.set_title("Exercício 2: Pista de forma parabólica")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.grid(True)

# Pista
x3_curve = np.linspace(0, 2.5, 500)
y3_curve = [f3(xc) for xc in x3_curve]
ax.plot(x3_curve, y3_curve, 'b', label="Pista parabólica")

# Bola
bola3, = ax.plot([], [], 'ro', markersize=10, label="Bola")

def init():
    bola3.set_data([], [])
    return bola3,

def update(frame):
    if frame < len(x_list):
        bola3.set_data([x_list[frame]], [y_list[frame]])
    return bola3,

ani = FuncAnimation(fig, update, frames=len(x_list), init_func=init, blit=True, interval=1)

plt.legend()
plt.tight_layout()
plt.show()