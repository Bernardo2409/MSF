import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

g = 9.8
dt = 0.001
x_max = 2.5
fator_aceleracao = 2.0

x0 = 0.0
vx0 = 0.0


def f2(x):
    if 0 <= x < 2:
        return 0.025 * (x - 2) ** 2
    else:
        return 0

def f2_prime(x):
    if 0 <= x < 2:
        return 0.05 * (x - 2)
    else:
        return 0


x = x0
vx = vx0
t = 0.0

x_list = [x0]
vx_list = [vx0]
y_list = [f2(x)]
t_list = [t]

while x < x_max:
    ax = -g * f2_prime(x) * fator_aceleracao
    vx += ax * dt
    x += vx * dt
    t += dt

    x_list.append(x)
    vx_list.append(vx)
    y_list.append(f2(x))
    t_list.append(t)



print(f"Pista parabólica - tempo até x = 2.5 m: {t:.3f} s, velocidade final: {vx:.3f} m/s")

fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlim(0, 2.6)
ax.set_ylim(-0.05, 0.15)
ax.set_title("Exercício 2: Pista de forma parabólica")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.grid(True)

# Pista
x2_curve = np.linspace(0, 2.5, 500)
y2_curve = [f2(xc) for xc in x2_curve]
ax.plot(x2_curve, y2_curve, 'b', label="Pista parabólica")

# Bola
bola2, = ax.plot([], [], 'ro', markersize=10, label="Bola")

def init():
    bola2.set_data([], [])
    return bola2,

def update(frame):
    if frame < len(x_list):
        bola2.set_data([x_list[frame]], [y_list[frame]])
    return bola2,

ani = FuncAnimation(fig, update, frames=len(x_list), init_func=init, blit=True, interval=1)

plt.legend()
plt.tight_layout()
plt.show()