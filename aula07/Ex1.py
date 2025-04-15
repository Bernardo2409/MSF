import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

g = 9.8              
dt = 0.001           
x_max = 2.5           

x0 = 0.0
vx0 = 0.0

def f(x):
    if 0 <= x < 2:
        return 0.1 - 0.05 * x
    else:
        return 0.0

def f_prime(x):
    if 0 <= x < 2:
        return -0.05
    else:
        return 0.0

x = x0
vx = vx0
t = 0.0

x_list = [x0]
vx_list = [vx0]
y_list = [f(x)]
t_list = [t]

while x < x_max:
    ax = -g * f_prime(x)
    vx += ax * dt
    x += vx * dt
    t += dt

    x_list.append(x)
    vx_list.append(vx)
    y_list.append(f(x))
    t_list.append(t)

print(f"Velocidade final: {vx:.4f} m/s")
print(f"Tempo até x = 2.5 m: {t:.4f} s")

y0 = f(0)
Ep = g * y0
Ec = Ep
vf_energia = np.sqrt(2 * Ec)
print(f"Velocidade final com as energias: {vf_energia:.4f} m/s")

fig, ax = plt.subplots()
ax.set_xlim(0, 2.6)
ax.set_ylim(-0.05, 0.15)
ax.set_title("Exercício 1: bola numa pista inclinada")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.grid(True)

# Pista
x_curve = np.linspace(0, 2.5, 500)
y_curve = [f(xc) for xc in x_curve]
ax.plot(x_curve, y_curve, 'b-', label="Pista")

# Bola
bola, = ax.plot([], [], 'ro', markersize=10, label="Bola")

def init():
    bola.set_data([], [])
    return bola,

def update(frame):
    if frame < len(x_list):
        bola.set_data([x_list[frame]], [y_list[frame]])
    return bola,

ani = FuncAnimation(fig, update, frames=len(x_list), init_func=init, blit=True, interval=1)

plt.legend()
plt.tight_layout()
plt.show()