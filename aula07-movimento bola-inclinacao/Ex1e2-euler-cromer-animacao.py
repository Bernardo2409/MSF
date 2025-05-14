import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

g = 9.8              
dt = 0.001           
x_max = 2.5          
acel = 2.0  

x0 = 0.0
vx0 = 0.0

def f1(x):
    if 0 <= x < 2:
        return 0.1 - 0.05 * x
    else:
        return 0.0

def f1_prime(x):
    if 0 <= x < 2:
        return -0.05
    else:
        return 0.0

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

def simular(f_prime, f, aceleracao_extra=1.0):
    x = x0
    vx = vx0
    t = 0.0

    x_list = [x]
    vx_list = [vx]
    y_list = [f(x)]
    t_list = [t]

    while x < x_max:
        ax = -g * f_prime(x) * aceleracao_extra
        vx += ax * dt
        x += vx * dt
        t += dt

        x_list.append(x)
        vx_list.append(vx)
        y_list.append(f(x))
        t_list.append(t)

    return np.array(t_list), np.array(x_list), np.array(vx_list), np.array(y_list)

t1, x1, vx1, y1 = simular(f1_prime, f1)
print("\nPista inclinada:")
print(f"Tempo até x = 2.5 m: {t1[-1]:.4f} s")
print(f"Velocidade final: {vx1[-1]:.4f} m/s")

y0 = f1(0)
Ep = g * y0
Ec = Ep
vf_energia = np.sqrt(2 * Ec)
print(f"Velocidade final teórica (conservação de energia): {vf_energia:.4f} m/s")

t2, x2, vx2, y2 = simular(f2_prime, f2, acel)
print("\nPista parabólica:")
print(f"Tempo até x = 2.5 m: {t2[-1]:.4f} s")
print(f"Velocidade final: {vx2[-1]:.4f} m/s")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

for ax in (ax1, ax2):
    ax.set_xlim(0, 2.6)
    ax.set_ylim(-0.05, 0.15)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True)

ax1.set_title("Exercício 1: Pista inclinada")
x_curve1 = np.linspace(0, 2.5, 500)
y_curve1 = [f1(xc) for xc in x_curve1]
ax1.plot(x_curve1, y_curve1, 'b-', label="Pista")
bola1, = ax1.plot([], [], 'ro', markersize=10, label="Bola")

ax2.set_title("Exercício 2: Pista parabólica")
x_curve2 = np.linspace(0, 2.5, 500)
y_curve2 = [f2(xc) for xc in x_curve2]
ax2.plot(x_curve2, y_curve2, 'b', label="Pista")
bola2, = ax2.plot([], [], 'ro', markersize=10, label="Bola")

def init():
    bola1.set_data([], [])
    bola2.set_data([], [])
    return bola1, bola2,

def update(frame):
    frame_idx = min(frame, len(x1)-1, len(x2)-1)
    
    if frame_idx < len(x1):
        bola1.set_data([x1[frame_idx]], [y1[frame_idx]])
    if frame_idx < len(x2):
        bola2.set_data([x2[frame_idx]], [y2[frame_idx]])
    
    return bola1, bola2,

ani = FuncAnimation(fig, update, frames=max(len(x1), len(x2)),
                   init_func=init, blit=True, interval=1)

plt.tight_layout()
plt.show()