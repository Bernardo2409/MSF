import numpy as np
import matplotlib.pyplot as plt

# Parâmetros
k = 0.2
alpha = 1.0
b = 0.01
F0 = 5.0
omega_f = 0.6
tf = 50.0

def acelera(t, x, vx):
    return -k*x - 4*alpha*x**3 - b*vx + F0*np.cos(omega_f*t)

def rk4_x_vx(t, x, vx, acelera, dt):
    ax1 = acelera(t, x, vx)
    c1v = ax1 * dt
    c1x = vx * dt

    ax2 = acelera(t + dt/2., x + c1x/2., vx + c1v/2.)
    c2v = ax2 * dt
    c2x = (vx + c1v/2.) * dt

    ax3 = acelera(t + dt/2., x + c2x/2., vx + c2v/2.)
    c3v = ax3 * dt
    c3x = (vx + c2v/2.) * dt

    ax4 = acelera(t + dt, x + c3x, vx + c3v)
    c4v = ax4 * dt
    c4x = (vx + c3v) * dt

    xp = x + (c1x + 2*c2x + 2*c3x + c4x) / 6.
    vxp = vx + (c1v + 2*c2v + 2*c3v + c4v) / 6.
    return xp, vxp

dt_list = [0.1, 0.075, 0.050, 0.020, 0.005]

resultados = []

for dt in dt_list:
    n_steps = int(tf / dt)
    x = 1.0
    vx = 0.0
    t = 0.0
    
    for i in range(n_steps):
        x, vx = rk4_x_vx(t, x, vx, acelera, dt)
      
    
    resultados.append((dt, x, vx))

print("Passo dt | Posição final x(50s) | Velocidade final vx(50s)")
for dt, x_final, vx_final in resultados:
    print(f"{dt:.4f}    | {x_final:.6f}          | {vx_final:.6f}")
