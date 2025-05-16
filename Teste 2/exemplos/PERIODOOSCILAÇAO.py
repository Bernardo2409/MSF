import numpy as np
import matplotlib.pyplot as plt

def maxminv(x0, x1, x2, y0, y1, y2):
    xab = x0 - x1
    xac = x0 - x2
    xbc = x1 - x2
    a = y0 / (xab * xac)
    b = -y1 / (xab * xbc)
    c = y2 / (xac * xbc)
    xmla = (b + c) * x0 + (a + c) * x1 + (a + b) * x2
    xm = 0.5 * xmla / (a + b + c)
    xta = xm - x0
    xtb = xm - x1
    xtc = xm - x2
    ymax = a * xtb * xtc + b * xta * xtc + c * xta * xtb
    return xm, ymax

L = 1.0
g = 9.8
theta0 = 0.1
omega0 = 0.0

dt = 0.001
t_max = 10
N = int(t_max / dt)

t = np.zeros(N + 1)
theta = np.zeros(N + 1)
omega = np.zeros(N + 1)

theta[0] = theta0
omega[0] = omega0

for i in range(N):
    omega[i + 1] = omega[i] - (g / L) * np.sin(theta[i]) * dt
    theta[i + 1] = theta[i] + omega[i + 1] * dt
    t[i + 1] = t[i] + dt


maximos = []
for i in range(1, N):
    if theta[i - 1] < theta[i] > theta[i + 1]:
        xm, ymax = maxminv(t[i - 1], t[i], t[i + 1], theta[i - 1], theta[i], theta[i + 1])
        maximos.append((xm, ymax))
        if len(maximos) == 2:
            break  
if len(maximos) == 2:
    t1, y1 = maximos[0]
    t2, y2 = maximos[1]
    T_exp = t2 - t1
    T_teorico = 2 * np.pi * np.sqrt(L / g)
    print(f"Período experimental ≈ {T_exp:.6f} s")
    print(f"Período teórico      ≈ {T_teorico:.6f} s")
    print(f"Erro relativo        ≈ {abs(T_exp - T_teorico) / T_teorico * 100:.4f}%")
else:
    print("Não foram encontrados dois máximos.")