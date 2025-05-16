import numpy as np
import matplotlib.pyplot as plt

# Faça uma simulação do movimento da bola usando o metodo de Euler-Cromer com as
# seguintes condições inicias: x(t = 0) = 0 m; vx(t = 0) = 0 m/s

t0 = 0.0 # condição inicial, tempo [s]
tf = 4.0 # limite do domínio, tempo final [s]
dt = 0.001 # passo [s]

vx0 = 0.0 # condição inicial, módulo da velocidade inicial [m/s]
x0 = 0.0 # condição inicial, coordenada x da posição inicial [m]
y0 = 0.1 # condição inicial, coordenada y da posição inicial [m]
g = 9.80665 # aceleração gravitacional (valor standard) [m/s^2]

# Trajetória y(x)
# y(x) = 0.025 * (x - 2)**2 para x < 2, de outra forma y(x) = 0
def y_func(x: float) -> float:
    return 0.025 * (x - 2)**2 if x < 2.0 else 0.0

# Derivada de y em ordem a x
# dy/dx = 0.05 * (x - 2) para x < 2, de outra forma dy/dx = 0
def dydx_func(x: float) -> float:
    return 0.05 * (x - 2) if x < 2.0 else 0.0

# inicializar domínio [ano]
t = np.arange(t0, tf, dt)
# inicializar solução, aceleração a 1D [m/s^2]
ax = np.zeros(np.size(t))
# inicializar solução, velocidade [m/s]
vx = np.zeros(np.size(t))
vx[0] = vx0
# inicializar solução, posição [m]
x = np.zeros(np.size(t))
y = np.zeros(np.size(t))
x[0] = x0
y[0] = y0

for i in range(np.size(t) - 1):
    # aceleração
    ax[i] = -g * dydx_func(x[i])
    # Metodo de Euler-Cromer
    vx[i + 1] = vx[i] + ax[i] * dt
    x[i + 1] = x[i] + vx[i + 1] * dt
    y[i + 1] = y_func(x[i+1])

fig, ax1 = plt.subplots()
color = 'tab:blue'
ax1.set_xlabel('time (s)')
ax1.set_ylabel('x', color=color)
ax1.plot(t[x<2.5], x[x<2.5], color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax2 = ax1.twinx() # partilhar eixo horizontal
color = 'tab:red'
ax2.set_ylabel('y', color=color)
ax2.plot(t[x<2.5], y[x<2.5], color=color)
ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()
plt.show()