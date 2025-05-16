import numpy as np
import matplotlib.pyplot as plt

t0 = 0.0 # condição inicial, tempo [s]
tf = 4.0 # limite do domínio, tempo final [s]
dt = 0.001 # passo [s]

vx0 = 0.0 # condição inicial, módulo da velocidade inicial [m/s]
x0 = 0.0 # condição inicial, coordenada x da posição inicial [m]
y0 = 0.1 # condição inicial, coordenada y da posição inicial [m]
g = 9.80665 # aceleração gravitacional (valor standard) [m/s^2]

# Trajetória y(x)
# y(x) = 0.1 - 0.05*x para x < 2, de outra forma y(x) = 0

def y_func(x: float) -> float:
    return 0.1 - 0.05 * x if x < 2.0 else 0.0

# Derivada de y em ordem a x
# dy/dx = -0.05 para x < 2, de outra forma dy/dx = 0
def dydx_func(x: float) -> float:
    return -0.05 if x < 2.0 else 0.0

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

x1 = x
y1 = y
vx1 = vx
i25 = np.size(x[x<=2.5])
v25 = vx[i25]
t25 = t[i25]
print("Quando x = 2.5 m, a velocidade é v = {0:.5f} m/s²".format(v25))
print("Quando x = 2.5 m, o tempo decorrido é t = {0:.5f} s".format(t25))