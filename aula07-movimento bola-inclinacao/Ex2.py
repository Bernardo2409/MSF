import numpy as np
import matplotlib.pyplot as plt

# Condições iniciais
x0 = 0.0  # posição inicial em x [m]
y0 = 0.1  # posição inicial em y [m]
vx0 = 0.0  # velocidade inicial em x [m/s]
g = 9.80665  # aceleração gravitacional [m/s^2]
t0 = 0.0  # tempo inicial [s]
tf = 2.0  # tempo final [s]
dt = 0.01  # passo de tempo [s]

# Trajetória y(x)
def y_func(x: float) -> float:
    return 0.025 * (x - 2)**2 if x < 2.0 else 0.0

# Derivada de y em ordem a x
def dydx_func(x: float) -> float:
    return 0.05 * (x - 2) if x < 2.0 else 0.0

# Inicializar domínio
t = np.arange(t0, tf, dt)

# Inicializar soluções
ax = np.zeros(np.size(t))      # aceleração em x
vx = np.zeros(np.size(t))      # velocidade em x
x = np.zeros(np.size(t))       # posição em x
y = np.zeros(np.size(t))       # posição em y

# Condições iniciais
vx[0] = vx0
x[0] = x0
y[0] = y0

# Método de Euler-Cromer
for i in range(np.size(t) - 1):
    ax[i] = -g * dydx_func(x[i])
    vx[i + 1] = vx[i] + ax[i] * dt
    x[i + 1] = x[i] + vx[i + 1] * dt
    y[i + 1] = y_func(x[i + 1])

# Encontrar índice onde x = 2.5 m
i25 = np.argmax(x >= 2.5)
v25 = vx[i25]
t25 = t[i25]
print(f"Quando x = 2.5 m, a velocidade é v = {v25:.5f} m/s")
print(f"Quando x = 2.5 m, o tempo decorrido é t = {t25:.5f} s")

# Gráficos
fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('tempo (s)')
ax1.set_ylabel('x (m)', color=color)
ax1.plot(t, x, color=color, label='x(t)')
ax1.tick_params(axis='y', labelcolor=color)
ax1.legend(loc='upper left')

ax2 = ax1.twinx()  # partilhar eixo horizontal

color = 'tab:red'
ax2.set_ylabel('y (m)', color=color)
ax2.plot(t, y, color=color, label='y(t)')
ax2.tick_params(axis='y', labelcolor=color)
ax2.legend(loc='upper right')

fig.tight_layout()
plt.title("Posição x(t) e y(t)")
plt.show()

# Gráfico de velocidade em função de y
plt.plot(y, vx, 'b-', label='vx(y)')
plt.xlabel("y [m]")
plt.ylabel("v [m/s]")
plt.title("Velocidade vx em função de y")
plt.legend()
plt.grid()
plt.show()