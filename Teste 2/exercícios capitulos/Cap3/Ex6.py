import numpy as np
import matplotlib.pyplot as plt

# --- Constantes ---
k = 1.0          # Constante da mola (N/m)
m = 1.0          # Massa (kg)
omega = np.sqrt(k / m)

# --- Condições iniciais ---
x0 = 4.0         # Posição inicial (m)
v0 = 0.0         # Velocidade inicial (m/s)
A = x0           # Amplitude
phi = 0          # Fase (porque v0 = 0, posição máxima)

# --- Tempo de simulação ---
t_max = 20
dt = 0.01
t = np.arange(0, t_max, dt)
N = len(t)

# --- Solução analítica ---
x_analitico = A * np.cos(omega * t + phi)
v_analitico = -A * omega * np.sin(omega * t + phi)

# --- Método numérico (Euler-Cromer) ---
x_num = np.zeros(N)
v_num = np.zeros(N)
x_num[0] = x0
v_num[0] = v0

for i in range(N - 1):
    a = -k * x_num[i] / m
    v_num[i + 1] = v_num[i] + a * dt
    x_num[i + 1] = x_num[i] + v_num[i + 1] * dt

# --- Gráfico: Posição ---
plt.figure(figsize=(10, 5))
plt.plot(t, x_analitico, label='x(t) analítico', linestyle='--')
plt.plot(t, x_num, label='x(t) numérico (Euler-Cromer)', alpha=0.7)
plt.title("Posição x(t) do corpo ligado à mola")
plt.xlabel("Tempo (s)")
plt.ylabel("Posição (m)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Gráfico: Velocidade ---
plt.figure(figsize=(10, 5))
plt.plot(t, v_analitico, label='v(t) analítico', linestyle='--')
plt.plot(t, v_num, label='v(t) numérico (Euler-Cromer)', alpha=0.7)
plt.title("Velocidade v(t) do corpo ligado à mola")
plt.xlabel("Tempo (s)")
plt.ylabel("Velocidade (m/s)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
