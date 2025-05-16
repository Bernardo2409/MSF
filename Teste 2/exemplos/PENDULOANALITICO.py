import numpy as np
import matplotlib.pyplot as plt

# Parâmetros físicos (os mesmos do primeiro código)
L = 1.0       # Comprimento do pêndulo (m)
g = 9.8       # Aceleração da gravidade (m/s²)
theta0 = 0.1  # Ângulo inicial (rad)
omega0 = 0.0  # Velocidade angular inicial (rad/s)

# Parâmetros da simulação (os mesmos)
dt = 0.01     # Passo de tempo (s)
t_max = 10.0  # Tempo total de simulação (s)
n_steps = int(t_max / dt)  # Número de passos

# Arrays para armazenar os resultados (como no primeiro código)
t = np.linspace(0, t_max, n_steps)
theta_num = np.zeros(n_steps)  # Solução numérica
omega = np.zeros(n_steps)

# Condições iniciais
theta_num[0] = theta0
omega[0] = omega0

# Método de Euler-Cromer (igual ao primeiro código)
for i in range(n_steps - 1):
    omega[i+1] = omega[i] - (g/L) * np.sin(theta_num[i]) * dt
    theta_num[i+1] = theta_num[i] + omega[i+1] * dt

# --- NOVO: Solução analítica para pequenos ângulos ---
theta_anal = theta0 * np.cos(np.sqrt(g/L) * t)  # θ(t) = θ₀·cos(√(g/L)·t)

# Plot comparativo (modificado do primeiro código)
plt.figure(figsize=(10, 5))
plt.plot(t, theta_num, 'b', label='Solução Numérica (Euler-Cromer)')
plt.plot(t, theta_anal, 'r--', label='Solução Analítica (Pequenos Ângulos)')
plt.title('Comparação entre Solução Numérica e Analítica\n(θ₀ = 0.1 rad)')
plt.xlabel('Tempo (s)')
plt.ylabel('Ângulo (rad)')
plt.legend()
plt.grid(True)
plt.show()

# Cálculo e plot do erro (novo)
error = theta_num - theta_anal
plt.figure(figsize=(10, 4))
plt.plot(t, error, 'g')
plt.title('Diferença entre Solução Numérica e Analítica')
plt.xlabel('Tempo (s)')
plt.ylabel('Erro (rad)')
plt.grid(True)
plt.show()