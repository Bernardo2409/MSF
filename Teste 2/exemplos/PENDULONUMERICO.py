import numpy as np
import matplotlib.pyplot as plt

# Parâmetros físicos
L = 1.0      # Comprimento do pêndulo (m)
g = 9.8      # Aceleração da gravidade (m/s²)

# Condições iniciais
theta0 = 0.1  # Ângulo inicial (rad)
omega0 = 0.0  # Velocidade angular inicial (rad/s)

# Parâmetros da simulação
dt = 0.01     # Passo de tempo (s)
t_max = 10.0  # Tempo total de simulação (s)
n_steps = int(t_max / dt)  # Número de passos

# Arrays para armazenar os resultados
t = np.linspace(0, t_max, n_steps)
theta = np.zeros(n_steps)
omega = np.zeros(n_steps)

# Condições iniciais
theta[0] = theta0
omega[0] = omega0

# Método de Euler-Cromer
for i in range(n_steps - 1):
    omega[i+1] = omega[i] - (g/L) * np.sin(theta[i]) * dt
    theta[i+1] = theta[i] + omega[i+1] * dt

# Plot dos resultados
plt.figure(figsize=(10, 5))
plt.plot(t, theta, label='Ângulo (rad)')
plt.plot(t, omega, label='Velocidade angular (rad/s)')
plt.title('Movimento do Pêndulo Simples - Método de Euler-Cromer')
plt.xlabel('Tempo (s)')
plt.ylabel('θ(t) e ω(t)')
plt.legend()
plt.grid(True)
plt.show()
