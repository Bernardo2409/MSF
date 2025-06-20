import numpy as np
import matplotlib.pyplot as plt

# Parâmetros
g = 9.8  # m/s²
L = 0.5  # m
theta0 = 0.1  # rad
dtheta0 = 0.5  # rad/s

# Frequência angular natural
omega0 = np.sqrt(g / L)

# Tempo
t_max = 10  # s
dt = 0.001  # Passo menor para melhor precisão
t = np.arange(0, t_max, dt)

# Arrays para armazenar as soluções
theta = np.zeros_like(t)
dtheta = np.zeros_like(t)

# Condições iniciais
theta[0] = theta0
dtheta[0] = dtheta0

# Método de Euler para resolver a EDO
for i in range(1, len(t)):
    dtheta[i] = dtheta[i-1] - (g / L) * theta[i-1] * dt
    theta[i] = theta[i-1] + dtheta[i-1] * dt

# Solução analítica para comparação
theta_analytical = theta0 * np.cos(omega0 * t) + (dtheta0 / omega0) * np.sin(omega0 * t)

# Amplitude teórica
amplitude_teorica = np.sqrt(theta0**2 + (dtheta0 / omega0)**2)

# Amplitude numérica (máximo valor absoluto de theta)
amplitude_numerica = np.max(np.abs(theta))

# Período teórico
periodo_teorico = 2 * np.pi / omega0

# Período numérico (tempo entre dois máximos consecutivos)
# Encontrar os picos (máximos locais)
peaks = np.where((theta[1:-1] > theta[:-2]) & (theta[1:-1] > theta[2:]))[0] + 1
if len(peaks) >= 2:
    periodo_numerico = t[peaks[1]] - t[peaks[0]]
else:
    periodo_numerico = periodo_teorico

# Gráfico
plt.figure(figsize=(10, 6))
plt.plot(t, theta, label='Solução Numérica', color='blue')
plt.plot(t, theta_analytical, '--', label='Solução Analítica', color='red')
plt.xlabel('Tempo (s)')
plt.ylabel('Ângulo (rad)')
plt.title('Movimento do Pêndulo Simples')
plt.legend()
plt.grid()
plt.show()

print(f"Amplitude teórica: {amplitude_teorica:.4f} rad")
print(f"Amplitude numérica: {amplitude_numerica:.4f} rad")
print(f"Período teórico: {periodo_teorico:.4f} s")
print(f"Período numérico: {periodo_numerico:.4f} s")