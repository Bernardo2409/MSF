import numpy as np
import matplotlib.pyplot as plt

# Parâmetros do sistema
m = 70.0  # massa (kg)
P = 0.4 * 735.5  # Potência em Watts (0.4 cv)
v0 = 1.0  # velocidade inicial (m/s)
theta = np.radians(5)  # inclinação em radianos
k = m * 9.81 / (100/3.6)**2  # coeficiente de arrasto
g = 9.81  # gravidade

# Configuração da simulação
dt = 0.1  # passo de tempo (s)
t_max = 600  # tempo máximo (s)
n_steps = int(t_max/dt)  # número de passos
t = np.linspace(0, t_max, n_steps)  # vetor de tempo

# Arrays para armazenar resultados
v = np.zeros(n_steps)
dist = np.zeros(n_steps)
v[0] = v0

# Método de Euler para resolver a EDO
for i in range(n_steps - 1):
    dvdt = (P/v[i] - k*v[i]**2 - m*g*np.sin(theta)) / m
    v[i+1] = v[i] + dvdt * dt
    dist[i+1] = dist[i] + v[i] * dt

# Encontrar velocidade terminal (últimos 10% da simulação)
v_terminal = np.mean(v[-int(n_steps*0.1):])
print(f"a) Velocidade terminal na subida: {v_terminal:.2f} m/s ({v_terminal*3.6:.2f} km/h)")

# Tempo para percorrer 2 km
idx_2km = np.argmax(dist >= 2000)
t_2km = t[idx_2km]
print(f"b) Tempo para percorrer 2 km na subida: {t_2km:.1f} s ({t_2km/60:.1f} minutos)")

# Gráficos
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(t, v)
plt.axhline(v_terminal, color='r', linestyle='--', label='Velocidade terminal')
plt.xlabel('Tempo (s)')
plt.ylabel('Velocidade (m/s)')
plt.title('Evolução da Velocidade')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(t, dist/1000)
plt.axhline(2, color='g', linestyle='--', label='2 km')
plt.xlabel('Tempo (s)')
plt.ylabel('Distância (km)')
plt.title('Distância Percorrida')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()