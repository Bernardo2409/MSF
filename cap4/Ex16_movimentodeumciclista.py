import numpy as np
import matplotlib.pyplot as plt

# Constantes
P = 0.4 * 735.5  # potência em watts
m = 70           # massa em kg
Cd = 0.9         # coeficiente de arrasto
A = 0.5          # área frontal em m^2
rho = 1.225      # densidade do ar em kg/m^3
v0 = 1.0         # velocidade inicial em m/s

# Velocidade terminal (teórica)
v_terminal = ((2 * P) / (Cd * rho * A))**(1/3)
print(f"Velocidade terminal: {v_terminal:.2f} m/s")

# Simulação temporal com Euler
dt = 0.1  # passo de tempo em segundos
t_max = 1000  # duração total da simulação
n = int(t_max / dt)

t = np.linspace(0, t_max, n)
v = np.zeros(n)
x = np.zeros(n)
v[0] = v0

for i in range(n - 1):
    F_res = 0.5 * Cd * rho * A * v[i]**2
    P_disp = P / v[i] if v[i] > 0 else 0  # evitar divisão por zero
    a = (P_disp - F_res) / m
    v[i+1] = v[i] + a * dt
    x[i+1] = x[i] + v[i] * dt

# a) Velocidade terminal já calculada acima

# b) Tempo para atingir 90% da velocidade terminal
v_90 = 0.9 * v_terminal
idx_90 = np.argmax(v >= v_90)
tempo_90 = t[idx_90]
print(f"Tempo para atingir 90% da velocidade terminal: {tempo_90:.2f} s")

# c) Tempo para percorrer 2 km
distancia_alvo = 2000  # metros
idx_dist = np.argmax(x >= distancia_alvo)
tempo_dist = t[idx_dist]
print(f"Tempo para percorrer 2 km: {tempo_dist:.2f} s")

# Gráfico da velocidade e da posição
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(t, v)
plt.axhline(v_terminal, color='r', linestyle='--', label='Velocidade terminal')
plt.axhline(v_90, color='g', linestyle='--', label='90% Vel. terminal')
plt.xlabel("Tempo (s)")
plt.ylabel("Velocidade (m/s)")
plt.legend()
plt.title("Evolução da velocidade")

plt.subplot(1, 2, 2)
plt.plot(t, x)
plt.axhline(distancia_alvo, color='orange', linestyle='--', label='2 km')
plt.xlabel("Tempo (s)")
plt.ylabel("Distância (m)")
plt.legend()
plt.title("Distância percorrida")

plt.tight_layout()
plt.show()
