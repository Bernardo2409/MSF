import numpy as np
import matplotlib.pyplot as plt

# Parâmetros do problema
v0 = 200 * 1000 / 3600  # Velocidade inicial (m/s)
vt = 6.8  # Velocidade terminal (m/s)
g = 9.8  # Aceleração da gravidade (m/s²)
dt = 0.01  # Intervalo de tempo (s)
t_final = 5  # Tempo final (s)

# Método de Euler para velocidade
def euler_velocidade(v0, vt, g, dt, t_final):
    t = np.arange(0, t_final + dt, dt)
    v = np.zeros_like(t)
    v[0] = v0  # Corrigido para começar com v0
    for i in range(len(t) - 1):
        v[i + 1] = v[i] + g * (1 - v[i] / vt) * dt
    return t, v

# Cálculo da velocidade usando o método de Euler
t_euler, v = euler_velocidade(v0, vt, g, dt, t_final)

# Gráfico da aceleração
a_t = g * (1 - v / vt)
plt.figure(figsize=(16, 5))
plt.subplot(1, 2, 1)
plt.plot(t_euler, a_t, label="Aceleração a(t)", color='b')
plt.axhline(y=0, color='k', linestyle='--', linewidth=1)
plt.xlabel("Tempo (s)")
plt.ylabel("Aceleração (m/s²)")
plt.title("Aceleração do volante em função do tempo")
plt.legend()
plt.grid()

# Gráfico da velocidade
plt.subplot(1, 2, 2)
plt.plot(t_euler, v, label="Velocidade v(t)", color='r')
plt.axhline(y=vt, color='k', linestyle='--', linewidth=0.8, label="Velocidade terminal")
plt.xlabel("Tempo (s)")
plt.ylabel("Velocidade (m/s)")
plt.title("Velocidade do volante em função do tempo")
plt.legend()
plt.grid()
plt.show()

# c) Velocidade ao fim de 1 segundo
v_1s = v[np.argmin(np.abs(t_euler - 1))]  # Encontrar o valor mais próximo de t=1
v_1s_kmh = v_1s * 3.6
print(f"Velocidade ao fim de 1 segundo: {v_1s:.2f} m/s ({v_1s_kmh:.2f} km/h)")

# d) Tempo em que a velocidade é reduzida a 50% do valor inicial
v_metade = v0 / 2
index_metade = np.argmin(np.abs(v - v_metade))  # Encontrar o tempo em que v(t) é 50% de v0
tempo_metade = t_euler[index_metade]
print(f"Tempo para reduzir a velocidade a 50% de v0: {tempo_metade:.2f} segundos")
