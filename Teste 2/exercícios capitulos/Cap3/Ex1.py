import numpy as np
import matplotlib.pyplot as plt

# --- Parâmetros iniciais ---
v0_kmh = 100                # Velocidade inicial (km/h)
v0 = v0_kmh / 3.6           # Conversão para m/s
theta_deg = 10              # Ângulo de lançamento (graus)
theta = np.radians(theta_deg)
g = 9.8                     # Gravidade (m/s²)
vt_kmh = 100                # Velocidade terminal (km/h)
vt = vt_kmh / 3.6           # Velocidade terminal (m/s)
D = g / vt**2               # Constante de resistência do ar

# Componentes iniciais da velocidade
v0x = v0 * np.cos(theta)
v0y = v0 * np.sin(theta)

# --- Parâmetros da simulação ---
dt = 0.01
t_max = 20
N = int(t_max / dt)
t = np.linspace(0, t_max, N)

# Arrays para sem resistência
x_no_air = np.zeros(N)
y_no_air = np.zeros(N)
vx_no_air = np.full(N, v0x)
vy_no_air = np.zeros(N)
vy_no_air[0] = v0y

# Arrays para com resistência
x_air = np.zeros(N)
y_air = np.zeros(N)
vx_air = np.zeros(N)
vy_air = np.zeros(N)
vx_air[0] = v0x
vy_air[0] = v0y

# --- Simulação sem resistência do ar ---
for i in range(N - 1):
    vy_no_air[i+1] = vy_no_air[i] - g * dt
    x_no_air[i+1] = x_no_air[i] + vx_no_air[i] * dt
    y_no_air[i+1] = y_no_air[i] + vy_no_air[i] * dt
    if y_no_air[i+1] < 0:
        break

# Resultados sem resistência
t_no_air = t[i]
alcance_no_air = x_no_air[i]
altura_max_no_air = np.max(y_no_air)
t_altura_max_no_air = t[np.argmax(y_no_air)]

# --- Simulação com resistência do ar ---
for i in range(N - 1):
    v = np.sqrt(vx_air[i]**2 + vy_air[i]**2)
    ax = -D * v * vx_air[i]
    ay = -g - D * v * vy_air[i]
    vx_air[i+1] = vx_air[i] + ax * dt
    vy_air[i+1] = vy_air[i] + ay * dt
    x_air[i+1] = x_air[i] + vx_air[i] * dt
    y_air[i+1] = y_air[i] + vy_air[i] * dt
    if y_air[i+1] < 0:
        break

# Resultados com resistência
t_air = t[i]
alcance_air = x_air[i]
altura_max_air = np.max(y_air)
t_altura_max_air = t[np.argmax(y_air)]

# --- Mostrar resultados ---
print("=== SEM RESISTÊNCIA DO AR ===")
print(f"Altura máxima: {altura_max_no_air:.2f} m")
print(f"Tempo até à altura máxima: {t_altura_max_no_air:.2f} s")
print(f"Alcance: {alcance_no_air:.2f} m")
print(f"Tempo total de voo: {t_no_air:.2f} s\n")

print("=== COM RESISTÊNCIA DO AR ===")
print(f"Altura máxima: {altura_max_air:.2f} m")
print(f"Tempo até à altura máxima: {t_altura_max_air:.2f} s")
print(f"Alcance: {alcance_air:.2f} m")
print(f"Tempo total de voo: {t_air:.2f} s")

# --- Gráfico comparativo ---
plt.plot(x_no_air[:i+1], y_no_air[:i+1], label='Sem resistência', color='orange')
plt.plot(x_air[:i+1], y_air[:i+1], label='Com resistência', color='blue')
plt.title("Trajetória da bola (comparação)")
plt.xlabel("Distância horizontal (m)")
plt.ylabel("Altura (m)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
