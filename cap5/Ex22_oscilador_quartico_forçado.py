import numpy as np
import matplotlib.pyplot as plt

# Parâmetros do sistema
m = 1.0       # massa (kg)
beta = 1.0    # constante quártica (N/m³)
b = 0.05      # coeficiente de amortecimento (kg/s)
alpha = 0.25  # coeficiente adicional (N/m²)
F0 = 7.5      # amplitude da força externa (N)
w = 1.0       # frequência da força externa (rad/s)

# Funções do sistema
def U(x):
    return beta * x**4

def F(x, v, t):
    return -4 * beta * x**3 - b * v + F0 * np.cos(w * t)

# Implementação do RK4
def rk4_step(f, t, y, dt):
    k1 = f(t, y)
    k2 = f(t + dt/2, y + dt/2 * k1)
    k3 = f(t + dt/2, y + dt/2 * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + dt * (k1 + 2*k2 + 2*k3 + k4) / 6

def quartic_oscillator(t, y):
    x, v = y
    dxdt = v
    dvdt = F(x, v, t)/m
    return np.array([dxdt, dvdt])

# Configuração da simulação
dt = 0.001
t_max = 80
n_steps = int(t_max/dt)
t_values = np.linspace(0, t_max, n_steps)

# Condições iniciais
x0_1 = 2.999
x0_2 = 3.001  # Perturbação de 0.001 m
v0 = 0

# Simulação para duas condições iniciais ligeiramente diferentes
def simulate(x0):
    y = np.array([x0, v0])
    x_values = np.zeros(n_steps)
    v_values = np.zeros(n_steps)
    x_values[0] = y[0]
    v_values[0] = y[1]
    
    for i in range(1, n_steps):
        y = rk4_step(quartic_oscillator, t_values[i-1], y, dt)
        x_values[i] = y[0]
        v_values[i] = y[1]
    return x_values

x1 = simulate(x0_1)
x2 = simulate(x0_2)

# Cálculo da divergência
difference = np.abs(x1 - x2)

# Encontrar o tempo onde a diferença se torna significativa
threshold = 0.1  # 10 cm de diferença
critical_index = np.where(difference > threshold)[0]
if len(critical_index) > 0:
    t_critical = t_values[critical_index[0]]
else:
    t_critical = t_max

# Visualização
plt.figure(figsize=(12, 6))


# Trajetórias individuais

plt.plot(t_values, x1, 'b-', label='x₀=2.999m')
plt.plot(t_values, x2, 'r-', label='x₀=3.001m')
plt.title('Trajetórias do Oscilador')
plt.xlabel('Tempo (s)')
plt.ylabel('Posição (m)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print(f"Tempo crítico de previsibilidade: {t_critical:.2f} segundos")