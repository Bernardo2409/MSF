
import numpy as np
import matplotlib.pyplot as plt

# Parâmetros
m = 1.0
alpha = 0.15
b = 0.02
F0 = 7.5
omega_f = 1.0

# Equações do sistema (estado: y = [x, v])
def quartic_oscillator(t, y):
    x, v = y
    dxdt = v
    dvdt = (-4 * alpha * x**3 - b * v + F0 * np.cos(omega_f * t)) / m
    return np.array([dxdt, dvdt])

# Método RK4
def rk4_step(f, t, y, dt):
    k1 = f(t, y)
    k2 = f(t + dt/2, y + dt/2 * k1)
    k3 = f(t + dt/2, y + dt/2 * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + dt * (k1 + 2*k2 + 2*k3 + k4) / 6

# Simulação
T_total = 1000         # tempo total
dt = 0.01              # passo de tempo
N = int(T_total / dt)
t_values = np.linspace(0, T_total, N)

x_values = np.zeros(N)
v_values = np.zeros(N)

# Condições iniciais
y = np.array([2.0, 0.0])  # x=2 m, v=0 m/s
x_values[0] = y[0]
v_values[0] = y[1]

# Integração com RK4
for i in range(1, N):
    y = rk4_step(quartic_oscillator, t_values[i-1], y, dt)
    x_values[i] = y[0]
    v_values[i] = y[1]

# Gráfico da posição vs tempo
plt.figure(figsize=(10, 4))
plt.plot(t_values, x_values)
plt.title("Oscilador Quártico Forçado - RK4")
plt.xlabel("Tempo (s)")
plt.ylabel("Posição x (m)")
plt.grid(True)
plt.tight_layout()
plt.show()

# ==============================================
# (b) Propagação da incerteza na posição inicial
# ==============================================

# Condições iniciais ligeiramente diferentes
y1 = np.array([2.000, 0.0])
y2 = np.array([2.001, 0.0])

x1 = np.zeros(N)
x2 = np.zeros(N)
x1[0] = y1[0]
x2[0] = y2[0]

# Evolução com RK4
for i in range(1, N):
    y1 = rk4_step(quartic_oscillator, t_values[i-1], y1, dt)
    y2 = rk4_step(quartic_oscillator, t_values[i-1], y2, dt)
    x1[i] = y1[0]
    x2[i] = y2[0]

# Diferença absoluta entre as posições
delta_x = np.abs(x1 - x2)

# Verificar até quando a diferença se mantém < 0.001 m
limite = 0.001
indice_quebra = np.argmax(delta_x > limite)
tempo_quebra = t_values[indice_quebra] if indice_quebra > 0 else None

# Gráfico da diferença
plt.figure(figsize=(10, 4))
plt.plot(t_values, delta_x, label='|Δx(t)|')
plt.axhline(limite, color='red', linestyle='--', label='Limite de incerteza (0.001 m)')
if tempo_quebra:
    plt.axvline(tempo_quebra, color='orange', linestyle='--', label=f'Perda de previsibilidade ≈ {tempo_quebra:.1f} s')
plt.title('Crescimento da Incerteza na Lei do Movimento')
plt.xlabel('Tempo (s)')
plt.ylabel('Diferença de posição |Δx(t)| (m)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Impressão do resultado
if tempo_quebra:
    print(f"(b) A lei do movimento deixa de ser unívoca por causa da incerteza inicial por volta de t ≈ {tempo_quebra:.1f} s.")
else:
    print(f"(b) A diferença |Δx(t)| permaneceu inferior a 0.001 m durante toda a simulação.")
