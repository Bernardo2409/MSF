import numpy as np
import matplotlib.pyplot as plt

# Parâmetros
g = 9.8
L = 1.0
initial_angles_deg = [5, 10, 20, 30]  # Ângulos iniciais em graus
initial_angles_rad = np.deg2rad(initial_angles_deg)  # Conversão para radianos

# Configuração da simulação
dt = 0.001  # Passo de tempo pequeno para precisão
t_max = 20   # Tempo total de simulação

# Função das EDOs
def f(t, y):
    theta, omega = y
    dtheta_dt = omega
    domega_dt = -(g / L) * np.sin(theta)
    return np.array([dtheta_dt, domega_dt])

# Método Range-Kutta de quarta ordem
def rk4_step(t, y, dt):
    k1 = f(t, y) * dt
    k2 = f(t + dt/2, y + k1/2) * dt
    k3 = f(t + dt/2, y + k2/2) * dt
    k4 = f(t + dt, y + k3) * dt
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6

# Simulação para cada ângulo inicial
results = {}
for theta0_deg, theta0_rad in zip(initial_angles_deg, initial_angles_rad):
    # Condições iniciais: [theta, omega] = [ângulo inicial, 0]
    y = np.array([theta0_rad, 0.0])
    t_values = [0]
    theta_values = [theta0_rad]
    
    # Integração no tempo
    for t in np.arange(dt, t_max, dt):
        y = rk4_step(t, y, dt)
        t_values.append(t)
        theta_values.append(y[0])
    
    # Encontrar os cruzamentos por zero (theta = 0)
    zero_crossings = []
    for i in range(1, len(theta_values)):
        if theta_values[i-1] * theta_values[i] < 0:  # Mudança de sinal
            zero_crossings.append(t_values[i])
    
    # Calcular o período (tempo entre dois cruzamentos consecutivos)
    if len(zero_crossings) >= 2:
        period = 2 * (zero_crossings[1] - zero_crossings[0])  # Período completo
    else:
        period = 2 * np.pi * np.sqrt(L / g)  # Fallback: período linear
    
    results[theta0_deg] = period

# Resultados
print("Períodos calculados:")
for theta0_deg, period in results.items():
    print(f"Ângulo inicial = {theta0_deg}° → Período = {period:.3f} s")