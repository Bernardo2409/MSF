import numpy as np
import matplotlib.pyplot as plt

# Dados
m = 1.0
k = 1.0
b = 0.16
F0 = 2.0
wf = 2.0

x0 = 4.0
v0 = 0.0

t0 = 0.0
tf = 200.0
dt = 0.01
n = int((tf - t0) / dt) + 1
t = np.linspace(t0, tf, n)
x = np.zeros(n)
v = np.zeros(n)

# Condições iniciais
x[0] = x0
v[0] = v0

for i in range(n-1):
    F = -k * x[i] - b * v[i] + F0 * np.cos(wf * t[i])  # Força total
    a = F / m  # Aceleração
    v[i+1] = v[i] + a * dt
    x[i+1] = x[i] + v[i] * dt


plt.figure(figsize=(10, 5))
plt.plot(t, x, 'b-', linewidth=1)
plt.title('Lei do movimento do oscilador harmônico forçado e amortecido')
plt.xlabel('Tempo (s)')
plt.ylabel('Posição (m)')
plt.grid(True)
plt.show()

# b) - Amplitude e periodo no regime estacionario

t_estacionario = t[t>30]
x_estacionario = x[t>30]

maximos = []

for i in range(1, len(x_estacionario)-1):
    if x_estacionario[i] > x_estacionario[i-1] and x_estacionario[i] > x_estacionario[i+1]:
        maximos.append(x_estacionario[i])
amplitude = np.mean(maximos)
print(f"Amplitude no regime estacionário: {amplitude:.2f} m")

#Calculamos o periodo encontrando o tempo entre maximos consecutivos

indices_maximos = []
for i in range(1, len(x_estacionario)-1):
    if x_estacionario[i] > x_estacionario[i-1] and x_estacionario[i] > x_estacionario[i+1]:
        indices_maximos.append(i)

if len(indices_maximos) > 1:
    periodos = np.diff(t_estacionario[indices_maximos])
    periodo = np.mean(periodos)
    print(f"Período no regime estacionário: {periodo:.4f} s")
    print(f"Frequência angular medida: {2*np.pi/periodo:.4f} rad/s (deve ser próxima de wf = {wf} rad/s)")
else:
    print("Não foram encontrados máximos suficientes para calcular o período")

# Configuração da simulação para a parte c)
wf_values = np.linspace(0.2, 2.0, 30)  # Valores de frequência a testar
amplitudes = []  # Armazenar amplitudes para cada frequência

for wf in wf_values:
    # Reinicializar arrays para cada frequência
    x = np.zeros(n)
    v = np.zeros(n)
    x[0] = x0
    v[0] = v0
    
    # Simulação temporal
    for i in range(n-1):
        F = -k * x[i] - b * v[i] + F0 * np.cos(wf * t[i])
        a = F / m
        v[i+1] = v[i] + a * dt
        x[i+1] = x[i] + v[i] * dt
    
    # Calcular amplitude no regime estacionário
    x_est = x[t > 30]  # Considerar apenas o regime estacionário
    maximos = []
    for i in range(1, len(x_est)-1):
        if x_est[i] > x_est[i-1] and x_est[i] > x_est[i+1]:
            maximos.append(x_est[i])
    
    if len(maximos) > 0:
        amplitude = np.mean(maximos)
    else:
        amplitude = np.max(np.abs(x_est))
    
    amplitudes.append(amplitude)

# Plot do gráfico de amplitude vs frequência
plt.figure(figsize=(10, 5))
plt.plot(wf_values, amplitudes, 'b-', linewidth=2, marker='o', markersize=5)
plt.title('Resposta em Amplitude do Oscilador Forçado')
plt.xlabel('Frequência da Força Externa (rad/s)')
plt.ylabel('Amplitude (m)')
plt.grid(True)

# Destacar a frequência de ressonância teórica
w0 = np.sqrt(k/m)  # Frequência natural
plt.axvline(x=w0, color='r', linestyle='--', label=f'Frequência natural ($\omega_0$ = {w0:.2f} rad/s)')
plt.legend()

plt.show()