import numpy as np
import matplotlib.pyplot as plt

# PARTE 1: Diagrama de energia potencial
m = 1
Xeq = 2
K = 1
E_total = 1

x_ep = np.linspace(-5, 5, 1000)

def ep(x):
    return 0.5 * K * (np.abs(x) - Xeq)**2  # Corrigida para energia potencial do oscilador duplo

def Fx(x):
    if x > 0:
        return -K * (x - Xeq)  # Região x > 0
    else:
        return K * (-x - Xeq)  # Região x < 0

Ep_values = ep(x_ep)

plt.figure(figsize=(10, 5))
plt.plot(x_ep, Ep_values, color='blue', label='Energia Potencial')
plt.xlabel('x (m)')
plt.ylabel('Energia Potencial (J)')
plt.title('Diagrama de Energia Potencial')
plt.grid(True)
plt.legend()
plt.show()

# PARTE 2: Simulação com E_total = 1J 
dt = 0.01
t1 = np.arange(0, 10, dt)
x1 = np.zeros_like(t1)
v1 = np.zeros_like(t1)

x1[0] = Xeq + np.sqrt(2 * E_total / K)
v1[0] = 0.0

for i in range(1, len(t1)):
    a = Fx(x1[i-1]) / m
    v1[i] = v1[i-1] + a * dt
    x1[i] = x1[i-1] + v1[i-1] * dt

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(t1, x1, label='x(t)', color='blue')
plt.axhline(Xeq, color='green', linestyle='--', label='x_eq = 2 m')
plt.xlabel('Tempo (s)')
plt.ylabel('Posição (m)')
plt.title('Movimento (E = 1J)')
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(t1, v1, label='v(t)', color='orange')
plt.xlabel('Tempo (s)')
plt.ylabel('Velocidade (m/s)')
plt.title('Velocidade (E = 1J)')
plt.grid()
plt.legend()
plt.show()

# PARTE 3: SOLUÇÃO PARA E_total = 0.75J 


E_total = 0.75

# 1. Amplitude e pontos de retorno (relativos a Xeq = 0)
A = np.sqrt(2 * E_total / K)  # 1.225 m
print(f"Amplitude: A = {A:.3f} m (relativa a Xeq = 0)")

# 2. Frequência (oscilador harmônico simples)
omega = np.sqrt(K / m)  # 1 rad/s
freq = omega / (2 * np.pi)
print(f"Frequência angular: ω = {omega:.3f} rad/s")
print(f"Frequência: f = {freq:.3f} Hz")

# 3. Lei do movimento analítica (Xeq = 0)
def x_analitico(t):
    return A * np.cos(omega * t)  # x(t) = 1.225 cos(t)

def v_analitico(t):
    return -A * omega * np.sin(omega * t)

# 4. Simulação numérica com condições iniciais x₀=1.225m, v₀=0m/s
t2 = np.arange(0, 10, dt)
x2 = np.zeros(np.size(t2))
v2 = np.zeros(np.size(t2))

x2[0] = A  
v2[0] = 0  

for i in range(1, len(t2)):
    F = -K * x2[i-1]  
    v2[i] = v2[i-1] + (F/m) * dt
    x2[i] = x2[i-1] + v2[i-1] * dt

# 5. Gráficos comparativos
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(t2, x_analitico(t2), label='Analítico: 1.225 cos(t)', color='blue')
plt.plot(t2, x2, '--', label='Numérico (Euler)', color='red')
plt.xlabel('Tempo (s)')
plt.ylabel('Posição (m)')
plt.title('Lei do Movimento (E=0.75J)')
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(x2, v2, label='Diagrama de Fase', color='green')
plt.xlabel('Posição (m)')
plt.ylabel('Velocidade (m/s)')
plt.title('Espaço de Fase')
plt.grid()
plt.legend()
plt.show()

#PARTE 4: Simulação com E_total = 1.5J

E_total = 1.5

# 1. Amplitude e pontos de retorno (relativos a Xeq = 0)
A = np.sqrt(2 * E_total / K)  # 1.225 m
print(f"Amplitude: A = {A:.3f} m (relativa a Xeq = 0)")

# 2. Frequência (oscilador harmônico simples)
omega = np.sqrt(K / m)  # 1 rad/s
freq = omega / (2 * np.pi)
print(f"Frequência angular: ω = {omega:.3f} rad/s")
print(f"Frequência: f = {freq:.3f} Hz")

# 3. Lei do movimento analítica (Xeq = 0)
def x_analitico(t):
    return A * np.cos(omega * t)  # x(t) = 1.225 cos(t)

def v_analitico(t):
    return -A * omega * np.sin(omega * t)

# 4. Simulação numérica com condições iniciais x₀=1.225m, v₀=0m/s
t2 = np.arange(0, 10, dt)
x2 = np.zeros(np.size(t2))
v2 = np.zeros(np.size(t2))

x2[0] = A  
v2[0] = 0  

for i in range(1, len(t2)):
    F = -K * x2[i-1]  
    v2[i] = v2[i-1] + (F/m) * dt
    x2[i] = x2[i-1] + v2[i-1] * dt

# 5. Gráficos comparativos
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(t2, x_analitico(t2), label='Analítico: 1.225 cos(t)', color='blue')
plt.plot(t2, x2, '--', label='Numérico (Euler)', color='red')
plt.xlabel('Tempo (s)')
plt.ylabel('Posição (m)')
plt.title('Lei do Movimento (E=0.75J)')
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(x2, v2, label='Diagrama de Fase', color='green')
plt.xlabel('Posição (m)')
plt.ylabel('Velocidade (m/s)')
plt.title('Espaço de Fase')
plt.grid()
plt.legend()
plt.show()