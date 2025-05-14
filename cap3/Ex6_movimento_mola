import numpy as np
import matplotlib.pyplot as plt

k = 1.0
m = 1.0
w = np.sqrt(k/m)

A = 1.0
phi = 0.0

# a) - Cálculo analítico
def x_analitico(t):
    return A * np.cos(w * t + phi)

def v_analitico(t):
    return -A * w * np.sin(w * t + phi)

t = np.linspace(0, 10, 500)
x_ana = x_analitico(t)
v_ana = v_analitico(t)

# b) Método de Euler
x0 = A * np.cos(phi)
v0 = -A * w * np.sin(phi)

dt = 0.01
tf = 10.0
t2 = np.arange(0, tf, dt)
n = len(t2)  # Número de pontos
x2 = np.zeros(n)
v2 = np.zeros(n)

x2[0] = x0
v2[0] = v0

for i in range(1, n):
    a = -k/m * x2[i-1]
    v2[i] = v2[i-1] + a * dt
    x2[i] = x2[i-1] + v2[i-1] * dt  # Usar v2[i-1] em vez de v2[i]

# Posição
plt.subplot(2, 1, 1)
plt.plot(t2, x2, 'b-', label='Solução Numérica')
plt.plot(t, x_ana, 'r--', label='Solução Analítica')
plt.title('Comparação entre Soluções Numérica e Analítica')
plt.ylabel('Posição (m)')
plt.legend()
plt.grid(True)

# Velocidade
plt.subplot(2, 1, 2)
plt.plot(t2, v2, 'b-', label='Solução Numérica')
plt.plot(t, v_ana, 'r--', label='Solução Analítica')
plt.xlabel('Tempo (s)')
plt.ylabel('Velocidade (m/s)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()