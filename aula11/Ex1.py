import numpy as np
import matplotlib.pyplot as plt

m = 1.0        
k = 1.0        
b = 0.05       
F0 = 7.5       

x0 = 4.0      
v0 = 0.0       

dt = 0.001
T = 200
N = int(T / dt)
t = np.linspace(0, T, N)

# Frequência externa para a parte (a) e (b)
omega_f = 0.5

# Método de Euler
def simular(omega_f):
    x = np.zeros(N)
    v = np.zeros(N)
    x[0] = x0
    v[0] = v0

    for i in range(N - 1):
        a = (-k * x[i] - b * v[i] + F0 * np.cos(omega_f * t[i])) / m
        v[i + 1] = v[i] + a * dt
        x[i + 1] = x[i] + v[i] * dt

    return x, v

# a) Lei do movimento
x, v = simular(omega_f)

plt.figure(figsize=(10, 4))
plt.plot(t, x)
plt.title("a) Movimento do oscilador forçado (wf = 0.5 rad/s)")
plt.xlabel("Tempo (s)")
plt.ylabel("Posição (m)")
plt.grid()
plt.tight_layout()
plt.show()

# b) Amplitude e período no regime estacionário (últimos 60s)

def encontrar_maximos(t, x):
    maximos_t = []
    maximos_x = []
    for i in range(1, len(x) - 1):
        if x[i - 1] < x[i] > x[i + 1]:
            maximos_t.append(t[i])
            maximos_x.append(x[i])
    return np.array(maximos_t), np.array(maximos_x)

t_regime = t[-int(50/dt):]
x_regime = x[-int(50/dt):]
maximos_t, maximos_x = encontrar_maximos(t_regime, x_regime)

amplitude = np.mean(maximos_x)
periodo = np.mean(np.diff(maximos_t))

print("b) Resultados no regime estacionário (wf = 0.5 rad/s):")
print(f"   Amplitude média = {amplitude:.4f} m")
print(f"   Período médio   = {periodo:.4f} s")

# (c) Variação da amplitude com wf entre 0.2 e 2.0 rad/s
omega_vals = np.linspace(0.2, 2.0, 10)
amplitudes = []

for omega in omega_vals:
    x, _ = simular(omega)
    x_regime = x[-int(50/dt):]
    t_regime = t[-int(50/dt):]
    maximos_t, maximos_x = encontrar_maximos(t_regime, x_regime)
    if len(maximos_x) > 0:
        amplitudes.append(np.mean(maximos_x))
    else:
        amplitudes.append(0)


# Gráfico da amplitude vs. frequência
plt.figure(figsize=(8, 4))
plt.plot(omega_vals, amplitudes)
plt.title("c) Amplitude em função da frequência angular")
plt.xlabel("Frequência angular ωf (rad/s)")
plt.ylabel("Amplitude (m)")
plt.grid()
plt.tight_layout()
plt.show()


# Frequência da maior amplitude
indice_max = np.argmax(amplitudes)
omega_resonancia = omega_vals[indice_max]
amp_max = amplitudes[indice_max]

print(f"   Frequência angular = {omega_resonancia:.4f} rad/s")
print(f"   Amplitude máxima = {amp_max:.4f} m")