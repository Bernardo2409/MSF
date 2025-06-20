import numpy as np
import matplotlib.pyplot as plt

# Parâmetros
k = 1.0       # N/m
alpha = 1.0   # N/m^2
beta = 0.05   # kg/s
F0 = 7.5      # N
omega = 1.0   # rad/s
m = 1.0       # kg

# Tempo de simulação
T_total = 700

# a) x0 = 3 m, v0 = 0 m/s, dt = 0.01
# Verificação com passo maior

T_a1 = T_total
dt_a1 = 0.01
N_a1 = int(T_a1 / dt_a1)
t_a1 = np.linspace(0, T_a1, N_a1)

def simular_oscilador(x0, v0, dt, N, t):
    x = np.zeros(N)
    v = np.zeros(N)
    x[0] = x0
    v[0] = v0
    for i in range(N - 1):
        a = (-k * x[i] * (1 + 2 * alpha * x[i]**2) - beta * v[i] + F0 * np.cos(omega * t[i])) / m
        v[i+1] = v[i] + a * dt
        x[i+1] = x[i] + v[i+1] * dt
    return x, v

x_a1, v_a1 = simular_oscilador(3.0, 0.0, dt_a1, N_a1, t_a1)
print(f"a) dt = 0.01: x(700 s) = {x_a1[-1]:.7f} m")

# a) dt = 0.001, valor de referência
T_a2 = T_total
dt_a2 = 0.001
N_a2 = int(T_a2 / dt_a2)
t_a2 = np.linspace(0, T_a2, N_a2)

x_a2, v_a2 = simular_oscilador(3.0, 0.0, dt_a2, N_a2, t_a2)
print(f"a) dt = 0.001: x(700 s) = {x_a2[-1]:.7f} m")

# b) Regime estacionário (últimos 100 s de a2)
x_regime = x_a2[-int(100/dt_a2):]
v_regime = v_a2[-int(100/dt_a2):]
t_regime = t_a2[-int(100/dt_a2):]

amp_max = np.max(x_regime)
amp_min = np.min(x_regime)

from numpy import diff, where
zeros = where(np.diff(np.signbit(x_regime)))[0]
tempos_zeros = t_regime[zeros]
periodos = 2 * np.diff(tempos_zeros)
periodo_medio = np.mean(periodos)

print(f"b) Amplitude: {amp_max:.4f} m / {amp_min:.4f} m")
print(f"b) Período médio: {periodo_medio:.2f} s")

plt.figure()
plt.plot(x_regime, v_regime)
plt.title('Espaço da Fase (Regime estacionário - b)')
plt.xlabel('x (m)')
plt.ylabel('v (m/s)')
plt.grid(True)
plt.tight_layout()
plt.show()

# c) x0 = -3 m, v0 = -3 m/s
x_c, v_c = simular_oscilador(-3.0, -3.0, dt_a2, N_a2, t_a2)

# d) Regime estacionário de c)
x_regime_c = x_c[-int(100/dt_a2):]
v_regime_c = v_c[-int(100/dt_a2):]
t_regime_c = t_a2[-int(100/dt_a2):]

amp_max_c = np.max(x_regime_c)
amp_min_c = np.min(x_regime_c)

zeros_c = where(np.diff(np.signbit(x_regime_c)))[0]
tempos_zeros_c = t_regime_c[zeros_c]
periodos_c = 2 * np.diff(tempos_zeros_c)
periodo_medio_c = np.mean(periodos_c)

plt.figure()
plt.plot(x_regime_c, v_regime_c)
plt.title('Espaço da Fase (Regime estacionário - d)')
plt.xlabel('x (m)')
plt.ylabel('v (m/s)')
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"d) Amplitude: {amp_max_c:.4f} m / {amp_min_c:.4f} m")
print(f"d) Período médio: {periodo_medio_c:.2f} s")

# d) Análise de Fourier
fft_c = np.fft.fft(x_regime_c)
freqs = np.fft.fftfreq(len(fft_c), dt_a2)

plt.figure()
plt.plot(freqs[:len(freqs)//2], np.abs(fft_c[:len(freqs)//2]))
plt.title('Análise de Fourier - Regime estacionário (d)')
plt.xlabel('Frequência (Hz)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.tight_layout()
plt.show()

# e) Energia mecânica
def energia_mec(x, v):
    Ep = 0.5 * k * x**2 + 0.25 * alpha * k * x**4
    Ec = 0.5 * m * v**2
    return Ep + Ec

E_a = energia_mec(x_a2, v_a2)
E_c = energia_mec(x_c, v_c)

plt.figure()
plt.plot(t_a2, E_a, label='Energia a)')
plt.plot(t_a2, E_c, label='Energia c)')
plt.title('Energia Mecânica Total')
plt.xlabel('t (s)')
plt.ylabel('Energia (J)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# f) Caracterização
print("\nf) Caracterização:")
print("- O sistema apresenta oscilações forçadas com regime estacionário após transientes.")
print(f"- A amplitude no regime estacionário (b): [{amp_min:.4f}, {amp_max:.4f}] m")
print(f"- A amplitude no regime estacionário (d): [{amp_min_c:.4f}, {amp_max_c:.4f}] m")
print(f"- O período médio estimado (b): {periodo_medio:.2f} s")
print(f"- O período médio estimado (d): {periodo_medio_c:.2f} s")
print("- A energia não é constante devido à força externa e amortecimento.")
print("- A resposta contém múltiplas frequências (ver análise de Fourier).")