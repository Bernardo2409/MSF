import numpy as np
import matplotlib.pyplot as plt

# Função para encontrar picos manualmente
def find_peaks(x):
    peaks = []
    for i in range(1, len(x)-1):
        if x[i-1] < x[i] and x[i] > x[i+1]:  # Pico local
            peaks.append(i)
    return np.array(peaks)

# Parâmetros
m = 1.0       # kg
k = 1.0       # N/m
b = 0.05      # kg/s
alpha = 0.002 # N/m²
Fe = 7.5      # N
omega = 1.0   # rad/s

# Função do sistema de EDOs
def f(y, t):
    x, v = y
    dxdt = v
    dvdt = -x * (1 + 2 * alpha * x**2) - b * v + Fe * np.cos(omega * t)
    return np.array([dxdt, dvdt])

# Método de Runge-Kutta de 4ª ordem
def rk4_step(y, t, dt, f):
    k1 = f(y, t)
    k2 = f(y + 0.5 * k1 * dt, t + 0.5 * dt)
    k3 = f(y + 0.5 * k2 * dt, t + 0.5 * dt)
    k4 = f(y + k3 * dt, t + dt)
    return y + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6

# Simulação
t_max = 100.0    # Tempo total
dt = 0.01        # Passo de tempo
n_steps = int(t_max / dt)
t = np.linspace(0, t_max, n_steps)

# (a) Condições iniciais: x(0) = 3 m, v(0) = 0 m/s
y_a = np.zeros((n_steps, 2))
y_a[0] = [3.0, 0.0]  # x(0) = 3, v(0) = 0

for i in range(1, n_steps):
    y_a[i] = rk4_step(y_a[i-1], t[i-1], dt, f)

x_a = y_a[:, 0]
v_a = y_a[:, 1]

# Gráfico para (a)
plt.figure(figsize=(10, 5))
plt.plot(t, x_a, label='x(t)')
plt.xlabel('Tempo (s)')
plt.ylabel('Posição (m)')
plt.title('Lei do movimento para x(0)=3 m, v(0)=0 m/s (RK4)')
plt.grid()
plt.legend()
plt.show()

# (b) Amplitude e período no regime estacionário (últimas oscilações)
t_steady = t[-2000:]  # últimos 20%
x_steady = x_a[-2000:]

# Encontrar picos manualmente
peaks = find_peaks(x_steady)
if len(peaks) >= 2:
    amplitudes = x_steady[peaks]
    amplitude_avg = np.mean(np.abs(amplitudes))
    period = np.mean(np.diff(t_steady[peaks]))
    print(f"(b) Amplitude no regime estacionário: {amplitude_avg:.4f} m")
    print(f"(b) Período no regime estacionário: {period:.4f} s")
else:
    print("Não foram encontrados picos suficientes para calcular amplitude e período.")

# (c) Condições iniciais: x(0) = -2 m, v(0) = -4 m/s
y_c = np.zeros((n_steps, 2))
y_c[0] = [-2.0, -4.0]  # x(0) = -2, v(0) = -4

for i in range(1, n_steps):
    y_c[i] = rk4_step(y_c[i-1], t[i-1], dt, f)

x_c = y_c[:, 0]
v_c = y_c[:, 1]

# Gráfico para (c)
plt.figure(figsize=(10, 5))
plt.plot(t, x_c, label='x(t)')
plt.xlabel('Tempo (s)')
plt.ylabel('Posição (m)')
plt.title('Lei do movimento para x(0)=-2 m, v(0)=-4 m/s (RK4)')
plt.grid()
plt.legend()
plt.show()

# (d) Amplitude e período no regime estacionário (últimas oscilações)
x_steady_c = x_c[-2000:]
peaks_c = find_peaks(x_steady_c)
if len(peaks_c) >= 2:
    amplitudes_c = x_steady_c[peaks_c]
    amplitude_avg_c = np.mean(np.abs(amplitudes_c))
    period_c = np.mean(np.diff(t_steady[peaks_c]))
    print(f"(d) Amplitude no regime estacionário: {amplitude_avg_c:.4f} m")
    print(f"(d) Período no regime estacionário: {period_c:.4f} s")
else:
    print("Não foram encontrados picos suficientes para calcular amplitude e período.")

# (e) Energia mecânica
def energy(x, v):
    return 0.5 * m * v**2 + 3 * x**2 * (1 + alpha * x**2)

E_a = energy(x_a, v_a)
E_c = energy(x_c, v_c)

# Gráfico da energia para (a)
plt.figure(figsize=(10, 5))
plt.plot(t, E_a, label='Energia mecânica')
plt.xlabel('Tempo (s)')
plt.ylabel('Energia (J)')
plt.title('Energia mecânica para x(0)=3 m, v(0)=0 m/s (RK4)')
plt.grid()
plt.legend()
plt.show()

# Gráfico da energia para (c)
plt.figure(figsize=(10, 5))
plt.plot(t, E_c, label='Energia mecânica')
plt.xlabel('Tempo (s)')
plt.ylabel('Energia (J)')
plt.title('Energia mecânica para x(0)=-2 m, v(0)=-4 m/s (RK4)')
plt.grid()
plt.legend()
plt.show()

print("(e) A energia mecânica não é constante devido ao amortecimento e à força externa.")