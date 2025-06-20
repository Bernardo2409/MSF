import numpy as np
import matplotlib.pyplot as plt

# Parâmetros
k = 1.0  # N/m
m = 1.0  # kg
x0 = 4.0  # m
v0 = 0.0  # m/s

# Tempo
t = np.linspace(0, 10, 1000)

# Solução analítica
x_analytical = x0 * np.cos(np.sqrt(k/m) * t)
v_analytical = -x0 * np.sqrt(k/m) * np.sin(np.sqrt(k/m) * t)

# Energia mecânica
E_analytical = 0.5 * m * v_analytical**2 + 0.5 * k * x_analytical**2

# Gráficos
plt.figure(figsize=(12, 8))

# Movimento
plt.subplot(2, 1, 1)
plt.plot(t, x_analytical, label='x(t) = 4 cos(t)')
plt.xlabel('Tempo (s)')
plt.ylabel('Posição (m)')
plt.legend()
plt.grid()


# Energia Mecânica
plt.subplot(2, 1, 2)
plt.plot(t, E_analytical, label='E = 8 J (constante)')
plt.xlabel('Tempo (s)')
plt.ylabel('Energia (J)')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# Amplitude e período
amplitude = x0
period = 2 * np.pi * np.sqrt(m / k)

print(f"Amplitude do movimento: {amplitude} m")
print(f"Período do movimento: {period:.3f} s")
print(f"Energia mecânica: {E_analytical[0]} J (constante)")