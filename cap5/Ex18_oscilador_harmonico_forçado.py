import numpy as np
import matplotlib.pyplot as plt

# Dados
m = 1.0
k = 1.0
b = 0.05
F0 = 7.5
wf = 1.0

x0_a = 4.0 #m
v0_0 = 0.0 #m/s

t_start = 0
t_end = 100
dt = 0.01
t = np.arange(t_start, t_end, dt)

def F_ext(t):
    return F0 * np.cos(wf * t)

x_a = np.zeros_like(t)
v_a = np.zeros_like(t)
x_a[0] = x0_a
v_a[0] = v0_0

for i in range(1, len(t)):
    a = (-k * x_a[i-1] - b * v_a[i-1] + F_ext(t[i-1])) / m
    v_a[i] = v_a[i-1] + a * dt
    x_a[i] = x_a[i-1] + v_a[i-1] * dt

#Gráfico da posição em função do tempo

plt.figure(figsize=(10, 5))
plt.plot(t, x_a)
plt.title("Lei do movimento - x0 = 4m, v0 = 0 m/s")
plt.xlabel("Tempo (s)")
plt.ylabel("Posição (m)")
plt.grid(True)
plt.show()

#Analisar apenas o regime estacionário (últimos 20 segundos)

t_estacionario = t[-int(20/dt):]
x_estacionario = x_a[-int(20/dt):]

maximos=  []

for i in range(1, len(x_estacionario) - 1):
    if x_estacionario[i] > x_estacionario[i-1] and x_estacionario[i] > x_estacionario[i+1]:
        maximos.append(x_estacionario[i])

amplitude = np.mean(maximos)
print(f"Amplitudeno no regime estacionário: {amplitude:.4f} m")



# Calcular período médio
indices_maximos = [i for i in range(1, len(x_estacionario)-1) 
                  if x_estacionario[i] > x_estacionario[i-1] and x_estacionario[i] > x_estacionario[i+1]]
periodos = np.diff(t_estacionario[indices_maximos])
periodo_medio = np.mean(periodos)
print(f"Período no regime estacionário: {periodo_medio:.4f} s")
print(f"Frequência angular da força externa: {wf:.4f} rad/s")
print(f"Frequência angular natural do sistema: {np.sqrt(k/m):.4f} rad/s")


# Lei do movimento, com v0 = -4m/s e x0 = -2m

x0_c = -2.0 #m
v0_c = -4.0 #m/s

x_c = np.zeros_like(t)
v_c = np.zeros_like(t)
x_c[0] = x0_c
v_c[0] = v0_c

for i in range(1, len(t)):
    a = (-k * x_c[i-1] - b * v_c[i-1] + F_ext(t[i-1])) / m
    v_c[i] = v_c[i-1] + a * dt
    x_c[i] = x_c[i-1] + v_c[i-1] * dt

plt.figure(figsize=(12, 6))
plt.plot(t, x_c)
plt.title("Lei do movimento - x0 = -2m, v0 = -4 m/s")
plt.xlabel("Tempo (s)")
plt.ylabel("Posição (m)")
plt.grid(True)
plt.show()

# d) Amplitude e periodo no regime estacionario

x_estacionario_c = x_c[-int(20/dt):]

maximos_c = []
for i in range(1, len(x_estacionario_c) - 1):
    if x_estacionario_c[i] > x_estacionario_c[i-1] and x_estacionario_c[i] > x_estacionario_c[i+1]:
        maximos_c.append(x_estacionario_c[i])

amplitude_c = np.mean(maximos_c)
print(f"Amplitude no regime estacionário (x0 = -2m, v0 = -4 m/s): {amplitude_c:.4f} m")

# Calcular período médio
indices_maximos_c = [i for i in range(1, len(x_estacionario_c)-1)
                      if x_estacionario_c[i] > x_estacionario_c[i-1] and x_estacionario_c[i] > x_estacionario_c[i+1]]

periodos_c = np.diff(t_estacionario[indices_maximos_c])
periodo_medio_c = np.mean(periodos_c)
print(f"Período no regime estacionário (x0 = -2m, v0 = -4 m/s): {periodo_medio_c:.4f} s")

#Energia mecanica ao longo do tempo

E_a = np.zeros_like(t)
E_c = np.zeros_like(t)

for i in range(len(t)):
    E_a[i] = 0.5 * k * x_a[i]**2 + 0.5 * m * v_a[i]**2
    E_c[i] = 0.5 * k * x_c[i]**2 + 0.5 * m * v_c[i]**2

plt.figure(figsize=(12, 6))
plt.plot(t, E_a, label="x0  = 4m, v0  = 0m/s")
plt.plot(t, E_c, label="x0 = -2m, v0 = -4m/s")
plt.title("Energia Mecânica ao longo do tempo")
plt.xlabel("Tempo (s)")
plt.ylabel("Energia Mecânica (J)")
plt.legend()
plt.grid(True)
plt.show()


# Verificar se a energia é constante no regime estacionário
E_estacionario_a = E_a[-int(20/dt):]
E_estacionario_c = E_c[-int(20/dt):]

print("\nVariação da energia no regime estacionário:")
print(f"Caso a: Média = {np.mean(E_estacionario_a):.4f} J, Desvio padrão = {np.std(E_estacionario_a):.4f} J")
print(f"Caso c: Média = {np.mean(E_estacionario_c):.4f} J, Desvio padrão = {np.std(E_estacionario_c):.4f} J")