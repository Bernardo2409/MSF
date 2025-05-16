# As seguintes librarias serão necessárias para resolver os exercícios
import numpy as np
import matplotlib.pyplot as plt

# Valores dados
g = 9.8  # Aceleração gravítica na terra

mu = 0.04  # Coeficiente de resistência do alcatrão
rho_ar = 1.225  # Densidade do ar
A = 2  # Área frontal do carro
m = 2000  # Massa do carro
C_res = 0.25  # Coeficiente de resistência do ar

potencia = -30_000

x0 = 0
v0 = 20

# Parâmetros
dt = 0.001
t0 = 0
tf = 200

# Inclinação em radianos
incl = np.radians(5)


# Esta função calcula a aceleração a partir da velocidade atual do carro
def accel(v):
    # Aceleração pela potência do carro
    accel_p = potencia / (m * v)
    # Aceleração pela resistência do ar
    accel_res = -C_res / (2 * m) * A * rho_ar * np.abs(v) * v
    # Aceleração pelo atrito
    accel_atrito = -(mu * np.cos(incl) * g * np.abs(v) / v) / m
    # Aceleração pelo peso
    accel_peso = np.sin(incl) * g
    # Aceleração total
    return accel_p + accel_res + accel_atrito + accel_peso


# Número de passos/iterações
#
# + 0.1 para garantir que não há arrendodamentos
# para baixo
n = int((tf - t0) / dt + 0.1)

t = np.zeros(n + 1)
x = np.zeros(n + 1)
v = np.zeros(n + 1)
a = np.zeros(n + 1)

# Valores iniciais
a[0] = accel(v0)
v[0] = v0
x[0] = x0
t[0] = t0

for i in range(n):
    a[i + 1] = accel(v[i])
    v[i + 1] = v[i] + a[i] * dt
    x[i + 1] = x[i] + v[i] * dt
    t[i + 1] = t[i] + dt

# a) Evolução temporal da posição e velocidade em gráficos
plt.plot(t, x, "r", label="Posição")
plt.xlabel("t (s)")
plt.ylabel("x(t) (m)")
plt.title("Posição carro")
plt.show()

plt.plot(t, v, "g", label="Velocidade")
plt.xlabel("t (s)")
plt.ylabel("v(t) (m/s)")
plt.title("Velocidade carro")
plt.show()

# b) Tempo que leva a descer 2km

for i in range(n):
    # Procurar os zeros com a posição modificada
    if x[i] == 2000 or (x[i] - 2000) * (x[i + 1] - 2000) < 0:
        idx = i
        break

x2000 = x[idx]
t2000 = t[idx]

# Resultado: O carro atinge os 2km (≈ 1999.977692641462) aos 83.10800000003277 segundos
print(f"O carro atinge os 2km (≈ {x2000}) aos {t2000} segundos")

# c) O trabalho feito pelo motor nesta viagem
# O trabalho pode ser obtido como a variação cinética entre o ponto inicial e final

v2000 = v[idx]
v0 = v[0]

E_c0 = 1 / 2 * m * v0**2
E_c1 = 1 / 2 * m * v2000**2

diff_E_c = E_c1 - E_c0

# Resultado: O trabalho feito pelo motor do carro foi de 548909.275743847
print(f"O trabalho feito pelo motor do carro foi de {diff_E_c}")

# e) A diferença da bateria na subida e na descida
W_subir = 463418.3486216352
Bat_restorado = diff_E_c / 2
Bat_diff = Bat_restorado - W_subir

# Resultado: A diferença na bateria foi de -188963.71074971167
print(f"A diferença na bateria foi de {Bat_diff}")
