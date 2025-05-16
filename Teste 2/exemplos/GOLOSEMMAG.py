import matplotlib.pyplot as plt
import numpy as np

v0 = 100 / 3.6  # velocidade inicial m/s
angulo = 16 / 180 * np.pi  # ângulo de lançamento em radianos
v0x = v0 * np.cos(angulo)
v0y = v0 * np.sin(angulo)
#vt = 100 / 3.6  # velocidade terminal
g = -9.8  # aceleração da gravidade SEM MAG G É NEGATIVO
x0 = 0  # posição inicial em x
y0 = 0  # posição inicial em y
dt = 0.0001  # passo de tempo

# Parâmetros do gol
distancia_baliza = 20  # distância até o gol (m)
altura_baliza = 2.40  # altura do gol (m)
largura_baliza = 7.50

# Vetores para armazenar os resultados
t1 = np.arange(0, 100, dt)
y1 = np.zeros(t1.size)
x1 = np.zeros(t1.size)
vx1 = np.zeros(t1.size)
vy1 = np.zeros(t1.size)

# Condições iniciais
x1[0] = x0
y1[0] = y0
vx1[0] = v0x
vy1[0] = v0y
D = 0.0127

# Método de Euler
for i in range(0, t1.size - 1):
    v = np.sqrt(vx1[i]**2 + vy1[i]**2)
    ax = -D * vx1[i] * abs(v)
    ay = g - D * vy1[i] * abs(v)
    
    vx1[i + 1] = vx1[i] + ax * dt  # velocidade no instante
    vy1[i + 1] = vy1[i] + ay * dt  # velocidade no instante
    x1[i + 1] = x1[i] + vx1[i] * dt  # posição no instante
    y1[i + 1] = y1[i] + vy1[i] * dt  # posição no instante
    if y1[i + 1] < 0:
        break

# Ajustar os vetores ao tamanho calculado
t1 = t1[:i + 2]
x1 = x1[:i + 2]
y1 = y1[:i + 2]
vx1 = vx1[:i + 2]
vy1 = vy1[:i + 2]

# Verificar se é golo
indice_baliza = np.argmax(x1 >= distancia_baliza)  # Índice onde x >= 20 m
altura_na_baliza = y1[indice_baliza]  # Altura da bola na linha do gol
tempo_baliza = t1[indice_baliza]

if 0 < altura_na_baliza <= altura_baliza:
    resultado = "GOLO!"
    cor_resultado = 'green'
else:
    resultado = "NÃO É GOLO!"
    cor_resultado = 'red'

# Plotar a trajetória
plt.plot(x1, y1, label="Com Resistência do ar")
plt.axvline(x=distancia_baliza, color='gray', linestyle='--', label="Linha da baliza")
plt.axhline(y=altura_baliza, color='red', linestyle='--', label="Altura da baliza")
plt.scatter(distancia_baliza, altura_na_baliza, color=cor_resultado, label=f"Altura na baliza: {altura_na_baliza:.2f}m ({resultado})")
plt.xlabel("x(t) (m)")
plt.ylabel("y(t) (m)")
plt.legend()
plt.grid()

# Altura máxima e tempo associado
print("\nCom resistência do ar:")
taltmax1 = t1[np.where(y1 == np.max(y1))][0]
print(f"Altura máxima: {np.max(y1):.2f} m" + f"\nTempo até atingir a altura máx: {taltmax1:.2f} s")

# Alcance e tempo associado
talc1 = t1[np.where(x1 == x1[-2])][0]
print(f"Alcance: {x1[-2]:.2f} m" + f"\nTempo até chegar ao solo: {talc1:.2f} s")

# Resultado do golo
print(f"\nNa linha da baliza (x = {distancia_baliza} m):")
print(f"Altura na baliza: {altura_na_baliza:.2f} m")
print(f"Tempo para chegar à baliza: {tempo_baliza:.2f} s")
print(f"Resultado: {resultado}")

plt.show()