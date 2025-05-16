import matplotlib.pyplot as plt
import numpy as np

# Constantes
g = 9.8 # COM MAG G É POSITIVO
m = 0.45  # massa da bola
r = 0.11  # raio da bola
PAr = 1.225  # densidade do ar
#vt = 100 / 3.6  # velocidade terminal
dt = 0.001

t = np.arange(0, 5 + dt, dt)
v0 = 100 / 3.6  # velocidade inicial
ang0 = np.radians(16)  # ângulo
A = np.pi * (r)**2  # área da bola
D = 0.0127  # coeficiente de resistência do ar

mag = 0.5 * A * PAr * r

Rx = np.zeros(t.size)
Ry = np.zeros(t.size)
Rz = np.zeros(t.size)

Vx = np.zeros(t.size)
Vy = np.zeros(t.size)
Vz = np.zeros(t.size)

# Posição inicial
Rx[0] = 0
Ry[0] = 0
Rz[0] = 0

# Velocidade inicial
Vx[0] = v0 * np.cos(ang0)
Vy[0] = v0 * np.sin(ang0)
Vz[0] = 0.0

# Rotação inicial
Wx = 0
Wy = 0
Wz = -10

# Parâmetros do gol
distancia_baliza = 20  # distância até o gol (m)
altura_baliza = 2.40  # altura do gol (m)
largura_baliza = 7.50

# Método de Euler
for i in range(0, t.size - 1):
    v = np.sqrt(Vx[i]**2 + Vy[i]**2 + Vz[i]**2)
    
    # Consoante a rotação
    amagx = mag * np.cross([Wx, Wy, Wz], [Vx[i], Vy[i], Vz[i]])[0] / m
    amagy = mag * np.cross([Wx, Wy, Wz], [Vx[i], Vy[i], Vz[i]])[1] / m
    amagz = mag * np.cross([Wx, Wy, Wz], [Vx[i], Vy[i], Vz[i]])[2] / m
    
    # Acelerações
    ax = -D * Vx[i] * abs(v) + amagx
    ay = - g - D * Vy[i] * abs(v) + amagy
    az = -D * Vz[i] * abs(v) + amagz
    
    # Atualizar velocidades
    Vx[i + 1] = Vx[i] + ax * dt
    Vy[i + 1] = Vy[i] + ay * dt
    Vz[i + 1] = Vz[i] + az * dt
    
    # Atualizar posições
    Rx[i + 1] = Rx[i] + Vx[i] * dt
    Ry[i + 1] = Ry[i] + Vy[i] * dt
    Rz[i + 1] = Rz[i] + Vz[i] * dt

# Verificar altura máxima
for i in range(0, t.size - 1):   
    if Ry[i + 1] < Ry[i]:
        print(f"Y Máximo: {Ry[i]:.2f} m")
        print(f"Tempo para altura máxima: {t[i]:.2f} s")
        break

# Verificar alcance
for i in range(0, t.size - 1):
    if Ry[i + 1] < 0:
        print(f"Alcance: {Rx[i]:.2f} m")
        print(f"Tempo para alcance máximo: {t[i]} s")
        break

# Verificar se é golo
indice_baliza = np.argmax(Rx >= distancia_baliza)  # Índice onde Rx >= 20 m
altura_na_baliza = Ry[indice_baliza]
tempo_na_baliza = t[indice_baliza]  # Altura da bola na linha do gol

if 0 < altura_na_baliza <= altura_baliza:
    resultado = "GOLO!"
    cor_resultado = 'green'
else:
    resultado = "NÃO É GOLO!"
    cor_resultado = 'red'

# Exibir resultado do golo
print(f"\nNa linha da baliza (x = {distancia_baliza} m):")
print(f"Altura na baliza: {altura_na_baliza:.2f} m")
print(f"Tempo para chegar à baliza: {tempo_na_baliza:.2f} s")
print(f"Resultado: {resultado}")

# Ajustar os vetores ao tamanho calculado
t = t[:i]
Vx = Vx[:i]
Vy = Vy[:i]
Vz = Vz[:i]

Rx = Rx[:i]
Ry = Ry[:i]
Rz = Rz[:i]

# Plotar a trajetória
plt.plot(Rx, Ry, label="Trajetória da bola")
plt.axvline(x=distancia_baliza, color='gray', linestyle='--', label="Linha da baliza")
plt.axhline(y=altura_baliza, color='red', linestyle='--', label="Altura da baliza")
plt.scatter(distancia_baliza, altura_na_baliza, color=cor_resultado, label=f"Altura na baliza: {altura_na_baliza:.2f}m ({resultado})")
plt.xlabel("x(t) (m)")
plt.ylabel("y(t) (m)")
plt.legend()
plt.grid()
plt.show()