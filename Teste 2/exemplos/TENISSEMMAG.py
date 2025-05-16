import matplotlib.pyplot as plt
import numpy as np


vt = 20
g = -9.8
dt = 0.0001

t1 = np.arange(0,100, dt)

#Posições iniciais 3D ou 2D
x1 = np.zeros(t1.size)
y1 = np.zeros(t1.size)
z1 = np.zeros(t1.size)

#Velocidades inciais 3D ou 2D
vx1 = np.zeros(t1.size)
vy1 = np.zeros(t1.size)
vz1 = np.zeros(t1.size)

#Posições iniciais
x1[0] = 0
y1[0] = 3
z1[0] = 0

#Velociade iniciais 

vx1[0] = 30
vy1[0] = 0
vz1[0] = 0

meiodocampo = 12
altura_rede = 1
zonaponto = 18.4

D = -g/(vt**2)

for i in range(0, t1.size-1):
    v = np.sqrt(vx1[i]**2 + vy1[i]**2)
    ax = -D*vx1[i]*abs(v)
    ay = g-D*vy1[i]*abs(v)
    
    vx1[i+1] = vx1[i] + ax*dt# velocidade no instante
    vy1[i+1] = vy1[i] + ay*dt # velocidade no instante
    x1[i+1] = x1[i] + vx1[i] * dt # posiçao no instante
    y1[i+1] = y1[i] + vy1[i] * dt # posiçao no instante
    if y1[i+1] < 0:
        break

t1 = t1[:i+2]
x1 = x1[:i+2]
y1 = y1[:i+2]
z1 = z1[:i+2]
vx1 = vx1[:i+2]
vy1 = vy1[:i+2]
vz1 = vz1[:i+2]

plt.plot(x1,y1, label="Com Resistência do ar")
#plt.axvline(x=meiodocampo, color='black', linestyle='--', label="Meio do campo")
#plt.axhline(y=altura_rede, color='red', linestyle='--', label="Altura da rede")
plt.plot([meiodocampo, meiodocampo], [0, altura_rede], color='blue', linestyle='-', label="Rede")
plt.scatter(zonaponto, 0, color='red', label=f"Ponto (x={zonaponto}, y=0)")
plt.xlabel("x(t) (m)")
plt.ylabel("y(t) (m)")
plt.legend()
plt.grid()

#ALCANCE E TEMPO ASSOCIADO
talc1 = t1[np.where(x1 == x1[-2])][0]
print(f"-Alcance: {x1[-2]:.1f} m"f"\n-Tempo até chegar ao solo: {talc1:.1f} s")
#altura da bola para x = 12
print("-Altura da bola para x = 12: " + str(y1[np.where(x1 >= 12)][0])+" s")
print(f"Como o alcance é aproximadamente 19.7m, ou seja maior que 18.4m, o serviço não é válido")

plt.show()