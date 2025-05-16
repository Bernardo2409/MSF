
import matplotlib.pyplot as plt
import numpy as np


vt = 20
g = -9.8


dt = 0.0001


t1 = np.arange(0,100, dt)
y1 = np.zeros(t1.size)
x1 = np.zeros(t1.size)
vx1 = np.zeros(t1.size)
vy1 = np.zeros(t1.size)


x1[0] = 0
y1[0] = 3
vx1[0] = 30
vy1[0] = 0
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
vx1 = vx1[:i+2]
vy1 = vy1[:i+2]

plt.plot(x1,y1, label="Com Resistência do ar")
plt.xlabel("x(t) (m)")
plt.ylabel("y(t) (m)")

plt.legend()
plt.grid()



#ALCANCE E TEMPO ASSOCIADO
talc1 = t1[np.where(x1 == x1[-2])][0]
print("alcance: " + str(x1[-2]) + "\nTempo até chegar ao solo -> " + str(talc1))

#altura da bola para x = 12
print("\nAltura da bola para x = 12: " + str(y1[np.where(x1 >= 12)][0]))

plt.show()


#RESPOSTA ÀS QUESTÕES
# a bola encontra -se a uma altura de 2.03 metros quando passa pela rede, logo passa pela rede
# a bola cai no chão a uma distância de 19.70m, sendo portanto superior a 18,4 metros concluindo-se que o serviço não é válido 