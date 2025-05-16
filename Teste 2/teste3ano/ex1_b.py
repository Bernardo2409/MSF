import matplotlib.pyplot as plt
import numpy as np

#Constantes
g = 9.8
m = 0.057  #massa da bola
r = 0.034  #raio da bola
PAr = 1.225  #densidade do ar
vt = 20 #velocidade terminal
dt = 0.001

t = np.arange(0, 5+dt, dt)
A = np.pi*(r)**2   #area da bola
D = g / vt**2 #coeficiente de resistência do ar

mag = 0.5*A*PAr*r

Rx = np.zeros(t.size)
Ry = np.zeros(t.size)
Rz = np.zeros(t.size)

Vx = np.zeros(t.size)
Vy = np.zeros(t.size)
Vz = np.zeros(t.size)

#Posição inicial
Rx[0] = 0
Ry[0] = 3
Rz[0] = 0

#Velocidade inicial
Vx[0] = 30
Vy[0] = 0
Vz[0] = 0.0

#Rotação inicial
Wx = 0
Wy = 0
Wz = -60

#Método de Euler
for i in range(0, t.size-1):
    v = np.sqrt(Vx[i]**2+Vy[i]**2+Vz[i]**2)
    
    #Consoante a rotaçao -> mudar isto
    amagx = mag* np.cross([Wx,Wy,Wz],[Vx[i],Vy[i],Vz[i]])[0] / m 
    amagy = mag* np.cross([Wx,Wy,Wz],[Vx[i],Vy[i],Vz[i]])[1] / m 
    amagz = mag* np.cross([Wx,Wy,Wz],[Vx[i],Vy[i],Vz[i]])[2] / m 
    
    #ha de estar certo para os casos gerais
    ax = -D * Vx[i] * abs(v) + amagx
    ay = - g - D * Vy[i] * abs(v) + amagy
    az = -D * Vz[i] * abs(v) + amagz
    
    Vx[i+1] = Vx[i] + ax * dt
    Vy[i+1] = Vy[i] + ay * dt
    Vz[i+1] = Vz[i] + az * dt
    
    Rx[i+1] = Rx[i] + Vx[i] *dt
    Ry[i+1] = Ry[i] + Vy[i] * dt
    Rz[i+1] = Rz[i] + Vz[i] * dt
    

for i in range(0, t.size-1):
    #alcance
    if(Ry[i+1] < 0):
        print("Alcance: ",Rx[i+1])
        print("Tempo para alcance máxima: ", t[i+1])
        break

print("\nAltura da bola para x = 12: " + str(Ry[np.where(Rx >= 12)][0]))

t = t[:i]
Vx = Vx[:i]
Vy = Vy[:i]
Vz = Vz[:i]

Rx = Rx[:i]
Ry = Ry[:i]
Rz = Rz[:i]



plt.plot(Rx,Ry)
plt.legend()
plt.grid()
plt.show()


#RESPOSTA ÀS QUESTÕES
# a bola encontra -se a uma altura de 1.82metros quando passa pela rede, logo passa pela rede
# a bola cai no chão a uma distância de 18.17m, sendo portanto inferior a 18,4 metros  e superior a 12m concluindo-se que o serviço é válido