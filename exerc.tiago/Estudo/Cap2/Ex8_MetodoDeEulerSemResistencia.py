import numpy as np
import matplotlib.pyplot as plt

def Euler(y0,v0,t0,tf,dt,g):

    #N = Número de passos 
    N = int((tf - t0) / (dt) + 0.1) #(0.1 para arredondar para cima)

    #Iniciar variáveis
    y = np.zeros(N+1)
    v = np.zeros(N+1)
    t = np.zeros(N+1)

    #Valores iniciais
    v[0] = v0
    t[0] = t0
    y[0] = y0

    #Preencher vetores
    for i in range(N):
        y[i+1] = y[i] - dt*v[i] #Distância percorrida
        t[i+1] = t[i] + dt
        v[i+1] = v[i] + g * dt
    
    return t,y,v


g = 9.8

#a)
#A aceleração instantânea é apenas a fórmula derivada da velocidade em função do tempo

#b)
y0 = 100
v0 = 0
t0 = 0
tf = 4
dt = 0.001
t,y,v = Euler(y0,v0,t0,tf,dt,g)

plt.plot(t,y,label="Posição",color="blue")
plt.legend()
plt.show()
plt.plot(t,v,label="Velocidade",color="red")
plt.legend()
plt.show()

idx_3s = np.where(t >= 3.0)[0][0]
v3 = v[idx_3s]
print("A velocidade no intante 3s = {:.2f} m/s".format(v3))

#c)
y0 = 100
v0 = 0
t0 = 0
tf = 4
dt = 0.1
t10,y10,v10 = Euler(y0,v0,t0,tf,dt,g)

idx_3s = np.where(t10 >= 3.0)[0][0]
v3 = v10[idx_3s]
print("A velocidade no intante 3s com passo 10x menor= {:.2f} m/s".format(v3))

#d)
#Quanto menor o passo mais aproximado fica a solução

#e)
idx_2s = np.where(t >= 2.0)[0][0]
y2 = y[idx_2s]
print("A posição no intante 2s (y0 = 100m)= {:.2f} m".format(y2))

y0 = 0
v0 = 0
t0 = 0
tf = 4
dt = 0.01
t_0,y_0,v_0 = Euler(y0,v0,t0,tf,dt,g)
idx_2s = np.where(t_0 >= 2.0)[0][0]
y2 = y_0[idx_2s]
print("A posição no intante 2s (y0 = 0m)= {:.2f} m".format(y2))