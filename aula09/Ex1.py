import numpy as np
import matplotlib.pyplot as plt

#Tempo
dt = 0.001
tf = 5.0 #s
t0 = 0.0 #s
t = np.arange(t0,tf,dt)
Nt = np.size(t) #número de passos de tempo

#Parâmetros
N = 2 # número de esferas
d = 0.1
l = 10 * d
m = 0.3 #kg
k = 10**7 #N/m^2
g = 9.8 #m/s^2
x0 = np.arange(0, N, 1) * d # posição inicial das esferas (m)

# Grandezas fisicas
x_arr = np.zeros((N, Nt)) # matriz de posições
v_arr = np.zeros((N, Nt)) # matriz de velocidades
a_arr = np.zeros((N, Nt)) # matriz de acelerações

#Condições iniciais
x_arr[:,0] = x0
x_arr[0,0] = -5 * d
v_arr[:,0] = np.zeros(N) # velocidade inicial

def acc_toque(dx,d):
    # calcular a aceleração de uma esfera devido ao contacto com a esfera à sua direita
    k = 1e7
    q = 2.0
    if dx<d:
        a = (-k*abs(dx-d)**q)/m
    else:
        a = 0.0
    return a

def acc_i(i,x):
    #calcular a aceleração de esfera i, cuja posicao de equilibrio é d*i
    a = 0

    if i>0: # a primeira esfera não tem vizinho à sua esquerda
        a -= acc_toque(x[i] - x[i-1],d)
    if i < (N-1): # a última esfera não tem vizinho à sua direita
        a += acc_toque(x[i+1] - x[i], d)

    # aceleração de gravidade, afeta todas as esferas
    a -= g*(x[i]-d*i)/l
    return a

#Metodo Euler-Cromer
for i in range(np.size(t)-1):
    for j in range(0, N):
        a_arr[j,i] = acc_i(j,x_arr[:,i])
        
    v_arr[:,i+1] = v_arr[:,i] + a_arr[:,i] * dt
    x_arr[:,i+1] = x_arr[:,i] + v_arr[:,i+1] * dt

#Representação gráfica
for i in range(0, N):
    plt.plot(t, x_arr[i,:], label='Esfera ' + str(i+1))


plt.legend()
plt.show()