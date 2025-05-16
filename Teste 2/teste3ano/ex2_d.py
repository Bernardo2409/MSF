import matplotlib.pyplot as plt
import numpy as np

dt = 0.001
t = np.arange(0,100+dt,dt)

ax = np.zeros(t.size)
vx = np.zeros(t.size)
x = np.zeros(t.size)
fx = np.zeros(t.size)
fy = np.zeros(t.size)


u = 0.004               #coeficiente de resistencia de rolamento
Cres = 0.25              #coeficiente de resistencia do ar
area = 2                #area frontal
Par = 1.225             #densidade do ar
m = 2000                  #massa do ciclista
Potencia = -30000        #294.20 potencia do ciclista passado de hp para w
vx[0] = 20               #velocidade inicial dada pelo empurrão
g = 9.8
P= m*g 
inclinação = 5
Px=P*np.sin(np.radians(inclinação))
Py=P*np.cos(np.radians(inclinação))
N=Py  

for i in range(t.size-1):
    Fcic = Potencia/vx[i] 
    FRes = -(Cres/2)*area*Par*vx[i]**2 
    FRol = u*N          #Força de resistencia ao rolamento Frol= u*N=u*m*g
    F = Fcic + FRes - FRol + Px
    ax[i] = F/m  
    vx[i+1] = vx[i] + ax[i]*dt
    x[i+1] = x[i] + vx[i]*dt

    fx[i] = m*ax[i]*vx[i] 
    fy[i] = 0

    if(x[i] == 2000): #Quando x == 2000m
        print("Tempo que demora a percorrer 2000m:")
        print(t[np.where(x == x[i])])
        Wx = dt*((fx[0] + fx[i+1])*0.5 + np.sum(fx[1:i]))
        Wy = dt*((fy[0] + fy[i+1])*0.5 + np.sum(fy[1:i]))
        print("O trabalho realizado foi {:.2f}J.".format(Wx+Wy))
        break
    
    elif(x[i] > 2000): #Vai buscar o primeiro valor acima de 2000m
        t1 = t[np.where(x == x[i])]
        t2 = t[np.where(x == x[i-1])]
        tmed = (t1+t2)/2
        print("Tempo que demora a percorrer 2000m:")
        print(tmed)
        Wx = dt*((fx[0] + fx[i+1])*0.5 + np.sum(fx[1:i]))
        Wy = dt*((fy[0] + fy[i+1])*0.5 + np.sum(fy[1:i]))
        print("O trabalho realizado foi {:.2f}J.".format(Wx+Wy))
        break


plt.plot(t,vx, label="velocity")
plt.grid()
plt.show()


#Resposta: O carro demora [96.8965] segundos a percorrer 2000m/2km
#O trabalho realizado foi 91795.28 Joules