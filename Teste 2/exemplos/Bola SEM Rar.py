import matplotlib.pyplot as plt
import numpy as np

v0 = 100/3.6            #velocidade inicial m/s
angle = 10/180*np.pi
v0x = v0*np.cos(angle)
v0y = v0*np.sin(angle)
vt = 100/3.6
g = -9.8
x0 = 0
y0 = 0
dt = 0.0001

t = np.arange(0,100, dt)
x = np.zeros(t.size)
y = np.zeros(t.size)
vx = np.zeros(t.size)
vy = np.zeros(t.size)


x[0] = x0
y[0] = y0
vx[0] = v0x
vy[0] = v0y

#metodo de Euler
for i in range(0, t.size-1):
    vx[i+1] = vx[i]# velocidade no instante
    vy[i+1] = vy[i] + g*dt # velocidade no instante
    x[i+1] = x[i] + vx[i] * dt # posiçao no instante
    y[i+1] = y[i] + vy[i] * dt # posiçao no instante
    if y[i+1] < 0:
        break

t = t[:i+2]
x = x[:i+2]
y = y[:i+2]
vx = vx[:i+2]
vy = vy[:i+2]


plt.plot(x,y, label = "Sem Resistência do ar")
plt.xlabel("x(t) (m)")
plt.ylabel("y(t) (m)")

plt.legend()
plt.grid()


print("\nSem resistencia do ar")

#ALTURA MAXIMA E TEMPO ASSOCIADO
taltmax =t[np.where(y == np.max(y))][0]
print("alt max: " + str(np.max(y)) + " Tempo -> " + str(taltmax) + "\n") 


#ALCANCE E TEMPO ASSOCIADO
talc = t[np.where(x == x[-2])][0]
print("alcance: " + str(x[-2]) + " Tempo -> " + str(talc))


plt.show()