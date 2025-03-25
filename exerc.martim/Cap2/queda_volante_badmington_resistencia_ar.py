import numpy as np
import matplotlib.pyplot as plt

g = 9.80
vT = 6.8
t0 = 0
tf = 5
v0 =  0
dt = 0.0001
y0 = 0

#a) vt=g/C

#b) 
D = g/vT**2

t = np.arange(t0,tf,dt)
y = np.empty(np.size(t))
v = np.empty(np.size(t))
a = np.empty(np.size(t))

y[0] = y0
v[0] = v0 

for i in range(np.size(t)-1):
    a[i] = g-D*v[i]*np.abs(v[i])
    v[i+1] = v[i] + a[i]*dt
    y[i+1] = y[i] + v[i]*dt


plt.plot(t,v, 'b-')
plt.xlabel("Tempo, t(s)")
plt.ylabel("Velocidade, v(m)")
plt.show()

plt.plot(t,y, 'b-')
plt.xlabel("Tempo, t(s)")
plt.ylabel("Posição, y(m)")
plt.show()

