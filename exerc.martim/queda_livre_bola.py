import numpy as np
import matplotlib.pyplot as plt


#Exercicio 1 - Pratica 4 


t0 = 0
tf = 4.0
x0 = 0
v0 = 0
dt = 0.1

g = 9.8

t = np.arange(t0,tf,dt)
v = np.empty(np.size(t))
x = np.empty(np.size(t))


#a)
v[0]=v0
x[0] = x0

for i in range(np.size(t)-1):
    v[i+1] = v[i] + g*dt
    x[i+1] = x[1] + v[i]*dt

#b)

t=3
i3 = int(3.0/dt)
print("v(t=3) = ", v[i3], "m/s")
print("x(t=3) = ", x[i3], "m")

# c)

t0 = 0
tf = 4.0
x0 = 0
v0 = 0
dt = 0.01

g = 9.8

t = np.arange(t0,tf,dt)
v = np.empty(np.size(t))
x = np.empty(np.size(t))

v[0]=v0
x[0] = x0

for i in range(np.size(t)-1):
    v[i+1] = v[i] + g*dt
    x[i+1] = x[1] + v[i]*dt

t=3
i3 = int(3.0/dt)
print("v(t=3) = ", v[i3], "m/s")
print("x(t=3) = ", x[i3], "m")

# f) --------------------------------------------------

t0 = 0
tf = 4.0
x0 = 0
v0 = 0
dt = 0.001

g = 9.8

t = np.arange(t0,tf,dt)
v = np.empty(np.size(t))
x = np.empty(np.size(t))

v[0]=v0
x[0] = x0

for i in range(np.size(t)-1):
    v[i+1] = v[i] + g*dt
    x[i+1] = x[1] + v[i]*dt

t=3
i3 = int(3.0/dt)
print("v(t=3) = ", v[i3], "m/s")
print("x(t=3) = ", x[i3], "m")


#h) ------------------------------------------------------------

dt = 0.0001

t= np.arange(t0, tf, dt)
v = np.empty(np.size(t))
x = np.empty(np.size(t))

v[0] = v0
x[0] = x0

for i in range(np.size(t)-1):
    v[i+1]=v[i] + g*dt
    x[i+1] = x[i] + v[i]*dt

i2 = int(2.0/dt)

print("x(t=2) = ", x[i2], "m")
print("dx (t=2) =", np.abs(x[i2]-19.6), "m")

passo = np.array([0.1,0.01, 0.001, 0.0001])
desvio = np.array([0.98, 0.098, 0.0098, 0.00098])


plt.loglog(passo, desvio, 'o-')
plt.xlabel("Passo (s)")
plt.ylabel("Desvio da Posição, dx (m)")
plt.show()