import numpy as np
import matplotlib.pyplot as plt



t0 = 0
v0 = 50
tf = 10
y0 = 0
vT = 100
dt = 0.01
g = 9.8

D = g/vT**2

t = np.arange(t0, tf, dt)
y = np.empty(np.size(t))
v = np.empty(np.size(t))
a = np.empty(np.size(t))

y[0] = y0
v[0] = v0

for i in range(np.size(t)-1):
    a[i] = -g-D * v[i] * np.abs(v[i])
    v[i+1] = v[i] + a[i]*dt
    y[i+1] = y[i] + v[i]*dt

plt.plot(t,y, 'b-')
plt.xlabel("Tempo, t (s)")
plt.ylabel("Posição, y (m)")
plt.show()


#estimar os instantes da altura maxima e do retorno a posicao inicial

imax = np.argmax(y)
tmax = t[imax]

print("Tempo na altura máxima, tmax = ", tmax, "s")
print("Altura máxima, y = ", y[imax], "m")

#estimar a altura no momento da explosao

texp = int(5/dt)
yexp = y[texp]

print("Altura no momento da explosão, y = ", yexp, "m")