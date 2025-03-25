import numpy as np
import matplotlib.pyplot as plt

t0 = 0.0 
tf = 3.0 
y0 = 0.0 
v0 = 10.0 
vT = 100 * 1000 / 3600 
dt = 0.0001 
g = 9.8 

D = g / vT ** 2

t = np.arange(t0, tf, dt) 
y = np.empty(np.size(t)) 
v = np.empty(np.size(t)) 
a = np.empty(np.size(t)) 

y[0] = y0
v[0] = v0

for i in range(np.size(t) - 1):
    a[i] = - g - D * v[i] * np.abs(v[i])
    v[i+1] = v[i] + a[i] * dt
    y[i+1] = y[i] + v[i] * dt


plt.plot(t, y, 'b-')
plt.plot(t, y0 + v0 * t - 0.5 * g * t**2, 'r-')
plt.grid(axis = "y", color = 'black')
plt.xlabel("Tempo, t (s)")
plt.ylabel("Posição, y (m)")
plt.show()

# e - estimar os instantes da altura maxima e do retorno a posicao inicial

imax = np.argmax(y)
tmax = t[imax]
print("Tempo na altura máxima, tmax = ", tmax, "s")
print("Altura máxima, y = ", y[imax], "m")

izero = np.size(y) - np.size(y[y<0]) #posicao em que volta a origem
tzero = t[izero]
print("Tempo de retorno à orígem, tzero = ", tzero, "s")