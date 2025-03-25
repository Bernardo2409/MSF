import numpy as np
import matplotlib.pyplot as plt

def movimento(t, y0, v0, g):
    return y0 + v0 * t - 0.5 * g * t**2

y0 = 0
v0 = 10
g = 9.8
t = np.linspace(0, 2*v0/g, 100)

y = movimento(t, y0, v0, g)

plt.plot(t,y,'-')
plt.xlabel("Tempo (s)")
plt.ylabel("Posição (m)")
plt.show()

#b)
tl_max = v0 / g
y_max = movimento(tl_max, y0, v0, g)
print("Posição máxima: {:.2f}".format(y_max))

#c)
tl0 = 2*(v0/g)
print("Instante que volta a passar em y[0]: {}".format(tl0))

#d)
def euler_resistencia(y0, v0, g, vt, dt, tf):
    N = int(tf/dt) + 1
    t = np.linspace(0, tf, N)
    y = np.zeros(N)
    v = np.zeros(N)

    y[0] = y0
    v[0] = v0
    D = g / (vt**2)

    for i in range(N - 1):
        a = -g - D * v[i] * abs(v[i])
        v[i+1] = v[i] + a * dt
        y[i+1] = y[i] + v[i] * dt
        

    return t, y, v

y0 = 0
v0 = 10
g = 9.8
vt = 100 * 1000 / 3600 
dt = 0.01
tf = 2 * v0 / g  

t, y, v = euler_resistencia(y0, v0, g, vt, dt, tf)

plt.plot(t,y,'-')
plt.xlabel("Tempo (s)")
plt.ylabel("Posição (m)")
plt.show()

y_max = max(y)
print("Altura máxima com resitência do ar(m): {}".format(y_max))

tl0 = t[-1]

print("Tempo que volta a passar em y[0] com resitência do ar: {:.2f}".format(t[-1]))