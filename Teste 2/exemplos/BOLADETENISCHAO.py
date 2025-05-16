import numpy as np

massa = 0.057
v0 = 140/3.6
e_cinetica = 0.5 * massa * v0**2
g = 9.8
vT = 100/3.6
teta = (7/180)*np.pi
D = g / vT**2

dt = 0.001
tf = 1
N = int(tf/dt + 0.1)

a = np.zeros((N,2))
v = np.zeros((N, 2))
r = np.zeros((N, 2))

vv = np.zeros(N)

v[0, :] = [np.cos(teta) * v0, np.sin(teta) * v0]

#metodo de Euler
for i in range(N - 1):
    vv[i] = np.sqrt(v[i,0]**2 + v[i, 1]**2)
    a[i, 0] = - D * vv[i] * v[i, 0]
    a[i, 1] = - g - D * vv[i] * v[i, 1]
    v[i + 1, :] = v[i, :] + a[i, :] * dt
    r[i + 1, :] = r[i, :] + v[i, :] * dt

t0 = int(0.0/dt)
t1 = int(0.4 / dt)
t2 = int(0.8 / dt)
alcance = r[-1,0]


def forca_velocidadeX(v_norm, v_x):
    return massa * (- D * v_norm * v_x) * v_x

def forca_velocidadeY(v_norm, v_y):
    return massa * (- D * v_norm * v_y) * v_y

f1 = forca_velocidadeX(vv, v[:,0])
f2 = forca_velocidadeY(vv, v[:,1])

#separei as componente para fazer o produto interno --> (x1,y1).(x2,y2) = x1x2 + y1y2
Integral1 = dt*((f1[0] + f1[t1]) / 2 + np.sum(f1[1:t1]))
Integral2 = dt*((f2[0] + f2[t1]) / 2 + np.sum(f2[1:t1]))
trabalho = Integral1 + Integral2

print("Energia mecanica (0s): ", e_cinetica, "J")  #ISTO SE FOR EM Y = 0!!
print("Energia mecanica (0.4s): ", (0.5 * massa * vv[t1] ** 2) + (massa * g * r[t1, 1]), "J")
print("Energia mecanica (0.8s): ", (0.5 * massa * vv[t2] ** 2) + (massa * g * r[t2, 1]), "J")
print(f"Alcance: {alcance} m")
print(trabalho)
