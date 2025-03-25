import numpy as np
import matplotlib.pyplot as plt

#Fmag = 172 * Area (= pi*r**2) * densidade do ar * r * vetor w * vetor v
t0 = 0
tf = 1.0
dt = 0.001
g = 9.8

r0 = np.array([0.0,0.0,23.8])
v0 = np.array([25.0,5,-50])
m = 0.45 #kg
raio = 0.11
A = np.pi * raio**2
rho =  1.225
w = 390.0

t = np.arange(t0,tf,dt)

a = np.zeros([3, np.size(t)])

v = np.zeros([3, np.size(t)])
v[:,0] = v0

r = np.zeros([3, np.size(t)])
r[:,0] = r0

for i in range(np.size(t) - 1):
    a[0, i] = +A * rho * raio * w * v[2,i] / (2*m)
    a[1, i] = -g
    a[2, i] = - A * rho * raio * w * v[0,i] / (2*m)
    v[:, i + 1] = v[:, i] + a[:, i] * dt
    r[:, i + 1] = r[:, i] + v[:, i] * dt

plt.plot(r[0,:],r[1,:],'b-')
plt.xlabel("Posição horizontal, r_x [m]")
plt.ylabel("Posição vertical, r_y [m]")
plt.show()

ixzero = np.size(r[0,r[0,:]>=0])
txzero = t[ixzero]
print("Tempo correspondente ao cruzamento da linha de fundo, txzero = {:.2f}".format(txzero))
print("x = {:.2f} m".format(r[0,ixzero]))
print("y = {:.2f} m".format(r[1,ixzero]))
print("z = {:.2f} m".format(r[2,ixzero]))