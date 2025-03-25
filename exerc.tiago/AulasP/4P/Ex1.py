import numpy as np
import matplotlib.pyplot as plt

def Euler(tf, x0, vx0, dt, g):
    n = int(tf / dt) + 1 
    x = np.zeros(n)
    vx = np.zeros(n)
    t = np.zeros(n)
    
    x[0] = x0
    vx[0] = vx0

    for i in range(n - 1):
        aceler = g  # Queda livre
        x[i+1] = x[i] - vx[i] * dt
        vx[i+1] = vx[i] + aceler * dt
        t[i+1] = t[i] + dt
    
    return x, vx, t

tf = 4.0
x0 = 0
vx0 = 0
dt = 0.01
g = 9.8
x,vx,t = Euler(tf,x0,vx0,dt,g)
v_exact = g * 3
idx3 = int(3/dt)
print("Velocidade no instante 3 (s): {:.2f}".format(vx[idx3]))
print("Velocidade exata instante 3 (s): {:.2f}".format(v_exact))

#Quanto menor a diferença de abcissas, mais próximo o resultado

print("Posição no instante 3 (s): {:.2f}".format(x[idx3]))

pos_exata = -1/2*9.8*3.0**2
print("Posição exata no instante 3 (s): {:.2f}".format(pos_exata))

#Novamente, quanto menor a diferença de abcissas, mais próximo o resultado

plt.plot(t,vx,'-')
plt.xlabel("Tempo (s)")
plt.ylabel("Velocidade (m/s**2)")
plt.show()

plt.plot(t,x,'-')
plt.xlabel("Tempo (s)")
plt.ylabel("Posição (m)")
plt.show()
