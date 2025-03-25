import matplotlib.pyplot as plt
import numpy as np

mT = 58  
mB = 58 
h0 = 5  
vTb = 6.80  
vTt = 100 * 1000 / 3600  
g = 9.80  
dt = 0.001  

Dt = g / vTt**2
Db = g / vTb**2

t = np.arange(0, 2, dt)  
yt = np.empty(np.size(t))
vt = np.empty(np.size(t))
at = np.empty(np.size(t))
yb = np.empty(np.size(t))
vb = np.empty(np.size(t))
ab = np.empty(np.size(t))

yt[0] = h0
vt[0] = 0
yb[0] = h0
vb[0] = 0

indexT = None
indexB = None

for i in range(np.size(t) - 1):
    #Bola de Tênis
    at[i] = -g - Dt * vt[i] * np.abs(vt[i])
    vt[i+1] = vt[i] + at[i] * dt
    yt[i+1] = yt[i] + vt[i] * dt

    #Volante de Badminton
    ab[i] = -g - Db * vb[i] * np.abs(vb[i])
    vb[i+1] = vb[i] + ab[i] * dt
    yb[i+1] = yb[i] + vb[i] * dt

indexT = -1
valorT = 0 
indexB = -1
valorB = 0

for i in yt:
    if i>0:
        indexT+=1
        valorT = i

for b in yb:
    if b>0:
        indexB +=1
        valorB = vb


print("A bola de Tênis chega ao solo em {:.2f} s".format(t[indexT]))
print("O volante de Badminton chega ao solo em {:.2f} s".format(t[indexB]))

# Gráfico das trajetórias
plt.plot(t[:indexT], yt[:indexT], label="Bola de Tênis")
plt.plot(t[:indexB], yb[:indexB], label="Volante de Badminton")
plt.xlabel("Tempo (s)")
plt.ylabel("Altura (m)")
plt.legend()
plt.grid()
plt.show()
