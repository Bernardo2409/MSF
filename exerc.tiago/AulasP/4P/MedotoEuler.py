import numpy as np

N = 10
x = np.zeros(N+1)
vx = np.zeros(N+1)
t = np.zeros(N+1)
x0 = 0
vx0 = 0
x[0] = x0
vx[0] = vx0
dt = 1
g = 9.8
for i in range(N):
    aceler=g # queda livre
    x[i+1]=x[i]+vx[i]*dt
    vx[i+1]=vx[i]+aceler*dt
    t[i+1]=t[i]+dt
