
import numpy as np
import matplotlib.pyplot as plt

y0 = 800
vT_sl = 60.0
vT_pa = 5.0

mv = 1.225

# a) Tempo para chegar ao solo com o paraquedas fechado


t0 = 0 
tf = 22
dt = 0.0001
g=9.8

D_sl = g/vT_sl**2

t=np.arange(t0,tf,dt)
y_sl = np.empty(np.size(t))
v_sl = np.empty(np.size(t))
a_sl = np.empty(np.size(t))

y_sl[0] = y0
v_sl[0] = 0
for i in range(np.size(t)-1):
    a_sl[i] = -g-D_sl*v_sl[i]* np.abs(v_sl[i])
    v_sl[i+1] = v_sl[i] + a_sl[i]*dt
    y_sl[i+1] = y_sl[i] + v_sl[i]*dt


plt.plot(t,y_sl, 'b-')
plt.xlabel("Tempo, t (s)")
plt.ylabel("Posição, y (m)")
plt.show()

izero = np.size(y_sl) - np.size(y_sl[y_sl<0])
tzero = t[izero]
vzero = v_sl[izero]
print("Tempo de retorno à origem, tzero = ", tzero, "s")
print("Velocidade no retorno à origem = ", vzero)


# b) Alinea anterior mas considerando que o paraquedas fica aberto 10s depois do salto


t1=np.arange(t0,10,dt)

D_sl = g/vT_sl**2

y_sl = np.empty(np.size(t1))
v_sl = np.empty(np.size(t1))
a_sl = np.empty(np.size(t1))

y_sl[0] = y0
v_sl[0] = 0

for i in range(np.size(t1)-1):
    a_sl[i] = -g-D_sl*v_sl[i]* np.abs(v_sl[i])
    v_sl[i+1] = v_sl[i] + a_sl[i]*dt
    y_sl[i+1] = y_sl[i] + v_sl[i]*dt

t2=np.arange(10,100,dt)

D_pa = g/vT_pa**2
y_pa = np.empty(np.size(t2))
v_pa = np.empty(np.size(t2))
a_pa = np.empty(np.size(t2))

y_pa[0] = y_sl[-1]
v_pa[0] = v_sl[-1]

for i in range(np.size(t2)-1):
    a_pa[i] = -g-D_pa*v_pa[i]* np.abs(v_pa[i])
    v_pa[i+1] = v_pa[i] + a_pa[i]*dt
    y_pa[i+1] = y_pa[i] + v_pa[i]*dt

plt.plot(t1,y_sl, 'b-')
plt.plot(t2,y_pa, 'b-')
plt.xlabel("Tempo, t (s)")
plt.ylabel("Posição, y (m)")
plt.show()

y = np.concatenate((y_sl,y_pa))
v = np.concatenate((v_sl,v_pa))
t = np.concatenate((t1,t2))


izero = np.size(y) - np.size(y[y<0])
yzero = y[izero]
tzero = t[izero]
vzero = v[izero]
print("Tempo de retorno à origem, tzero = ", tzero, "s")
print("Velocidade no retorno à origem = ", vzero)
