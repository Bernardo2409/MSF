#ALINEA D NECESSITA DE CORREÇÃO!!!!!!!! 

import numpy as np
import matplotlib.pyplot as plt

y0 = 1*1000
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
print("Velocidade no retorno à origem = ", vzero*3.6)


# b) Tempo para chegar ao solo com o paraquedas aberto

t=np.arange(t0,210,dt)



D_pa = g/vT_pa**2
y_pa = np.empty(np.size(t))
v_pa = np.empty(np.size(t))
a_pa = np.empty(np.size(t))

y_pa[0] = y0
v_pa[0] = 0

for i in range(np.size(t)-1):
    a_pa[i] = -g-D_pa*v_pa[i]* np.abs(v_pa[i])
    v_pa[i+1] = v_pa[i] + a_pa[i]*dt
    y_pa[i+1] = y_pa[i] + v_pa[i]*dt

plt.plot(t,y_pa, 'b-')
plt.xlabel("Tempo, t (s)")
plt.ylabel("Posição, y (m)")
plt.show()

izero = np.size(y_pa) - np.size(y_pa[y_pa<0])
tzero = t[izero]
vzero = v_pa[izero]
print("Tempo de retorno à origem, tzero = ", tzero, "s")
print("Velocidade no retorno à origem = ", vzero*3.6)

# c) Alinea anterior mas considerando que o paraquedas fica aberto 20s depois do salto

t1=np.arange(t0,20,dt)

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

t2=np.arange(20,35,dt)

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

izero = np.size(y_pa) - np.size(y_pa[y_pa<0])
tzero = t[izero] + 20
vzero = v_pa[izero]
print("Tempo de retorno à origem, tzero = ", tzero, "s")
print("Velocidade no retorno à origem = ", vzero*3.6)

# d) Alinea anterior mas considerando a densidade do ar
t1 = np.arange(0, 20, dt)
t2 = np.arange(20, 210, dt)

y_sl = np.empty(np.size(t1))
v_sl = np.empty(np.size(t1))
a_sl = np.empty(np.size(t1))

y_sl[0] = y0
v_sl[0] = 0

for i in range(np.size(t1) - 1):
    coef = 1.225 * np.exp(-0.1378 * y_sl[i] / 1000)  # Correção na equação da densidade
    D_sl = g / vT_sl**2 * coef / 1.225
    a_sl[i] = -g - D_sl * v_sl[i] * np.abs(v_sl[i])
    v_sl[i + 1] = v_sl[i] + a_sl[i] * dt
    y_sl[i + 1] = y_sl[i] + v_sl[i] * dt

y_pa = np.empty(np.size(t2))
v_pa = np.empty(np.size(t2))
a_pa = np.empty(np.size(t2))

y_pa[0] = y_sl[-1]
v_pa[0] = v_sl[-1]

for i in range(np.size(t2) - 1):
    coef = 1.225 * np.exp(-0.1378 * y_pa[i] / 1000)  # Correção na equação da densidade
    D_pa = g / vT_pa**2 * coef / 1.225
    a_pa[i] = -g - D_pa * v_pa[i] * np.abs(v_pa[i])
    v_pa[i + 1] = v_pa[i] + a_pa[i] * dt
    y_pa[i + 1] = y_pa[i] + v_pa[i] * dt

t_total = np.concatenate((t1, t2))
y_total = np.concatenate((y_sl, y_pa))

plt.plot(t_total, y_total, 'b-')
plt.xlabel("Tempo, t (s)")
plt.ylabel("Posição, y (m)")
plt.show()

izero = np.size(y_pa) - np.size(y_pa[y_pa < 0])
tzero = t2[izero] + 20
vzero = v_pa[izero]
print("Tempo de retorno à origem, tzero =", tzero, "s")
print("Velocidade no retorno à origem =", vzero * 3.6, "km/h")
