
import numpy as np
import matplotlib.pyplot as plt

r0 = np.array([0, 2, 3])
v0 = np.array([160, 20, -20]) * 1000/3600

t0 = 0
tf = 0.4
dt = 0.001

m = 0.057
g = 9.80
vt = 120 * 1000 / 3600
D = g / (vt**2)

n = int((tf - t0) / dt + 0.1)
t = np.zeros(n+1)

vx = np.zeros(n+1)
vy = np.zeros(n+1)
vz = np.zeros(n+1)

rx = np.zeros(n+1)
ry = np.zeros(n+1)
rz = np.zeros(n+1)

rx[0] = r0[0]
ry[0] = r0[1]
rz[0] = r0[2]

vx[0] = v0[0]
vy[0] = v0[1]
vz[0] = v0[2]

for i in range(n):
    v = np.sqrt(vx[i]**2 + vy[i]**2 + vz[i]**2)
    
    ax = - m *D * v * vx[i]
    ay = - m * D * v * vy[i]
    az = -m *g -m *D * v * vz[i]
    
    vx[i+1] = vx[i] + ax * dt
    vy[i+1] = vy[i] + ay * dt
    vz[i+1] = vz[i] + az * dt
    
    rx[i+1] = rx[i] + vx[i] * dt
    ry[i+1] = ry[i] + vy[i] * dt
    rz[i+1] = rz[i] + vz[i] * dt
    
    t[i+1] = t[i] + dt

plt.figure(figsize=(10,10))
ax = plt.axes(projection='3d')
ax.plot3D(rx[rz >= 0], ry[rz >= 0], rz[rz >= 0], 'g')

rede_x = [11.9, 11.9, 11.9, 11.9]
rede_y = [0, 0, 8.2, 8.2]
rede_z = [0, 0.9, 0.9, 0]
ax.plot3D(rede_x, rede_y, rede_z)

campo1_x = [0, 11.9 *2 , 11.9 , 0, 0]
campo1_y = [0, 0, 8.2, 8.2, 0]
campo1_z = [0, 0, 0, 0, 0]


ax.plot3D(campo1_x, campo1_y, campo1_z, 'g')

linha_x = [0, 11.9 *2]
linha_y = [4.1, 4.1]
linha_z = [0, 0]
ax.plot3D(linha_x, linha_y, linha_z, 'g')

# Marcadores
ax.scatter(rx[0], ry[0], rz[0], c='g', s=100, label='Inicio')
ax.scatter(rx[-1], ry[-1], rz[-1], c='r', s=100, label='Fim')



ax.set_xlim3d(0, 25)
ax.set_ylim3d(0, 10)
ax.set_zlim3d(0, 5)
ax.set_box_aspect((14, 10, 3))
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

m = 0.057
Em = np.zeros(n+1)
Ep = np.zeros(n+1)
Ec = np.zeros(n+1)

for i in range(n+1):
    h = rz[i]
    Ep[i] = m * g * h
    v = np.sqrt(vx[i]**2 + vy[i]**2 + vz[i]**2)
    Ec[i] = 0.5 * m * v**2
    Em[i] = Ec[i] + Ep[i]


variation = Em[-1] - Em[0]
print(f"A variação(tf-t0) de energia mecânica é {variation:.2f} Joules")

plt.plot(t, Ep, label="Energia Potencial")
plt.plot(t, Ec, label="Energia Cinética")
plt.plot(t, Em, label="Energia Mecânica")
plt.legend(loc="center right")
plt.title("Energias da Bola")
plt.xlabel("Tempo (s)")
plt.ylabel("Energia (J)")
plt.show()

def integral(f, time_array, a, b, dt):
    # dt = (time_array[b] - time_array[0]) / len(f)

    #descobrir os indices
    i_a = int(a / dt) 
    i_b = int(b / dt)
    integral_sum = 0.0
    
    for i in range(i_a, i_b):
        integral_sum += (f[i] + f[i+1]) / 2.0 * dt
    
    return integral_sum
t0 = 0
t1 = 0.2
t2 = 0.4
time_array = np.linspace(t0, tf, n+1)

F_ar = np.zeros((n+1, 3))
for i in range(n+1):
    v = np.sqrt(vx[i]**2 + vy[i]**2 + vz[i]**2)
    F_ar[i, 0] = -D * v * vx[i]  
    F_ar[i, 1] = -D * v * vy[i]   
    F_ar[i, 2] = -D * v * vz[i]   

forca_ar= np.sum(F_ar * np.column_stack((vx, vy, vz)), axis=1)

W_air0 = integral(forca_ar, time_array, t0, t0, dt)
W_air1 = integral(forca_ar, time_array, t0, t1, dt)
W_air2 = integral(forca_ar, time_array, t0, t2, dt)

print(f"O trabalho realizado pela resistência do ar em t0 = {W_air0:.2f} Joules")
print(f"O trabalho realizado pela resistência do ar de t0 a t1 = {W_air1:.2f} Joules")
print(f"O trabalho realizado pela resistência do ar de t0 a t2 = {W_air2:.2f} Joules")


Ec0 = Ec[0]  
Ep0 = Ep[0]  

#indice em t1
i_t1 = int(t1 / dt)

Ec1 = Ec[i_t1] 
Ep1 = Ep[i_t1]  

W_res_t0_t1 = (Ec1 + Ep1) - (Ec0 + Ep0)

print(f"O trabalho realizado pela força de resistência do ar usando a conservação de energia = {W_res_t0_t1:.2f} Joules")  