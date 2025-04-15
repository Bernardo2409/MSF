import numpy as np
import matplotlib.pyplot as plt

m = 2000  
theta = np.radians(5)  
P_subida = 40000  
v0 = 1.0  
x0 = 0.0  
u = 0.04
C_res = 0.25
A = 2.0  
rho_ar = 1.225  
g = 9.81  

P_x = -m * g * np.sin(theta)
F_rol = -u * m * g * np.cos(theta)


def acceleration(v):
    if v == 0:
        F_motor = 0  
    else:
        F_motor = P_subida / v
    F_res = -0.5 * C_res * A * rho_ar * abs(v) * v
    F_total = F_motor + P_x + F_rol + F_res

    #Fr = m * a
    a = F_total / m
    return a

dt = 0.1  
t_max = 500  
steps = int(t_max / dt)

time = np.zeros(steps)
x = np.zeros(steps)
v = np.zeros(steps)
a = np.zeros(steps)

time[0] = 0
x[0] = x0
v[0] = v0

for i in range(1, steps):
    time[i] = time[i-1] + dt
    a[i-1] = acceleration(v[i-1])
    v[i] = v[i-1] + a[i-1] * dt
    x[i] = x[i-1] + v[i-1] * dt
    
    if x[i] >= 2000:
        break

time = time[:i+1]
x = x[:i+1]
v = v[:i+1]
a = a[:i]

t_2km = time[-1]
print(f"Tempo para percorrer 2 km: {t_2km:.2f} s")

W_motor = P_subida * t_2km
print(f"Trabalho realizado pelo motor: {W_motor:.2f} J")

plt.figure(figsize=(12, 8))

# Posição-Tempo
plt.subplot(2, 2, 1)
plt.plot(time, x, label='Posição (m)')
plt.xlabel('Tempo (s)')
plt.ylabel('Posição (m)')
plt.title('Posicao/Tempo')
plt.grid()
plt.legend()

# Velocidade-Tempo
plt.subplot(2, 2, 2)
plt.plot(time, v, label='Velocidade (m/s)', color='orange')
plt.xlabel('Tempo (s)')
plt.ylabel('Velocidade (m/s)')
plt.title('Velocidade/Tempo')
plt.grid()
plt.legend()

# Aceleração-Tempo
plt.subplot(2, 2, 3)
plt.plot(time[:-1], a, label='Aceleração (m/s²)', color='green')
plt.xlabel('Tempo (s)')
plt.ylabel('Aceleração (m/s²)')
plt.title('Aceleracao/Tempo')
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()