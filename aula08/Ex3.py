import numpy as np
import matplotlib.pyplot as plt

m = 2000  
theta = np.radians(5)  
u = 0.04
C_res = 0.25
A = 2.0  
rho_ar = 1.225  
g = 9.81  


P_x = -m * g * np.sin(theta)
F_rol = -u * m * g * np.cos(theta)

def acceleration(v, P_motor):
    if v == 0:
        F_motor = 0  
    else:
        F_motor = P_motor / v
    F_res = -0.5 * C_res * A * rho_ar * abs(v) * v
    F_total = F_motor + P_x + F_rol + F_res
    return F_total / m

def simulate(P_motor, v0, x0, direction_label):
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
        a[i-1] = acceleration(v[i-1], P_motor)
        v[i] = v[i-1] + a[i-1] * dt
        x[i] = x[i-1] + v[i-1] * dt

        if x[i] >= 2000:
            break

    time = time[:i+1]
    x = x[:i+1]
    v = v[:i+1]
    a = a[:i]

    t_2km = time[-1]
    W_motor = P_motor * t_2km

    print(f"{direction_label}")
    print(f"Tempo para percorrer 2 km: {t_2km:.2f} s")
    print(f"Trabalho realizado pelo motor: {W_motor:.2f} J\n")

    return time, x, v, a

# Subida
P_subida = 40000  
time_up, x_up, v_up, a_up = simulate(P_subida, v0=1.0, x0=0.0, direction_label="Subida")

# Descida
P_descida = -30000  
time_down, x_down, v_down, a_down = simulate(P_descida, v0=20.0, x0=0.0, direction_label="Descida")
x_down = -x_down
# Gráficos
plt.figure(figsize=(12, 10))

# Subida - posição
plt.subplot(3, 2, 1)
plt.plot(time_up, x_up)
plt.title('Subida - Posição vs Tempo')
plt.xlabel('Tempo (s)')
plt.ylabel('Posição (m)')
plt.grid()

# Subida - velocidade
plt.subplot(3, 2, 2)
plt.plot(time_up, v_up, color='orange')
plt.title('Subida - Velocidade vs Tempo')
plt.xlabel('Tempo (s)')
plt.ylabel('Velocidade (m/s)')
plt.grid()

# Subida - aceleração
plt.subplot(3, 2, 3)
plt.plot(time_up[:-1], a_up, color='green')
plt.title('Subida - Aceleração vs Tempo')
plt.xlabel('Tempo (s)')
plt.ylabel('Aceleração (m/s²)')
plt.grid()

# Descida - posição
plt.subplot(3, 2, 4)
plt.plot(time_down, x_down)
plt.title('Descida - Posição vs Tempo')
plt.xlabel('Tempo (s)')
plt.ylabel('Posição (m)')

plt.grid()




plt.tight_layout()
plt.show()
