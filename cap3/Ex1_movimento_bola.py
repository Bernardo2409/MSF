import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

v0 = 100*1000/3600  #km/h para m/s
theta = np.deg2rad(10)
g = 9.8
vT = 100*1000/3600
D = g/vT**2

v0x = v0 * np.cos(theta)
v0y = v0 * np.sin(theta)

#Lei do moviemento sem resistencia do ar
t_max = 2*v0y/g
t_vals = np.linspace(0, t_max, 100)
x_vals = v0x * t_vals
y_vals = v0y * t_vals - 0.5 * g * t_vals**2

plt.figure(figsize=(10, 5))
plt.plot(x_vals, y_vals, label="Trajetória sem resistência do ar")
plt.xlabel("Distância horizontal (m)")
plt.ylabel("Altura (m)")
plt.title("Trajetória da Bola de Futebol")
plt.grid()
plt.legend()
plt.show()

#Altura maxima sem resistencia do ar

t_max = v0y/g
y_max = v0y*t_max - 0.5*g*t_max**2
print("b) Altura máxima sem resistência do ar: {:.2f} m".format(y_max))
print("Instante: {:.2f} s".format(t_max))

#Alcance sem Resistencia do ar

t_total = 2*v0y/g
x_total = v0x * t_total
print("c) Alcance sem resistência do ar: {:.2f} m".format(x_total))
print("Instante total: {:.2f} s".format(t_total))

#Lei do movimento e lei da velocidade usando método de Euler

def MetodoEuler(x0, v0, t0, tf, dt):
    N = int((tf-t0)/dt+0.1)+1
    x = np.zeros(N)
    y = np.zeros(N)
    vx = np.zeros(N)
    vy = np.zeros(N)
    t = np.zeros(N)

    x[0]=x0[0]
    y[0]=x0[1]
    vx[0]=v0[0]
    vy[0]=v0[1]
    t[0]=t0

    for i in range(N-1):
        ax = 0
        ay = -g

        x[i+1] = x[i] + vx[i]*dt
        y[i+1] = y[i] + vy[i]*dt

        vx[i+1] = vx[i] + ax*dt
        vy[i+1] = vy[i] + ay*dt
        t[i+1] = t[i] + dt

    x = x[:i+2]
    y = y[:i+2]
    vx = vx[:i+2]
    vy = vy[:i+2]
    t = t[:i+2]

    return t, x, y, vx, vy


#Parâmetros iniciais
x0 = (0,0)
v0 = (v0x, v0y)
t0 = 0
tf = 2*v0y/g
dt = 0.01

t, x, y, vx, vy = MetodoEuler(x0, v0, t0, tf, dt)

plt.figure(figsize=(10,5))
plt.plot(x, y, label="Método de Euler (sem resistência)")
plt.xlabel("Distância horizontal (m)")
plt.ylabel("Altura (m)")
plt.title("Trajetória da Bola de Futebol (Método de Euler)")
plt.grid()
plt.legend()
plt.show()

#D - Método de Euler com resistência do ar

def MetodoEulerComResistencia(x0, v0, t0, tf, dt):
    N = int((tf-t0)/dt+0.1)+1
    x = np.zeros(N)
    y = np.zeros(N)
    vx = np.zeros(N)
    vy = np.zeros(N)
    t = np.zeros(N)

    x[0]=x0[0]
    y[0]=x0[1]
    vx[0]=v0[0]
    vy[0]=v0[1]
    t[0]=t0

    for i in range(N-1):
        v = np.sqrt(vx[i]**2 + vy[i]**2)
        ax = -D * vx[i] * v
        ay = -g - D * vy[i] * v

        x[i+1] = x[i] + vx[i]*dt
        y[i+1] = y[i] + vy[i]*dt

        vx[i+1] = vx[i] + ax*dt
        vy[i+1] = vy[i] + ay*dt

        t[i+1] = t[i] + dt

    x = x[:i+2]
    y = y[:i+2]
    vx = vx[:i+2]
    vy = vy[:i+2]
    t = t[:i+2]

    return t, x, y, vx, vy



t_drag, x_drag, y_drag, vx_drag, vy_drag = MetodoEulerComResistencia(x0, v0, t0, tf, dt)

plt.figure(figsize=(10, 5))
plt.plot(x, y, 'b-', label='Sem resistência do ar')
plt.plot(x_drag, y_drag, 'g-', label='Com resistência do ar')
plt.xlabel('Distância horizontal (m)')
plt.ylabel('Altura (m)')
plt.title('Trajetória da Bola - Efeito da Resistência do Ar')
plt.grid()
plt.legend()
plt.show()

#F - Altura máxima com resistência do ar

y_max_drag = np.max(y_drag)
print("Altura máxima com resistência do ar: {:.2f} m".format(y_max_drag))
print("Instante: {:.2f} s".format(t_drag[np.argmax(y_drag)]))

#G - Alcance e tempo com resistência do ar

alcance_drag = x_drag[-1]
tempo_drag = t_drag[-1]
print("Alcance com resistência do ar: {:.2f} m".format(alcance_drag))
print("Tempo total com resistência do ar: {:.2f} s".format(tempo_drag))

