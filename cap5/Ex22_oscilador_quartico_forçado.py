import numpy as np
import matplotlib.pyplot as plt

alpha = 0.25
b = 0.05
F0 = 7.5
omega_f = 1.0
m = 1.0

x0_1 = 3.000
x0_2 = 3.001
v0 = 0.0

t_max = 50.0
dt = 0.01
num_steps = int(t_max / dt)
t = np.linspace(0, t_max, num_steps)

def aceleracao(t, x, vx):
    return -4 * alpha * x - b * vx + F0 * np.cos(omega_f * t) / m


def rk4_x_vx(t,x,vx,acelera,dt):
    """
    Integração numérica de equação diferencial de 2ª ordem:
			d2x/dt2 = ax(t,x,vx)    com dx/dt= vx    de valor inicial
	Erro global:  proporcional a dt**4
    acelera=dvx/dt=Força(t,x,vx)/massa      : acelera é uma FUNÇÃO
    input:  t = instante de tempo
            x(t) = posição
            vx(t) = velocidade
            dt = passo temporal 
    output: xp = x(t+dt)
		    vxp = vx(t+dt)
    """
    ax1=acelera(t,x,vx)
    c1v=ax1*dt
    c1x=vx*dt
    ax2=acelera(t+dt/2.,x+c1x/2.,vx+c1v/2.)
    c2v=ax2*dt
    c2x=(vx+c1v/2.)*dt			# predicto:  vx(t+dt) * dt
    ax3=acelera(t+dt/2.,x+c2x/2.,vx+c2v/2.)
    c3v=ax3*dt
    c3x=(vx+c2v/2.)*dt
    ax4=acelera(t+dt,x+c3x,vx+c3v)
    c4v=ax4*dt
    c4x=(vx+c3v)*dt
      
    xp=x+(c1x+2.*c2x+2.*c3x+c4x)/6.
    vxp=vx+(c1v+2.*c2v+2.*c3v+c4v)/6.
    return xp,vxp

x0_1 = 3.000
x0_2 = 3.001
v0 = 0.0

t_max = 50.0
dt = 0.01
num_steps = int(t_max / dt)
t = np.linspace(0, t_max, num_steps)

x1 = np.zeros(num_steps)
x2 = np.zeros(num_steps)
v1 = np.zeros(num_steps)
v2 = np.zeros(num_steps)

x1[0] = x0_1
v1[0] = v0
x2[0] = x0_2
v2[0] = v0

for i in range(1, num_steps):
    x1[i], v1[i] = rk4_x_vx(t[i-1], x1[i-1], v1[i-1], aceleracao, dt)
    x2[i], v2[i] = rk4_x_vx(t[i-1], x2[i-1], v2[i-1], aceleracao, dt)

difference = np.abs(x1 - x2)
tolerance = 0.001

divergence_indices = np.where(difference > tolerance)[0]

if len(divergence_indices) > 0:
    print(f"Divergência detectada nos índices: {divergence_indices}")

else:
    print("Nenhuma divergência detectada.")

plt.figure(figsize=(12, 6))

plt.subplot(1,2,1)
plt.plot(t, x1, label='x0 = 3.000 m', color='blue')
plt.plot(t, x2, label='x0 = 3.001 m', color='orange')
plt.title('Posição x(t)')
plt.xlabel('Tempo (s)')
plt.ylabel('Posição (m)')
plt.legend()

plt.subplot(1,2,2)
plt.plot(t, v1, label='v0 = 0.0 m/s', color='blue')
plt.axhline(y=tolerance, color='red', linestyle='--', label='Tolerância')
plt.xlabel('Tempo (s)')
plt.ylabel('Velocidade (m/s)')
plt.title('Velocidade v(t)')

plt.tight_layout()
plt.show()
