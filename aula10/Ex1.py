import numpy as np
import matplotlib.pyplot as plt

g = 9.8      
L = 1.0      
dt = 0.01    
T = 10  #tempo     
n = int(T / dt)

# Euler-Cromer
def euler_cromer(theta0, omega0=0):
    theta = np.zeros(n)
    omega = np.zeros(n)
    t = np.linspace(0, T, n)
    
    theta[0] = theta0
    omega[0] = omega0
    
    for i in range(1, n):
        omega[i] = omega[i-1] - (g/L) * np.sin(theta[i-1]) * dt
        theta[i] = theta[i-1] + omega[i] * dt
    
    return t, theta

def solucao_analitica(theta0, t):
    A = theta0  # amplitude máxima (aulas teóricas)
    phi = 0  # velocidade inicial = 0
    return A * np.cos(np.sqrt(g/L) * t + phi)


#c)
angulos_iniciais = [0.1, 0.3, 0.5]  

fig, axs = plt.subplots(1, len(angulos_iniciais), figsize=(15, 4))

for idx, theta0 in enumerate(angulos_iniciais):
    t, theta_num = euler_cromer(theta0)
    theta_ana = solucao_analitica(theta0, t)
    
    ax = axs[idx]
    ax.plot(t, theta_num, label='Numérica (Euler-Cromer)')
    ax.plot(t, theta_ana, label='Analítica (linear)', linestyle='--')
    ax.set_title(f'theta = {theta0} rad')
    ax.set_xlabel('Tempo (s)')
    ax.set_ylabel('Ângulo theta (rad)')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()