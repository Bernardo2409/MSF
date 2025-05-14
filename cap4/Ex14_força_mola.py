import numpy as np
import matplotlib.pyplot as plt



k = 1
m = 1
x0 = 4
v0x = 0
dt = 0.01

t = np.arange(0, 5, dt) 


E = np.zeros(np.size(t))

#Calculo da energia total, nas condicoes iniciais

E0 = 0.5*m*v0x**2 + 0.5*k*x0**2
E[0] = E0

#Sendo que 0.5*k*x0**2 corresponde a energia potencial e 0.5*m*v0x**2 corresponde a energia potencial elástica

def metodo_euler():
    x = np.zeros(np.size(t)) 
    v = np.zeros(np.size(t))
    x[0] = x0 
    v[0] = v0x
    for i in range(1, np.size(t)):
        a = -k/m * x[i-1]
        v[i] = v[i-1] + a * dt
        x[i] = x[i-1] + v[i-1] * dt
    return x, v


def euler_cromer():
    x = np.zeros(np.size(t)) 
    v = np.zeros(np.size(t))
    x[0] = x0 
    v[0] = v0x
    for i in range(1, np.size(t)):
        a = -k/m * x[i-1]
        v[i] = v[i-1] + a * dt
        x[i] = x[i-1] + v[i] * dt


    return x, v

x_euler, v_euler = metodo_euler()
x_euler_cromer, v_euler_cromer = euler_cromer()

E_euler = 0.5*m*v_euler**2 + 0.5*k*x_euler**2
E_euler_cromer = 0.5*m*v_euler_cromer**2 + 0.5*k*x_euler_cromer**2

# Plot dos gráficos de energia
plt.figure(figsize=(12, 5))

# Gráfico 1: Método de Euler
plt.subplot(1, 2, 1)
plt.plot(t, E_euler, 'b-', label="Método de Euler")
plt.axhline(y=8, color='r', linestyle='--', label="Energia teórica (8 J)")
plt.xlabel("Tempo (s)")
plt.ylabel("Energia total (J)")
plt.title("Energia total, Método de Euler, dt = 0.01")
plt.legend()
plt.grid()

# Gráfico 2: Método de Euler-Cromer
plt.subplot(1, 2, 2)
plt.plot(t, E_euler_cromer, 'g-', label="Método de Euler-Cromer")
plt.axhline(y=8, color='r', linestyle='--', label="Energia teórica (8 J)")
plt.xlabel("Tempo (s)")
plt.ylabel("Energia total (J)")
plt.title("Energia total, Método de Euler-Cromer, dt = 0.01")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()