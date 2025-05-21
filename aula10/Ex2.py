import numpy as np
import matplotlib.pyplot as plt

def maxminv(xm1,xm2,xm3,ym1,ym2,ym3):  # moodle
    # Máximo ou mínimo usando o polinómio de Lagrange
    # Dados (input): (x0,y0), (x1,y1) e (x2,y2) 
    # Resultados (output): xm, ymax 
    xab=xm1-xm2
    xac=xm1-xm3
    xbc=xm2-xm3

    a=ym1/(xab*xac)
    b=-ym2/(xab*xbc)
    c=ym3/(xac*xbc)

    xmla=(b+c)*xm1+(a+c)*xm2+(a+b)*xm3
    xm=0.5*xmla/(a+b+c)

    xta=xm-xm1
    xtb=xm-xm2
    xtc=xm-xm3

    ymax=a*xtb*xtc+b*xta*xtc+c*xta*xtb
    return xm, ymax

L = 1.0    
g = 9.8   
theta0 = 0.1  
v0 = 0.0     
t0 = 0.0     
tf = 10  
dt = 0.001   

t = np.arange(t0, tf, dt)
theta = np.zeros_like(t) 
theta[0] = theta0

# Euler-Cromer
v = v0
for i in range(1, len(t)):
    v = v - (g/L) * np.sin(theta[i-1]) * dt
    theta[i] = theta[i-1] + v * dt

# Máximos locais
max_indices = []
for i in range(1, len(theta)-1):
    if theta[i] > theta[i-1] and theta[i] > theta[i+1]:
        max_indices.append(i)

# Interpolação
max_times = []
max_values = []

for idx in max_indices[:4]:  

    x_vals = [t[idx-1], t[idx], t[idx+1]]
    y_vals = [theta[idx-1], theta[idx], theta[idx+1]]
    
    t_max, theta_max = maxminv(x_vals[0], x_vals[1], x_vals[2], 
                        y_vals[0], y_vals[1], y_vals[2])
    max_times.append(t_max)
    max_values.append(theta_max)

# Medição do Período
if len(max_times) >= 2:
    periods = np.diff(max_times)
    average_period = np.mean(periods)
    theoretical_period = 2*np.pi*np.sqrt(L/g)
    
    print("Tempos dos máximos:", max_times)
    print("Períodos entre máximos consecutivos:", periods)
    print(f"Período médio medido: {average_period:.4f} s")
    print(f"Período teórico: {theoretical_period:.4f} s")
    print(f"Diferença relativa: {abs(average_period-theoretical_period)/theoretical_period*100:.2f}%")
else:
    print("Para medir o período, será necessário encontrar dois máximos seguidos")

plt.figure(figsize=(12, 6))
plt.plot(t, theta, label='Ângulo (t) [rad]')
if max_times:
    plt.scatter(max_times, max_values, c='red', label='Máximos identificados')
plt.title('Movimento do Pêndulo Simples')
plt.xlabel('Tempo (s)')
plt.ylabel('Ângulo (rad)')
plt.grid(True)
plt.legend()
plt.show()