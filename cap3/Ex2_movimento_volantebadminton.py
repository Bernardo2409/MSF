import numpy as np
import matplotlib.pyplot as plt


#Parametros do problema

v0 = 200*1000 / 3600  
theta = np.deg2rad(10)
y0 = 3.0
g = 9.8
vT = 6.8

D = g / vT ** 2 #coeficiente de arrasto

v0x = v0 * np.cos(theta) #componente x da velocidade inicial
v0y = v0 * np.sin(theta) #componente y da velocidade inicial

t_estimado = (v0y + np.sqrt(v0y**2 + 2*g*y0)) / g #tempo estimado de queda
t = np.linspace(0, t_estimado, 1000) 

x = np.zeros(np.size(t)) #inicializa o vetor de posicoes x
y = np.zeros(np.size(t)) #inicializa o vetor de posicoes y
vx = np.zeros(np.size(t)) #inicializa o vetor de velocidades x
vy = np.zeros(np.size(t)) #inicializa o vetor de velocidades y

x[0] = 0.0 #posicao inicial x
y[0] = y0 #posicao inicial y
vx[0] = v0x #velocidade inicial x
vy[0] = v0y #velocidade inicial y

dt = t[1] - t[0] #passo de tempo

for i in range(1, np.size(t)):

    v = np.sqrt(vx[i-1]**2 + vy[i-1]**2) #modulo da velocidade

    ax = -D * vx[i-1]*v
    ay = -g - D * vy[i-1]*v

    vx[i] = vx[i-1] + ax * dt
    vy[i] = vy[i-1] + ay * dt 
    x[i] = x[i-1] + vx[i-1] * dt 
    y[i] = y[i-1] + vy[i-1] * dt 

    if y[i] < 0.0: #se a bola atingir o chao, para o loop
        t = t[:i+1]
        x = x[:i+1]
        y = y[:i+1]
        break

# Plota o grafico da trajetoria

plt.figure(figsize=(10, 5))
plt.plot(x, y, 'r--', label = "Trajetória com resistência do ar")
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Trajetória da bola de badminton')
plt.grid()
plt.legend()
plt.show()


alcance = x[-1] #alcance horizontal
print("b) Alcance horizontal = ", alcance, "m")
print("Tempo de queda = ", t[-1], "s")
