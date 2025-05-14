import numpy as np
import matplotlib.pyplot as plt
# Capítulo 4 - Exercício 13

# Dados
v0_kmh = 100  # km/h
v0 = v0_kmh * 1000 / 3600  # m/s
theta = np.deg2rad(10)  # rad
m = 0.057  # kg
g = 9.8  # m/s²
dt = 0.001

# Componentes da velocidade
v0x = v0 * np.cos(theta)
v0y = v0 * np.sin(theta)


A = 0.033
Car = 0.5
Rar = 1.225


#Calculo da energia mecanica em qualquer intante, no caso de não considerar a resistencia do ar

E = 0.5 * m * v0**2

print(f"Energia mecânica: {E:.2f} J")

# Considerando a resistencia do ar, calcular a energia mecanica
t = np.arange(0, 10, dt)  
x = np.zeros(np.size(t))  # inicializa o vetor de posicoes x
y = np.zeros(np.size(t))  # inicializa o vetor de posicoes y
vx = np.zeros(np.size(t))  # inicializa o vetor de velocidades x
vy = np.zeros(np.size(t))  # inicializa o vetor de velocidades y
v = np.zeros(np.size(t))   # inicializa o vetor de velocidades totais
E = np.zeros(np.size(t))  # inicializa o vetor de energia mecanica
W = np.zeros(np.size(t))  # inicializa o vetor de trabalho




x[0]= 0.0  # posicao inicial x
y[0] = 0.0 
vx[0] = v0x  # velocidade inicial x
vy[0] = v0y  # velocidade inicial y
E[0]=0.5*m*(v0**2) + m*g*y[0] # energia mecanica inicial
W[0] =0.0

Karr = 0.5* m*Car*A*Rar #coeficiente de arrasto

for i in range(1, np.size(t)):
    v = np.sqrt(vx[i-1]**2 + vy[i-1]**2)
    Farr = -Karr*v**2

    ax = (-Karr * v * vx[i-1]) / m
    ay = (-Karr * v * vy[i-1]) / m - g

    vx[i] = vx[i-1] + ax * dt
    vy[i] = vy[i-1] + ay * dt

    x[i] = x[i-1] + vx[i-1] * dt
    y[i] = y[i-1] + vy[i-1] * dt

    E[i] = 0.5 * m * (vx[i]**2 + vy[i]**2) + m * g * y[i]  # energia mecanica   

t_interp = np.array([0.0, 0.4, 0.8])
E_interp = np.interp(t_interp, t, E)

# Resultados
print("Energia mecânica:")
for ti, Ei in zip(t_interp, E_interp):
    print(f"t = {ti:.1f} s: E = {Ei:.2f} J")


# Gráfico da energia mecânica
plt.figure(figsize=(10, 5))
plt.plot(t, E, 'b-', label="Energia mecânica (com arrasto)")
plt.scatter(t_interp, E_interp, color='red', label="Pontos de interesse")
plt.xlabel("Tempo (s)")
plt.ylabel("Energia (J)")
plt.title("Energia Mecânica da Bola de Tênis (Resistência do Ar)")
plt.grid()
plt.legend()
plt.show()


# Cálculo do trabalho realizado pela força de arrasto

t = np.arange(0, 10, dt)  
x = np.zeros(np.size(t))  # inicializa o vetor de posicoes x
y = np.zeros(np.size(t))  # inicializa o vetor de posicoes y
vx = np.zeros(np.size(t))  # inicializa o vetor de velocidades x
vy = np.zeros(np.size(t))  # inicializa o vetor de velocidades y
E = np.zeros(np.size(t))  # inicializa o vetor de energia mecanica
W = np.zeros(np.size(t))  # inicializa o vetor de trabalho
v = np.zeros(np.size(t))   # inicializa o vetor de velocidades totais


x[0]= 0.0  # posicao inicial x
y[0] = 0.0 
vx[0] = v0x  # velocidade inicial x
vy[0] = v0y  # velocidade inicial y
E[0]=0.5*m*(v0**2) + m*g*y[0] # energia mecanica inicial
W[0] =0.0


vT= 100*1000/3600 # m/s
k = (m * g) / (vT ** 2)  # Coeficiente de arrasto (kg/m)

for i in range(1, np.size(t)):
    # Atualiza velocidades com arrasto
    ax = (-k * v[i-1] * vx[i-1]) / m
    ay = -g + (-k * v[i-1] * vy[i-1]) / m
    vx[i] = vx[i-1] + ax * dt
    vy[i] = vy[i-1] + ay * dt
    v[i] = np.sqrt(vx[i]**2 + vy[i]**2)
    
    # Cálculo do trabalho (regra trapezoidal)
    W[i] = W[i-1] - k * ( (v[i-1]**3 + v[i]**3) / 2 ) * dt


# Valores nos instantes t = 0, 0.4, 0.8 s
t_interp = [0.0, 0.4, 0.8]
W_interp = np.interp(t_interp, t, W)

print("Trabalho realizado pela força de arrasto:")
for ti, Wi in zip(t_interp, W_interp):
    print(f"t = {ti:.1f} s: W = {Wi:.2f} J")