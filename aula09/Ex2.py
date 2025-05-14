import numpy as np
import matplotlib.pyplot as plt

#1 - 4 esferas , 1 levantada inicialmente

#Tempo
dt = 0.001
tf = 5.0 #s
t0 = 0.0 #s
t = np.arange(t0,tf,dt)
Nt = np.size(t) #número de passos de tempo

#Parâmetros
N = 4 # número de esferas
d = 0.1
l = 10 * d
m = 0.3 #kg
k = 10**7 #N/m^2
g = 9.8 #m/s^2
x0 = np.arange(0, N, 1) * d # posição inicial das esferas (m)

# Grandezas fisicas
x_arr = np.zeros((N, Nt)) # matriz de posições
v_arr = np.zeros((N, Nt)) # matriz de velocidades
a_arr = np.zeros((N, Nt)) # matriz de acelerações

#Condições iniciais
x_arr[:,0] = x0
x_arr[0,0] = -5 * d
v_arr[:,0] = np.zeros(N) # velocidade inicial

def acc_toque(dx,d):
    # calcular a aceleração de uma esfera devido ao contacto com a esfera à sua direita
    k = 1e7
    q = 2.0
    if dx<d:
        a = (-k*abs(dx-d)**q)/m
    else:
        a = 0.0
    return a

def acc_i(i,x):
    #calcular a aceleração de esfera i, cuja posicao de equilibrio é d*i
    a = 0

    if i>0: # a primeira esfera não tem vizinho à sua esquerda
        a -= acc_toque(x[i] - x[i-1],d)
    if i < (N-1): # a última esfera não tem vizinho à sua direita
        a += acc_toque(x[i+1] - x[i], d)

    # aceleração de gravidade, afeta todas as esferas
    a -= g*(x[i]-d*i)/l
    return a

#Metodo Euler-Cromer
for i in range(np.size(t)-1):
    for j in range(0, N):
        a_arr[j,i] = acc_i(j,x_arr[:,i])
        
    v_arr[:,i+1] = v_arr[:,i] + a_arr[:,i] * dt
    x_arr[:,i+1] = x_arr[:,i] + v_arr[:,i+1] * dt

#Representação gráfica
for i in range(0, N):
    plt.plot(t, x_arr[i,:], label='Esfera ' + str(i+1))


plt.legend()
plt.show()

#2 - 5 esferas, 2 levantadas inicialmente
#Tempo
dt = 0.001
tf = 5.0 #s
t0 = 0.0 #s
t = np.arange(t0,tf,dt)
Nt = np.size(t) #número de passos de tempo

#Parâmetros
N = 5 # número de esferas
d = 0.1
l = 10 * d
m = 0.3 #kg
k = 10**7 #N/m^2
g = 9.8 #m/s^2
x0 = np.arange(0, N, 1) * d # posição inicial das esferas (m)

# Grandezas fisicas
x_arr = np.zeros((N, Nt)) # matriz de posições
v_arr = np.zeros((N, Nt)) # matriz de velocidades
a_arr = np.zeros((N, Nt)) # matriz de acelerações
    
#Condições iniciais
x_arr[:,0] = x0
x_arr[0,0] = -5 * d
x_arr[4,0] = 5 * d
v_arr[:,0] = np.zeros(N) # velocidade inicial

def acc_toque(dx,d):
    # calcular a aceleração de uma esfera devido ao contacto com a esfera à sua direita
    k = 1e7
    q = 2.0
    if dx<d:
        a = (-k*abs(dx-d)**q)/m
    else:
        a = 0.0
    return a

def acc_i(i,x):
    #calcular a aceleração de esfera i, cuja posicao de equilibrio é d*i
    a = 0

    if i>0: # a primeira esfera não tem vizinho à sua esquerda
        a -= acc_toque(x[i] - x[i-1],d)
    if i < (N-1): # a última esfera não tem vizinho à sua direita
        a += acc_toque(x[i+1] - x[i], d)

    # aceleração de gravidade, afeta todas as esferas
    a -= g*(x[i]-d*i)/l
    return a

#Metodo Euler-Cromer
for i in range(np.size(t)-1):
    for j in range(0, N):
        a_arr[j,i] = acc_i(j,x_arr[:,i])
        
    v_arr[:,i+1] = v_arr[:,i] + a_arr[:,i] * dt
    x_arr[:,i+1] = x_arr[:,i] + v_arr[:,i+1] * dt

#Representação gráfica
for i in range(0, N):
    plt.plot(t, x_arr[i,:], label='Esfera ' + str(i+1))


plt.legend()
plt.show()

#3 - A minha maneira de fazer
from matplotlib.animation import FuncAnimation

# Parâmetros da simulação
dt = 0.001  # passo de tempo (s)
tf = 5.0    # tempo final (s)
t0 = 0.0    # tempo inicial (s)
t = np.arange(t0, tf, dt)
Nt = np.size(t)  # número de passos de tempo

# Parâmetros físicos
N = 5       # número de esferas
d = 0.1     # diâmetro das esferas (m)
l = 10 * d  # comprimento da corda (m)
m = 0.3     # massa de cada esfera (kg)
k = 1e7     # constante de rigidez (N/m²)
g = 9.8     # aceleração gravítica (m/s²)

# Arrays para armazenar os resultados
x_arr = np.zeros((N, Nt))  # posições
v_arr = np.zeros((N, Nt))  # velocidades
a_arr = np.zeros((N, Nt))  # acelerações

# Condições iniciais
x_arr[:, 0] = np.arange(0, N) * d  # posições de equilíbrio
x_arr[0, 0] = -5 * d  # primeira esfera deslocada para a esquerda
x_arr[4, 0] = 5 * d   # última esfera deslocada para a direita
v_arr[:, 0] = np.zeros(N)  # todas começam com velocidade zero

def acc_contact(dx, d):
    """Calcula a aceleração devido ao contato entre esferas"""
    if dx < d:
        return (-k * abs(dx - d)**2) / m
    return 0.0

def acc_i(i, x):
    """Calcula a aceleração total da esfera i"""
    a = 0.0
    
    # Força de contato com vizinhos
    if i > 0:  # vizinho à esquerda
        a -= acc_contact(x[i] - x[i-1], d)
    if i < N-1:  # vizinho à direita
        a += acc_contact(x[i+1] - x[i], d)
    
    # Força gravítica (aproximação de pêndulo)
    a -= g * (x[i] - d*i) / l
    
    return a

# Simulação com Euler-Cromer
for n in range(Nt-1):
    # Calcular acelerações
    for i in range(N):
        a_arr[i, n] = acc_i(i, x_arr[:, n])
    
    # Atualizar velocidades
    v_arr[:, n+1] = v_arr[:, n] + a_arr[:, n] * dt
    
    # Atualizar posições
    x_arr[:, n+1] = x_arr[:, n] + v_arr[:, n+1] * dt

# Configuração da animação
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(-1, (N-1)*d + 1)
ax.set_ylim(-6*d, 6*d)
ax.set_xlabel('Índice da Esfera', fontsize=12)
ax.set_ylabel('Deslocamento (m)', fontsize=12)
ax.set_title('Simulação do Movimento de 5 Esferas Acopladas', fontsize=14)
ax.grid(True, linestyle='--', alpha=0.5)

# Elementos da animação
balls = [ax.plot([], [], 'o', markersize=20, label=f'Esfera {i+1}')[0] for i in range(N)]
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

def init():
    """Inicialização da animação"""
    for ball in balls:
        ball.set_data([], [])
    time_text.set_text('')
    return balls + [time_text]

def update(frame):
    """Atualização do frame da animação"""
    for i, ball in enumerate(balls):
        ball.set_data(i*d, x_arr[i, frame])
    time_text.set_text(f'Tempo: {t[frame]:.2f} s')
    return balls + [time_text]

# Criar a animação
ani = FuncAnimation(fig, update, frames=range(0, Nt, 10),  # passo de 10 para otimização
                    init_func=init, blit=True, interval=20)

plt.legend(loc='upper right')
plt.tight_layout()
plt.show()