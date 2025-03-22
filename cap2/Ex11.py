import numpy as np
import matplotlib.pyplot as plt

# Parâmetros
g = 9.81  # aceleração devido à gravidade em m/s^2
k = 1.44  # constante de resistência ao ar em 1/s
v_terminal = 6.80  # velocidade terminal em m/s
t_max = 20  # tempo máximo para simulação em segundos
dt = 0.1  # passo de tempo em segundos

# Inicializando variáveis
t = np.arange(0, t_max, dt)  # vetor de tempo
v = np.zeros(len(t))  # vetor de velocidades
y = np.zeros(len(t))  # vetor de posições

# Condições iniciais
v[0] = 0  # velocidade inicial
y[0] = 0  # posição inicial

# Método de Euler para a velocidade e posição
for i in range(1, len(t)):
    # Atualização da velocidade
    v[i] = v[i-1] + dt * (g - k * v[i-1])
    # Atualização da posição
    y[i] = y[i-1] + dt * v[i-1]

# Gráfico da velocidade em função do tempo
plt.figure(figsize=(10, 5))
plt.plot(t, v, label='Velocidade (m/s)')
plt.axhline(v_terminal, color='r', linestyle='--', label='Velocidade Terminal')
plt.xlabel('Tempo (s)')
plt.ylabel('Velocidade (m/s)')
plt.title('Velocidade Instantânea em Função do Tempo')
plt.legend()
plt.grid(True)
plt.show()

# Gráfico da posição em função do tempo
plt.figure(figsize=(10, 5))
plt.plot(t, y, label='Posição (m)')
plt.xlabel('Tempo (s)')
plt.ylabel('Posição (m)')
plt.title('Posição em Função do Tempo')
plt.grid(True)
plt.show()
