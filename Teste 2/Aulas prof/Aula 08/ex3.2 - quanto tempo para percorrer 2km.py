import numpy as np
import matplotlib.pyplot as plt

t0 = 0.0 # condição inicial, tempo [s]
tf = 400 # limite do domínio, tempo final [s]
dt = 0.01 # passo [s]

N = int((tf - t0) / dt + 1) # Numero de passos
x0 = 0.0 # condição inicial, posição inicial [m]
v0 = 1.0 # condição inicial, velocidade inicial [m/s]
g = 9.8 # aceleração gravítica [m/s^2]
u = 0.04 # coeficiente de resistêcia ao rolamento
C = 0.25 # coeficiente de resistência do ar
A = 2.0 # Secção (área) efetiva do carro [m^2]
M = 2000.0 # Massa do carro [kg]
theta = 5 * np.pi / 180.0 # Inclinacao (radianos)
rho = 1.225 # Densidade do ar [kg/m^3]
Pcar = 40.0 * 1000 # Potência produzida pelo carro [W]

# inicializar domínio [s]
t = np.linspace(t0, tf, num=N)

# inicializar solução, aceleração [m/s^2]
a = np.zeros(np.size(t)) # aceleração resultante
a_grv = np.zeros(np.size(t)) # aceleração devido à gravidade
a_rol = np.zeros(np.size(t)) # aceleração ao rolamento
a_rar = np.zeros(np.size(t)) # aceleração à resistencia do ar
a_car = np.zeros(np.size(t)) # aceleração devido à propulsão do motor do carro

# inicializar solução, velocidade [m/s]
v = np.zeros(np.size(t))
v[0] = v0

# inicializar solução, posição [m]
x = np.zeros(np.size(t))
x[0] = x0
for i in range(np.size(t) - 1):
    # calcular aceleração
    a_grv[i] = - g * np.sin(theta)
    a_rol[i] = - u * g * np.cos(theta)
    a_rar[i] = - C * A * rho * np.abs(v[i]) * v[i] / (2 * M)
    a_car[i] = Pcar / (M * v[i])
    a[i] = a_grv[i] + a_rol[i] + a_rar[i] + a_car[i]
    v[i + 1] = v[i] + a[i] * dt
    x[i + 1] = x[i] + v[i] * dt

#solução
i = np.size(x[x <= 2000.0])
t2000 = t[i]
print("O carro demora {0:.2f} segundos a percorrer 2 km.".format(t2000))