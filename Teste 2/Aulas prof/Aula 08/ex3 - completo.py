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

fig, ax1 = plt.subplots()
color = 'tab:blue'
ax1.set_xlabel('time (s)')
ax1.set_ylabel('Posição, x (m)', color=color)
ax1.plot(t[x<3000], x[x<3000], color=color)
ax1.tick_params(axis='x', labelcolor='k')
ax2 = ax1.twinx() # partilhar eixo horizontal
color = 'tab:red'
ax2.set_ylabel('Velocidade, v (m/s)', color=color)
ax2.plot(t[x<3000], v[x<3000], color=color)
ax2.tick_params(axis='y', labelcolor='k')
fig.tight_layout()
plt.show()

#Quanto tempo leva a percorrer 2km?
i = np.size(x[x <= 2000.0])
t2000 = t[i]
print("O carro demora {0:.2f} segundos a percorrer 2 km.".format(t2000))

#Calcule o trabalho feito pelo motor do carro durante este viagem.
i = np.size(x[x <= 2000.0])
t2000 = t[i]
Wcar_sub = Pcar * t2000 # trabalho em J = W.s = kW.h / (1000*3600)
print("O motor do carro consome Wcar = {0:.2f} kWh durante a subida.".format(Wcar_sub))

#Calcule o tempo para percorrer 2km, e o trabalho feito pelo motor na descida.
Wcar_des = Pcar * t2000 # trabalho em J = W.s = kW.h / (1000*3600)
print("O motor do carro consome Wcar = {0:.2f} kWh durante a descida.".format(Wcar_des))

#Se 50% do trabalho na descida é recuperado para carregar a bateria do carro,
#qual a diferência de energia na bateria no final, depois de ter feito a subida e a descida,
# comparado com no início? (Assume 100% eficiencia do motor na subida.)
E_bat = Wcar_sub + Wcar_des * 0.5 # Energia em J = W.s = kW.h / (1000*3600)
print("A energia gasta pelas baterias é: ")
print("E_bat = {0:.2f} kWh durante toda a viagem.".format(E_bat/(1000*3600)))