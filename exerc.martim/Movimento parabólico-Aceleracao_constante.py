import matplotlib.pyplot as plt
import sympy as sy
import numpy as np

a = 3 # aceleração em m/s^2
v0 = 0 # velocidade inicial em m/s
s0 = 0 # posição inicial
vf = 250/3600*1000 #velocidade final em m/s
t = sy.symbols('t')
s = s0 + v0 * t + (1/2) * a * t**2

# Para visualizar a função do movimento do avião
s_func = sy.lambdify(t, s, 'numpy')
t_vals = np.linspace(0, 100, 400)
s_vals = s_func(t_vals)

plt.plot(t_vals, s_vals)
plt.show()

# Determinar o instante em que atinge a velocidade de decolagem
t_decolagem = (vf-v0)/a
print("t = ", t_decolagem, "s") 

# Determinar distância percorrida pelo avião
print("Distância percorrida = ", s_func(t_decolagem), "m")

# Integrar a aceleração para obter a velocidade
v = sy.integrate(a, t) + v0
print("Velocidade = ", v)

# Integrar a velocidade para obter a posição
s2 = sy.integrate(v, t) + s0
print("Posição  = ", s)


#Encontrar tempo e posicao
t2 = sy.nsolve(v - vf, 0)
print("t=", t2)
s2_func = sy.lambdify(t, s2, 'numpy')

print("Distancia percorrida =", s2_func(t2))


