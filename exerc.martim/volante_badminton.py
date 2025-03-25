import matplotlib.pyplot as plt
import sympy as sy
import numpy as np

g=10
t = sy.symbols('t')
v=6.8
y=v**2/g*sy.log(sy.cosh(g*t/v))


y_func = sy.lambdify(t,y,'numpy')

t_vals = np.linspace(0, 4, 100)
y_vals = y_func(t_vals)

plt.plot(t_vals, y_vals)
plt.show()

#Determinar a velocidade instantanea em funcao do tempo

v = sy.integrate(g, t) + 0

print(v)
v_func = sy.lambdify(t,v,'numpy')
v_vals=v_func(t_vals)
plt.plot(t_vals, v_vals)
plt.show()

#Determinar aceleracao em funcao do tempo

a = sy.diff(v, t)

print(a)

a2 = g - g/6.8**2*v*abs(v)
a2_lam = sy.lambdify(t, a2, "numpy")
a2_vals = a2_lam(t_vals)
plt.plot(t_vals, a2_vals)
plt.show()
print(a2)

# Determinar o tempo para atingir o solo 
altura_inicial = 20
tempo = sy.solve(y - altura_inicial, t)
print(tempo)