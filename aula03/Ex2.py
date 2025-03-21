import sympy as sy
import numpy as np
import matplotlib.pyplot as plt 

g=9.8
vt=6.8
t = sy.symbols('t')
y = (vt**2 / g) * sy.log(sy.cosh(g * t / vt))

funcao_y = sy.lambdify(t, y, 'numpy')
t_vals = np.linspace(0, 4, 200)
distancia = funcao_y(t_vals)

v = sy.diff(y, t)
funcao_v = sy.lambdify(t, v, 'numpy')
vv = funcao_v(t_vals)

a = sy.diff(v, t)
funcao_a = sy.lambdify(t, a, 'numpy')
aa = funcao_a(t_vals)

#5
altura = 20
tempo_resis = sy.nsolve(y - altura, t,0)
print(f"Tempo com resistência do ar: {tempo_resis}")
tempo_sem_resis = sy.nsolve(altura - (1/2)*g*t**2,t,0)
print(f"Tempo sem resistência do ar: {tempo_sem_resis}")


plt.plot(t_vals, distancia)
plt.xlabel('Tempo (s)')
plt.ylabel('distancia(t)')
plt.title('Lei do Movimento y(t)')
plt.grid(True)
plt.show()

plt.plot(t_vals, vv)
plt.xlabel('Tempo (s)')
plt.ylabel('Velocidade (v(t))')
plt.title('Velocidade Instantânea v(t)')
plt.grid(True)
plt.show()

plt.plot(t_vals, aa)
plt.xlabel('Tempo (s)')
plt.ylabel('Aceleração (v(t))')
plt.title('Aceleração Instantânea v(t)')
plt.grid(True)
plt.show()

ay = g -(g/vt**2) *vv**2

plt.plot(t_vals, ay)
plt.xlabel('Tempo (s)')
plt.ylabel('Aceleração (v(t))')
plt.title('Aceleração Instantânea v(t)')
plt.grid(True)
plt.show()




