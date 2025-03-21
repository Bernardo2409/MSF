import numpy as np
import sympy as sy
import matplotlib.pyplot as plt

vt = 6.8  # Velocidade terminal (m/s)
g = 9.8  # Aceleração da gravidade (m/s²)
y0 = 20  # Altura inicial (m)

t_sym = sy.Symbol('t')

# Definição da posição
y_sym = y0 - (vt**2 / g) * sy.log(sy.cosh(3 * t_sym))
v_sym = sy.diff(y_sym, t_sym)
a_sym = sy.diff(v_sym, t_sym)

y_func = sy.lambdify(t_sym, y_sym, 'numpy')
v_func = sy.lambdify(t_sym, v_sym, 'numpy')
a_func = sy.lambdify(t_sym, a_sym, 'numpy')

t = np.linspace(0, 4, 500)

y = y_func(t)
v = v_func(t)
a = a_func(t)

ay = g - (g / vt**2) * np.abs(v) * v

plt.figure(figsize=(10, 5))

plt.subplot(2, 2, 1)
plt.plot(t, y)
plt.title('Movimento')
plt.xlabel('Tempo (s)')
plt.ylabel('Posição (m)')
plt.grid()

plt.subplot(2, 2, 2)
plt.plot(t, v)
plt.title('Velocidade')
plt.xlabel('Tempo (s)')
plt.ylabel('Velocidade (m/s)')
plt.grid()

plt.subplot(2, 2, 3)
plt.plot(t, a)
plt.title('Aceleração')
plt.xlabel('Tempo(s)')
plt.ylabel('Aceleracao (m/s**2)')
plt.grid()

plt.subplot(2, 2, 4)
plt.plot(t, ay)
plt.title('Aceleração em y')
plt.xlabel('Tempo(s)')
plt.ylabel('Aceleracao (m/s**2)')
plt.grid()

plt.tight_layout()
plt.show()
