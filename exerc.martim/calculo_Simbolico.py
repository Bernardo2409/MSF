import sympy as sy
import numpy as np
import matplotlib.pyplot as plt

x,y,m,b = sy.symbols('x,y,m,b')
y=m*x**2+b
y2 = y.subs([(m,0.01), (b,0.0)])

y_em_1 = y2.evalf(subs={x:1})
y_lam = sy.lambdify(x,y2,"numpy")
y_lam(x)

fig, ax = plt.subplots()

x_vals = np.linspace(0, 2, 100)
y_vals = y_lam(x_vals)

ax.plot(x_vals, y_vals)
plt.show()

sy.diff(y,x)
sy.integrate(y,x)

