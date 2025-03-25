import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

vt= 6.8 #m/s
g= 9.8
t = sp.symbols('t')
y_t = vt**2/g * sp.ln(sp.cosh(g*t/vt))
t_vals = np.linspace(0,4,100)

#a
y_t_simplified = sp.simplify(y_t)
y_t_func = sp.lambdify(t, y_t_simplified, modules='numpy')
y_vals = y_t_func(t_vals)
plt.plot(t_vals,y_vals,label="Lei do movimento",color="red")
plt.legend()
plt.xlabel("t (s)")
plt.ylabel("y(t) (m)")
plt.show()

#b
v_t = sp.diff(y_t, t)
v_t_simplified = sp.simplify(v_t)
v_t_func = sp.lambdify(t, v_t_simplified, modules='numpy')
v_vals = v_t_func(t_vals)
plt.plot(t_vals,v_vals,label="Velocidade Instantânea",color="red")
plt.legend()
plt.xlabel("t (s)")
plt.ylabel("y(t) (m/s)")
plt.show()
#v_t = vt * tanh (g*t/vt)

#c
a_t = sp.diff(v_t,t)
a_t_simplified = sp.simplify(a_t)
a_t_func = sp.lambdify(t, a_t_simplified, modules='numpy')
a_vals = a_t_func(t_vals)
plt.plot(t_vals,a_vals,label="Aceleração Instantânea",color="red")
plt.legend()
plt.xlabel("t (s)")
plt.ylabel("a(t) (m/s**2)")
plt.show()
#a_t = g/(cosh(g*t/vt)**2)

#d
a_t = g -(g/vt**2)*sp.Abs(v_t)*v_t
a_t_simplified = sp.simplify(a_t)
a_t_func = sp.lambdify(t, a_t_simplified, modules='numpy')
a_vals = a_t_func(t_vals)
plt.plot(t_vals,a_vals,label="Aceleração Instantânea",color="red")
plt.legend()
plt.xlabel("t (s)")
plt.ylabel("a(t) (m/s**2)")
plt.show()
#É igual!
plt.ioff()

#e
h0 = 20
eq_cr = (6.80 / 9.81) * sp.ln(sp.cosh((9.81 / 6.80) * t)) - h0
t_sol = sp.nsolve(eq_cr,t,3) #n solve
print("Com resistência do ar, o volante demora {:.2f} segundos a cair de uma altura de 20 metros".format(t_sol))


eq_sr = h0 - 1/2*g*t**2
t_solutions_sr = sp.solve(eq_sr,t)
t_sol_sr = [sol for sol in t_solutions_sr if sol.is_real and sol > 0][0]  # Seleciona a raiz positiva
print("Sem resistência do ar, o volante demora {:.2f} segundos a cair de uma altura de 20 metros".format(t_sol_sr))

#f 
#Com resistência
t_queda = t_sol
v_final  = vt * sp.tanh(g*t_queda/vt)
a_final = g/(sp.cosh(g*t_queda/vt)**2)

print("Velocidade em que bate no solo: {:.2f}".format(v_final))
print("Aceleração em que bate no solo: {:.2f}".format(a_final))
