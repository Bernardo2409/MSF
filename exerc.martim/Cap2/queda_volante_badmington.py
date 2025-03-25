import numpy as np
import matplotlib.pyplot as plt
import sympy as sy

vT = 6.80
g = 9.80

t_sym = sy.symbols('t')

y_sym = (vT**2 / g) * sy.log(sy.cosh(g * t_sym / vT))
v_sym = sy.diff(y_sym, t_sym)
a_sym = sy.diff(v_sym, t_sym)

print("Expressão simbólica da velocidade v(t):")
print(v_sym)
print("Expressão simbólica da aceleração a(t):")
print(a_sym)

y_func = sy.lambdify(t_sym, y_sym, 'numpy')
v_func = sy.lambdify(t_sym, v_sym, 'numpy')
a_func = sy.lambdify(t_sym, a_sym, 'numpy')

t_num = np.linspace(0, 4, 100)

y_num = y_func(t_num)
v_num = v_func(t_num)
a_num = a_func(t_num)

plt.plot(t_num, y_num, label="v(t)")
plt.xlabel("t (s)")
plt.ylabel("y (m)")
plt.title("Posição em função do tempo")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.show()

plt.plot(t_num, v_num, label="v(t)")
plt.xlabel("t (s)")
plt.ylabel("v (m/s)")
plt.title("Velocidade em função do tempo")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.show()

plt.plot(t_num, a_num, label="a(t)")
plt.xlabel("t (s)")
plt.ylabel("a (m/s²)")
plt.title("Aceleração em função do tempo")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.show()


# d)
a_alter_sym = g - (g / vT**2) * v_sym**2
a_alter_simplified = sy.simplify(a_alter_sym)
print("Aceleracao simplificada", a_alter_simplified)

# e)

#Com resistencia do ar
h0 = 20
y_sym2 = h0 - (vT**2/g)*sy.log(sy.cosh(g * t_sym / vT))
t_solo = sy.nsolve(y_sym2, t_sym, 2)  


print("Tempo para atingir o solo com resistencia do ar: ", t_solo)


#Sem resistencia do ar

t_solo_sem_resistecia = sy.solve(h0 - (1/2)*g*t_sym**2, t_sym)
print("Tempo para atingir o solo sem resistencia do ar: ", t_solo_sem_resistecia)

# f)
v_sym2 = sy.diff(y_sym2, t_sym)
a_sym2 = sy.diff(v_sym2, t_sym)

v_sym2_func = sy.lambdify(t_sym,v_sym2, 'numpy')
a_sym2_func = sy.lambdify(t_sym, a_sym2, 'numpy')

t_solo = float(t_solo)

print("Velocidade ao chegar a solo: ", v_sym2_func(t_solo))
print("Aceleração ao chegar ao solo", a_sym2_func(t_solo))