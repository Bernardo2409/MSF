import numpy as np
import matplotlib.pyplot as plt

# 1) Faça o gráfico da lei do movimento y(t) de 0 a 4s

t = np.linspace(0.0, 4.0, 100) # definir dominio e resolução temporal [s]
v_T = 6.80 # velocidade terminal do volante [m/s]
g = 9.80 # aceleração gravitacional (valor standard) [m/s^2]

# posição do volante [m]
y = (v_T ** 2 / g) * np.log(np.cosh(g * t / v_T))

# Representar dados num grafico (usando o matplotlib)
plt.plot(t, y)
plt.xlabel("Tempo decorrido, t [s]")
plt.ylabel("Posição, y [m]")
plt.show()


# 2) Determine a velocidade instantânea em função do tempo, usando cálculo simbólico com o
# sympy . Faça o gráfico da velocidade em função do tempo de 0 a 4 s, usando o pacote matplotlib

# Definir symbolos
import sympy as sy
y, t, v_T, g = sy.symbols('y, t, v_T, g')
# Definir função posição do volante
y = (v_T ** 2 / g) * sy.log(sy.cosh(g * t / v_T))

# Calcular velocidade instantânea
v = sy.diff(y, t)  #equação da velocidade
print("v(t) = ", v)

# Converter expressão da velocidade em função de um array numpy
# Nota 1: Só a variável "t" é independente. Temos que especificar todos os restantes s
# Nota 2: v_lam é um array numpy se o argumento "t" for um array numpy
v_lam = sy.lambdify(t, v.subs({v_T: 6.80, g: 9.80}), "numpy")   #cria um array, em funcao do tempo, onde há substituição de variaveis
# Representar dados num grafico (usando o matplotlib)
t_array = np.linspace(0.0, 4.0, 100)
plt.plot(t_array, v_lam(t_array))
plt.xlabel("Tempo decorrido, t_array [s]")
plt.ylabel("Velocidade, v_lam [m/s]")
plt.show()

#3) Determine a aceleração instantânea em função do tempo, usando cálculo simbólico com o
# sympy . Faça o gráfico da aceleração em função do tempo de 0 a 4 s, usando o pacote matplotlib

# Calcular velocidade instantânea
a = sy.diff(v, t)
print("a(t) = ", a)

# Converter expressão da aceleração em função de um array numpy
# Nota 1: Só a variável "t" é independente. Temos que especificar todos os restantes s
# Nota 2: a_lam é um array numpy se o argumento "t" for um array numpy
a_lam = sy.lambdify(t, a.subs({v_T:6.80, g: 9.80}), "numpy")
# Representar dados num grafico (usando o matplotlib)
t_array = np.linspace(0.0, 4.0, 100)
plt.plot(t_array, a_lam(t_array))
plt.xlabel("Tempo decorrido, t_array [s]")
plt.ylabel(r"Aceleração, a_lam [m/s$^2$]")
plt.show()


#5) Se o volante for largado de uma altura de 20 m, quanto tempo demora a atingir o solo?
# Compare com o tempo que demoraria se não houvesse resistência do ar.
#sy.solve((20 - y).subs({v_T: 6.80, g: 9.80665}), t)
t20_ar = sy.nsolve((20 - y).subs({v_T: 6.80, g: 9.80665}), t, 0.0)
print("Com resistência do ar, o volante atinge o solo quando t = ", t20_ar, "s")

# Posição do volante no vácuo (sem resistência do ar)
y_vac = 1/2 * g * t**2
t20_vac = sy.nsolve((20 - y_vac).subs(g, 9.80665), t, 0.0)
print("Sem resistência do ar, o volante atinge o solo quando t = ", t20_vac, "s")