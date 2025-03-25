import numpy as np
import matplotlib.pyplot as plt
import sympy as sp


g = 9.81  
vT = 6.80  

#1
t_vals = np.linspace(0, 4, 100)

y_vals = (vT**2 / g) * np.log(np.cosh(g * t_vals / vT))

plt.plot(t_vals,y_vals,'r')
plt.xlabel('Tempo (s)')
plt.ylabel('Altura (m)')
plt.title("Posição Instântanea em função do tempo")
plt.show()

#2
t, g, vT = sp.symbols('t g vT')

y_t = (vT**2 / g) * sp.log(sp.cosh(g * t / vT))
v_t = sp.diff(y_t, t)

v_t = v_t.subs({vT: 6.80, g: 9.81}) # substituir

v_t_func = sp.lambdify(t, v_t, 'numpy') #Expressão simbolica para númerica
v_vals = v_t_func(t_vals)


plt.plot(t_vals,v_vals,'r')
plt.xlabel('Tempo (s)')
plt.ylabel('Velocidade (m7s)')
plt.title("Velocidade Instântanea em função do tempo")
plt.show()

#3
a = sp.symbols('a')
a_t = sp.diff(v_t,t)
a_t = a_t.subs({vT: 6.80, g: 9.81})
a_t_func = sp.lambdify(t, a_t, 'numpy') 
a_vals = a_t_func(t_vals)

plt.plot(t_vals,a_vals,'r')
plt.xlabel('Tempo (s)')
plt.ylabel('Aceleração (m/s**2)')
plt.title("Aceleração Instântanea em função do tempo")
plt.show()

#4
'''
ay_T = g - (g/vT**2)*v_t * np.abs(v_t)
ay_T_func = sp.lambdify(t, ay_T, 'numpy') 
ay_T_vals = ay_T_func(t_vals)

plt.plot(t_vals,ay_T_vals,'r')
plt.xlabel('Tempo (s)')
plt.ylabel('Altura (m)')
plt.title("Aceleração Instantanea em função do tempo")
plt.show()
'''
#5
y0 = 20
g = 9.81  
vT = 6.80  

def y_t(t):
    return y0 - (vT**2 / g) * np.log(np.cosh(g * t / vT))

t_vals = np.linspace(0, 4, 400)

y_vals = y_t(t_vals)

plt.plot(t_vals, y_vals, label='y(t)')
plt.show()

y_vals = (vT**2 / g) * np.log(np.cosh(g * t_vals / vT))
t20= sp.solve(20-y_vals).subs({vT: 6.80, g: 9.81})
print(t20)