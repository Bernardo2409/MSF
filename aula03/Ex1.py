import sympy as sy
import numpy as np
import matplotlib.pyplot as plt 
# v0 = 0
#ax = 3 m/s^2
#vf = 250 km/h

vi = 0
vf = 250 * 1000 / 3600
ax = 3
tf = (vf-vi) /ax
tempo = np.linspace(0, tf, 20)
movimento = 0 + 1/2 * ax * (tempo**2)


plt.figure(figsize=(10, 5))
plt.plot(movimento, tempo, 'o-', label='Atividade do avião')
plt.xlabel('Distância/m')
plt.ylabel('Tempo/s')
plt.title('Movimento do avião')
plt.legend()
plt.grid(True)
plt.show()


print(f"O tempo percorrido pelo avião até atingir a velocidade de descolagem é: {tf:.2f} segundos")
distancia = 0 +1/2 * ax *(tf**2)
print(f"A distância até atingir a velocidade de descolagem é: {distancia:.2f} metros")
# x = x0+v0*t + 1/2 at^2
# vf = v0 + at  

#descobrir o t, depois substituir na lei do movimento

t = sy.symbols('t')
v = sy.integrate(ax,t)
x = sy.integrate(v,t)

tempo_final = sy.nsolve(v - vf, t, 0)
distancia_final = sy.nsolve(x, t,0)

print(tempo_final)
print(x)
print(distancia_final)
