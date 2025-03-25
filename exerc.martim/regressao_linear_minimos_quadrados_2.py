import matplotlib.pyplot as plt
import numpy as np
from Práticas.calcular_regressao_linear import minimos_quadrados

T = np.array([0.0,5.0,10.0,15.0,20.0,25.0,30.0,35.0,40.0,45.0])
A = np.array([9.676, 6.355, 4.261, 2.729, 1.862, 1.184, 0.7680, 0.4883, 0.3461, 0.2119])

plt.scatter(T,A)
plt.xlabel("tempo, (dias)")
plt.ylabel("Atividade radioativa")
plt.show()

plt.semilogy(T,A)
plt.title('Log(A) vs Tempo')
plt.xlabel("Tempo, t (dias)")
plt.ylabel("log(A)")
plt.show()


x = T
y = np.log(A)

m,b,r2,dm,db = minimos_quadrados(x,y)

print("m = {0:.4f}".format(m))
print("b = {0:.2f} cm".format(b))
print("r² = {0:.4f}...".format(r2))
print("dm = {0:.4f}".format(dm))
print("db = {0:.2f} cm".format(db))

y = m * x + b
plt.scatter(x, y)
plt.plot(x, y)
plt.xlabel("tempo, [dias]")
plt.ylabel("ln(A)")
plt.show()

t=-np.log(2)/m
print("t", t)