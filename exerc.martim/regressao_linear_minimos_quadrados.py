import matplotlib.pyplot as plt
import numpy as np

from calcular_regressao_linear import minimos_quadrados

A = 100 #cm2
T = np.array([200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0, 1100.0])
P  = np.array([0.6950, 4.363, 15.53, 38.74, 75.08, 125.2, 257.9, 344.1, 557.4, 690.7])




plt.scatter(T, P)
plt.xlabel("Tempratura, T(K)")
plt.ylabel("Potencia, P(W)")
plt.show()

plt.semilogy(T,P, 'bo')
plt.title('Log10(P) vs Temperatura')
plt.xlabel("Temperatura, T (K)")
plt.ylabel("log10(P)")

plt.show()



plt.loglog(T, P, 'bo')
plt.title('Log10(P) vs log10(T)')
plt.xlabel("log10(T)")
plt.ylabel("log10(P)")

plt.show()

T = np.log(T)
P  = np.log(P)

m,b,r2,dm,db = minimos_quadrados(T,P)

print("r2: ", r2)

x = np.array([5.25, 7.0])
y = m * x + b

plt.scatter(T, P)
plt.plot(x, y)
plt.show()


print(m)
print(np.exp(b))

