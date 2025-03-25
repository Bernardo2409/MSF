import numpy as np
import matplotlib.pyplot as plt

def minimos_quadrados(X, Y, N):
    sumX = np.sum(X)
    sumY = np.sum(Y)
    sumX2 = np.sum(X ** 2)
    sumY2 = np.sum(Y ** 2)
    sumXY = np.sum(X * Y)

    m = (N * sumXY - sumX * sumY) / ((N * sumX2) - ((sumX)**2))

    b = (sumX2 * sumY - sumX * sumXY) / (N * sumX2 - (sumX**2))

    r2 = (N * sumXY - sumX * sumY)**2 / ((N* sumX2 - (sumX)**2)*(N * sumY2 - (sumY)**2))

    dm = abs(m)*(np.sqrt((( 1 / r2 ) - 1)/(N - 2)))

    db = dm*np.sqrt(sumX2 / N)
    return m,b,r2,dm,db

# Tempo [dias]
t_i = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0])

# Atividade radioativa [mCi]
A_i = np.array([9.676 , 6.355, 4.261, 2.729, 1.862, 1.184, 0.7680, 0.4883, 0.3461, 0.2119])

# Representar dados num grafico (usando o matplotlib)
plt.scatter(t_i, A_i)
plt.xlabel("tempo, [dias]")
plt.ylabel("Atividade radioativa, [mCi]")
plt.show()

# Representar dados num grafico (usando o matplotlib)
plt.semilogy(t_i, A_i, 'bo')
plt.title('Log10(A) vs Tempo')
plt.xlabel("Tempo, t [dias]")
plt.ylabel("log10(A)")
plt.show()

# Atenção à diferença entre o logaritmo natural "np.log" e o logarítmo de base 10 "np
X_i = t_i
Y_i = np.log(A_i)

# Calcular melhor reta

m, b, r2, dm, db = minimos_quadrados(X_i, Y_i,N=len(t_i))

# Imprimir resultados
print("m = {0:.4f}".format(m))
print("b = {0:.2f} cm".format(b))
print("r² = {0:.4f}...".format(r2))
print("Δm = {0:.4f}".format(dm))
print("Δb = {0:.2f} cm".format(db))

# Gerar valores do tempo e espaço percorrido para representação da reta
x = np.linspace(0.0, 45.0, 2)
y = m * x + b

# Representar dados num grafico (usando o matplotlib)
plt.scatter(X_i, Y_i)
plt.plot(x, y)
plt.xlabel("tempo, [dias]")
plt.ylabel("ln(A)")
plt.show()

print("tau = -1/m = ", -1 / m, "dias")