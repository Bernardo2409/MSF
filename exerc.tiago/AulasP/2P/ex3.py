import numpy as np
import matplotlib.pyplot as plt

T = np.array([200.0,300.0,400.0,500.0,600.0,700.0,800.0,900.0,1000.0,1100.0])
E = np.array([0.6950,4.363,15.53,38.74,75.08,125.2,257.9,344.1,557.4,690.7])
N = len(T)

def ex1(X, Y, N):
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
# Gráfico Log-Linear (semilogy)
plt.subplot(1, 2, 1)  # 1 linha, 2 colunas, primeiro gráfico
plt.semilogy(T, E, 'o-', label='Dados experimentais')
plt.xlabel('Temperatura (K)')
plt.ylabel('Energia (J) (escala log)')
plt.title('Gráfico Log-Linear')
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()

# Gráfico Log-Log
plt.subplot(1, 2, 2)  # 1 linha, 2 colunas, segundo gráfico
plt.loglog(T, E, 'o-', label='Dados experimentais')
plt.xlabel('Temperatura (K) (escala log)')
plt.ylabel('Energia (J) (escala log)')
plt.title('Gráfico Log-Log')
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()

# Mostrar os gráficos
plt.tight_layout()
plt.show()

#Linearização (importante)!!

X = np.log(T)
Y = np.log(E)


m,b,r2,dm,db = ex1(X,Y,N)
print("m = {:.4f}".format(m))
print("b = {:.4f}".format(b))
print("r2 = {:.4f}".format(r2))
print("dm = {:.4f}".format(dm))
print("db = {:.4f}".format(db))

#defenir 2 pontos para representar melhor a reta
x = np.array([5.25, 7.0])
y = m * x + b

# Representar dados num grafico (usando o matplotlib)
plt.scatter(X, Y)
plt.plot(x, y)
plt.xlabel("X = ln(T)")
plt.ylabel("Y = ln(P)")
plt.show()

print("n = m = ", m)
print("c = exp(b) = ", np.exp(b), "W/K^4")