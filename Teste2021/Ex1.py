import numpy as np
import matplotlib.pyplot as plt

M = np.array([0.15, 0.20, 0.16, 0.11, 0.25, 0.32, 0.40, 0.45, 0.50, 0.55])
T = np.array([1.21, 1.40, 1.26, 1.05, 1.60, 1.78, 2, 2.11, 2.22, 2.33])
def minimos_quadrados(X, Y, N):
    sumX = np.sum(X)
    sumY = np.sum(Y)
    sumX2 = np.sum(X ** 2)
    sumY2 = np.sum(Y ** 2)
    sumXY = np.sum(X * Y)

    m = (N * sumXY - sumX * sumY) / ((N * sumX2) - ((sumX)**2)) #declive

    b = (sumX2 * sumY - sumX * sumXY) / (N * sumX2 - (sumX**2)) #ordenada na origem

    r2 = (N * sumXY - sumX * sumY)**2 / ((N* sumX2 - (sumX)**2)*(N * sumY2 - (sumY)**2)) #coeficiente de correlação linear

    dm = abs(m)*(np.sqrt((( 1 / r2 ) - 1)/(N - 2))) 

    db = dm*np.sqrt(sumX2 / N)
    return m,b,r2,dm,db
N = len(T)
m,b,r2,dm,db = minimos_quadrados(T,M,N)



T = T **2


print(r2)


log_M = np.log(M)
log_T = np.log(T)

tempo = np.linspace(np.min(T), np.max(T) , 400)
declive, ordenadaNaOrigem = np.polyfit(T,M,1)

distancia_ajustada = declive*tempo + ordenadaNaOrigem

plt.plot(tempo, distancia_ajustada, label='Reta Ajustada', color='red')

plt.plot(T,M)
plt.scatter(T,M)
plt.show()


m,b = np.polyfit(log_M,log_T,1)

plt.scatter(log_M,log_T)
plt.plot(log_M,log_M*m+b, label=f'y = {m:.3f}x + {b:.2f}',color='red')
plt.show()

k = 4 * np.pi **2 / declive
print(k)