import numpy as np
import matplotlib.pyplot as plt

#Minimos quadrados (N = número de itens)
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

L = np.array([222.0,207.5,194.0,171.5,153.0,133.0,113.0,92.0])
X = np.array([2.3,2.2,2.0,1.8,1.6,1.4,1.2,1.0])
N = len(L)

m,b,r2,dm,db = minimos_quadrados(L,X,N)

plt.plot(L,X,'.')

#RETA DE REGRESSÃO MÍNIMOS QUADRADOS!

x = np.array([np.min(L), np.max(L)])
y = m*x+b

plt.plot(x,y,'-')
plt.show()

print("X aproximado quando L = 165.0: {}".format(m*165+b))