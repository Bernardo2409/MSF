import numpy as np
import matplotlib.pyplot as plt

X = np.array([2.3, 2.2, 2.0, 1.8, 1.8, 1.4, 1.2, 1.0]) 
Y = np.array([222.0,207.5,194.0,171.5,153.0,133.0,113.0,92.0])
N= len(X)

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

m,b,r2,dm,db = ex1(Y,X,N)
print("m = {:.4f}".format(m))
print("b = {:.4f}".format(b))
print("r2 = {:.4f}".format(r2))
print("dm = {:.4f}".format(dm))
print("db = {:.4f}".format(db))
#3
plt.scatter(Y, X)
#4
x = np.array([np.min(Y), np.max(Y)])
y = m * x + b
plt.plot(x, y)
plt.xlabel("Distância da fenda ao alvo, L [cm]")
plt.ylabel("Distância entre máximos, X [cm]")
plt.show()

#5
print("X(L=165.0 cm) = {0:.2f} cm".format(m * 165.0 + b))
