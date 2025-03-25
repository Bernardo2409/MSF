import numpy as np
import matplotlib.pyplot as plt
N=10
X=np.random.normal(4.5,0.5,size=N)
Y=np.random.normal(10.5,0.5,size=N)
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

m,b,r2,dm,db = ex1(Y,X,N)
print("m = {:.4f}".format(m))
print("b = {:.4f}".format(b))
print("r2 = {:.4f}".format(r2))
print("dm = {:.4f}".format(dm))
print("db = {:.4f}".format(db))