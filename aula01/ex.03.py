import numpy as np
import matplotlib.pyplot as plt
import random

N=10
X = np.random.normal(4.5,0.5,size=N)
Xmedia = np.mean(X)
Xerro = np.std(X)/np.sqrt(N)
print(X)

Y = np.random.normal(10,1, size= N)
Ymedia = np.mean(Y)
Yerro = np.std(Y)/np.sqrt(N)
print(Y)


Z = X + Y
Zmedia = np.mean(Z)
Zerro= np.std(Z) / np.sqrt(N)
Zerro_estimated = np.sqrt(Xerro**2 + Yerro**2)

print(Z)
print(Zmedia)
print(Zerro)
print(Zerro_estimated)
