import numpy as np
import matplotlib.pyplot as plt

N = np.logspace(start=1,stop=4,num=4,dtype=int)
print("N = ",N)

Xmedia = np.zeros(4)
Xerro = np.zeros(4)
i=0

for N_i in N:
    X = np.random.normal(4.500, 0.500, size = N_i)
    Xmedia[i] = np.mean(X)
    Xerro[i] = np.std(X)/np.sqrt(N_i)
    print("N_i = {:.3f}, Xmedia = {:.3f}, Xerro= {:.3f}".format(N_i,Xmedia[i],Xerro[i]))
    i+=1

plt.semilogx(N, Xmedia, 'b-o')
plt.errorbar(N, Xmedia, yerr=Xerro, fmt='g-d')
plt.xlabel("Número de medições, log(N)")
plt.ylabel("Valor médio")
plt.show()