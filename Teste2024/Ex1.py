import numpy as np
import matplotlib.pyplot as plt

t = np.array([0,1,2,3,4,5,6,7])
n = np.array([11,20,33,54,83,134,244,425])


plt.figure(figsize = (10, 5))


def minimos_quadrados(x, y):
    # Número de pontos
    N = x.size
    
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_x2 = np.sum(x ** 2)
    sum_y2 = np.sum(y ** 2)
    sum_xy = np.sum(x * y)

    # Coeficientes da reta
    m = (N * sum_xy - sum_x * sum_y) / (N * sum_x2 - sum_x ** 2)
    b = (sum_x2 * sum_y - sum_x * sum_xy) / (N * sum_x2 - sum_x ** 2)

    # Coeficiente de determinação
    r2 = ((N * sum_xy - sum_x * sum_y) ** 2) / ((N * sum_x2 - sum_x ** 2) * (N * sum_y2 - sum_y ** 2))

    # Incertezas
    dm = np.abs(m) * np.sqrt((1 / r2 - 1) / (N - 2))
    db = dm * np.sqrt(sum_x2 / N)

    return m, b, r2, dm, db     
m,b,r2,dm,db = minimos_quadrados(t,n)

print('r2 =',r2)

plt.scatter(t,n)
plt.plot(t,m*t+b,'r-',label='Ajuste')
plt.legend()
plt.xlabel('Tempo (h)')
plt.ylabel('Número de bactérias')
plt.show()


plt.semilogy(t, n, 'bo')  
y = np.log(n) 

m, b, r2, dm, db = minimos_quadrados(t, y)


print('m =',m)
print('b =',b)
print('r2 =',r2)
print('dm =',dm)
print('db =',db)

plt.plot(t, np.exp(m * t + b), 'r-', label="Ajuste")
plt.legend()
plt.xlabel("tempo (dias)")
plt.ylabel("Atividade (mCi)")
plt.show()