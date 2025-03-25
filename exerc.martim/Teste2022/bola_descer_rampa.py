import numpy as np
import matplotlib.pyplot as plt


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


t = np.array([0.5,1.5,2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5])
s = np.array([0.1,1.4,1.7,6.5,7.7,10.4,19.5, 26.1, 26.5, 45.9, 52.5])

m,b,r2,dm,db = minimos_quadrados(t,s)


plt.scatter(t,s)
plt.plot(t,m*t+b)
plt.xlabel('Tempo (s)')
plt.ylabel('Posição (m)')
plt.title('Posição x Tempo')
plt.show()

print('m =',m,'+-',dm)
print('b =',b,'+-',db)
print('r2 =',r2)

#b)

t_log = np.log(t)
s_log = np.log(s)

m,b,r2,dm,db = minimos_quadrados(t_log,s_log)
plt.plot(t,np.exp(m*t_log+b))

plt.loglog(t,s, 'bo')
plt.xlabel('Tempo (s)')
plt.ylabel('Posição (m)')
plt.title('Posição x Tempo')
plt.show()

print('m =',m,'+-',dm)
print('b =',b,'+-',db)
print('r2 =',r2)

#c)
t2 = t**2.011
m,b,r2,dm,db = minimos_quadrados(t2,s)
plt.plot(t2,m*t2+b)
plt.scatter(t2,s)
plt.xlabel('Tempo (s)')
plt.ylabel('Posição (m)')
plt.title('Posição x Tempo')
plt.show()

print('m =',m,'+-',dm)
print('b =',b,'+-',db)
print('r2 =',r2)

