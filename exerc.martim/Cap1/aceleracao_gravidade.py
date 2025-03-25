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
m=1
R = np.array([6.37, 7.02, 7.61, 8.02, 8.43, 8.92, 9.31, 9.78, 10.25, 10.74])
A = np.array([9.8, 8.0, 6.6, 6.3, 5.5, 5.1, 4.6, 4.1, 3.8, 3.6])


plt.scatter(1/R**2,A)
plt.xlabel("1/R**2(1/10**12 m**2)")
plt.ylabel("a (m/s**2)")
m, b, r2, dm, db = minimos_quadrados(1/R**2, A)



plt.show()
print("K = ({:.0f}+{:.0f}) m3/s2;" .format(m, dm))
#x = 1/r2
#y = a