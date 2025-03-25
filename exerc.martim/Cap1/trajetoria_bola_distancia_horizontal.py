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

x = np.array([1080, 1044, 1008, 972, 936,900, 864, 828, 792, 756, 720,540,360,180,0])
y = np.array([0, np.mean(np.array([2.25, 3.25, 4.5, 6.5])), np.mean(np.array([5.25, 6.5, 6.5, 8.75])), np.mean(np.array([7.5, 7.75, 8.25, 9.25])), np.mean(np.array([8.75, 9.25, 9.5, 10.5])), np.mean(np.array([12, 12.25, 12.5, 14.75])), np.mean(np.array([13.75, 16.00, 16.0, 16.5])), np.mean(np.array([14.75, 15.25, 15.5, 17.5])), np.mean(np.array([15.5, 16, 16.6, 16.75])), np.mean(np.array([17, 17, 17.5, 19.25])), np.mean(np.array([17.5, 18.5, 18.5, 19.0])), np.mean(np.array([19.5, 20, 20.25, 20.5])), np.mean(np.array([18.5, 18.5, 19, 19])), np.mean(np.array([13, 13, 13, 13])), 0])

plt.scatter(x,y)
coef = np.polyfit(x,y,2)
poly_func = np.poly1d(coef)

x_fit = np.linspace(0,1100,100)
y_fit = poly_func(x_fit)
print("polyfit c2 = {}".format(coef))

plt.plot(x_fit, y_fit)


plt.show()



