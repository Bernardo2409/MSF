import matplotlib.pyplot as plt
import numpy as np
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



l = np.array([222.0,207.5,194.0, 171.5, 153.0, 133.0, 113.0, 92.0])
x = np.array([2.3, 2.2, 2.0, 1.8, 1.6, 1.4, 1.2, 1.0])


# a) 
plt.scatter(l,x)
plt.xlabel("Temperatura, T(K)")
plt.ylabel("Potencia, P(W)")
plt.show()


# c)
m, b, r2, dm, db = minimos_quadrados(l, x)
print("m = {0:.4f}".format(m))
print("b = {0:.2f} cm".format(b))
print("r² = {0:.4f}...".format(r2))
print("dm = {0:.4f}".format(dm))
print("db = {0:.2f} cm".format(db))

# d)
y = m * l + b  # Ajuste para usar 'l' como eixo x para a reta
plt.scatter(l, x, label="Pontos experimentais")
plt.plot(l, y, color="red", label="Reta ajustada")
plt.xlabel("Temperatura, T(K)")
plt.ylabel("Potência, P(W)")
plt.legend()
plt.show()

# e)
l2 = 165.0
x2 = m * l2 + b
print("Valor de X quando l2 = 165.0cm: {:.2f} cm".format(x2))

# f)

x2 = np.array([2.8, 2.2, 2.0, 1.8, 1.6, 1.4, 1.2, 1.0])

m2, b2, r22, dm2, db2 = minimos_quadrados(l,x2)
print("Novo coeficiente de determinação (r²) com o valor alterado:")
print("r² original = {0:.4f}".format(r2))
print("r² modificado = {0:.4f}".format(r22))

y_new = m2 * l + b2  # Nova reta ajustada
plt.scatter(l, x2, label="Novos pontos experimentais", color="blue")
plt.plot(l, y_new, color="green", label="Nova reta ajustada")
plt.xlabel("Temperatura, T(K)")
plt.ylabel("Potência, P(W)")
plt.legend()
plt.title("Gráfico com pontos modificados e nova reta ajustada")
plt.show()