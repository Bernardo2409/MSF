# Ex1, mas com dados diferentes e gráfico plot
# x = X, y =L


import numpy as np
import matplotlib.pyplot as plt

def regressao_linear(x, y):
    N = len(x)
    soma_xy = np.sum(np.multiply(x, y))  
    soma_x = np.sum(x)  
    soma_y = np.sum(y)  
    soma_x2 = np.sum(np.multiply(x, x))  
    # soma_y2 = np.sum(np.multiply(y, y))  

    m = (N * soma_xy - soma_x * soma_y) / (N * soma_x2 - soma_x ** 2)
    b = (soma_y - m * soma_x) / N

    y_media = np.mean(y)
    ss_total = np.sum((y - y_media) ** 2)  
    ss_erro = np.sum((y - (m * x + b)) ** 2)  
    r2 = 1 - (ss_erro / ss_total)

    delta_m = np.sqrt(N / (N * soma_x2 - soma_x ** 2))
    delta_b = np.sqrt(soma_x2 / (N * soma_x2 - soma_x ** 2))

    return m, b, r2, delta_m, delta_b

#dados de teste
x = np.array([1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.3])
y = np.array([92.0, 113.0, 133.0, 153.0, 171.5, 194.0, 207.5, 222.0])

m, b, r2, delta_m, delta_b = regressao_linear(x, y)

print(f"m = {m}")
print(f"b = {b}")
print(f"r^2 = {r2}")
print(f"delta m = {delta_m}")
print(f"delta b = {delta_b}")

y_test = 165.0
x_test = (y_test-b) /m

print(f"O valor de X(X) quando y(L) = {y_test} cm é {x_test:.2f}")
erro = np.sqrt((-(y_test - b) / m**2 * delta_m)**2 + (-1 / m * delta_b)**2)
print(f"O Erro encontrado ao calcular X é:{erro}")

plt.figure(figsize=(10, 5))
plt.scatter(x, y, color='blue', label='Dados')
plt.plot(x, m * x + b, color='red', label=f'Regressão Linear com valores da difreação de luz  numa fenda: y = {m:.2f}x + {b:.2f}')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Regressão Linear')
plt.legend()
plt.grid(True)
plt.show()



# L(y) = 165,0 
#y = 98.32 *x -5.17

