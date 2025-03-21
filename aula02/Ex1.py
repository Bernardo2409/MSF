import numpy as np

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
    ss_residual = np.sum((y - (m * x + b)) ** 2)  
    r2 = 1 - (ss_residual / ss_total)

    delta_m = np.sqrt(N / (N * soma_x2 - soma_x ** 2))
    delta_b = np.sqrt(soma_x2 / (N * soma_x2 - soma_x ** 2))

    return m, b, r2, delta_m, delta_b

#dados de teste
x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 3, 4, 5])

m, b, r2, delta_m, delta_b = regressao_linear(x, y)

print(f"m = {m}")
print(f"b = {b}")
print(f"r^2 = {r2}")
print(f"delta m = {delta_m}")
print(f"delta b = {delta_b}")
