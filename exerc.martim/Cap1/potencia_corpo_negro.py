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


A = 100 
T = np.array([200,300,400,500,600,700,800,900,1000,1100])
E = np.array([0.6950, 4.363, 15.53, 38.74, 75.08, 125.2, 257.9, 344.1, 557.4, 690.7])

plt.scatter(T, E)
plt.xlabel("T(K)")
plt.ylabel("E(J)")
plt.show()

#R: Não
#Gráfico loglog



# Transformação logarítmica dos dados
log_T = np.log10(T)
log_E = np.log10(E)

# Ajuste linear nos dados transformados
m, b, r2, dm, db = minimos_quadrados(log_T, log_E)

# Geração da reta ajustada
log_E_fit = m * log_T + b

# Gráfico log-log com a reta ajustada
plt.loglog(T, E, 'bo')
plt.loglog(T, 10**log_E_fit, 'r-', label=f'Ajuste linear (r²={r2:.4f})')
plt.title('Log10(P) vs log10(T)')
plt.xlabel("log10(T)")
plt.ylabel("log10(P)")
print("m = {:.2f} +- {:.2f}, r2 = {:.3f}".format(m, dm, r2))
plt.legend()

plt.show()
