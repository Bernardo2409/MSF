import numpy as np
import matplotlib.pyplot as plt

x = np.array( [0.15, 0.20, 0.16, 0.11, 0.25, 0.32, 0.40, 0.45, 0.50, 0.55] )
y = np.array( [1.21, 1.40, 1.26, 1.05, 1.60, 1.78, 2 , 2.11, 2.22, 2.33] )

plt.figure(figsize=(13, 5))
plt.subplot(1, 2, 1)

plt.scatter(x,y, color = "blue", label = "Data Points")
plt.xlabel("Massa/Kg")
plt.ylabel("Tempo/s")
plt.grid(True)
plt.title('Primeiro')
plt.legend()




log_x = np.log(x)
log_y = np.log(y)

m,b = np.polyfit(log_x, log_y, 1)

plt.subplot(1, 2, 2)
plt.scatter(log_x, log_y, color="blue", label="Data Points")
plt.plot(log_x, m * log_x + b, color='red', label=f'y = {m:.3f}x + {b:.2f}')
plt.xlabel("log(Massa/Kg)")
plt.ylabel("log(Tempo/s)")
plt.grid(True)
plt.title("LOGS")
plt.legend()


plt.tight_layout()
plt.show()












# Cálculo do coeficiente de determinação (r²)
y_pred = m * log_x + b
ss_res = np.sum((log_y - y_pred) ** 2)
ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
r2 = 1 - (ss_res / ss_tot)

# Impressão dos resultados
print(f"Declive (m): {m:.3f}")
print(f"Ordenada na origem (b): {b:.2f}")
print(f"Coeficiente de determinação (r²): {r2:.5f}")

# Cálculo dos erros nos parâmetros
N = len(log_x)
soma_log_x = np.sum(log_x)
soma_log_xx = np.sum(log_x ** 2)
delta = N * soma_log_xx - soma_log_x ** 2

erro_m = np.sqrt(ss_res / (N - 2) / delta * N)
erro_b = np.sqrt(ss_res / (N - 2) * soma_log_xx / delta)

print(f"Erro no declive (Δm): {erro_m:.5f}")
print(f"Erro na ordenada na origem (Δb): {erro_b:.5f}")

print(f"Solução fornecida: m = 9.87 ± 0.08 s^2/kg; b = 0.02 ± 0.03 s^2; r² = 0.9995")





# Transformação: T^2 vs x
T2 = y**2

# Ajuste linear
coef = np.polyfit(x, T2, 1)  # coef[0] é o declive
slope_T2_x = coef[0]

# Cálculo da constante elástica k
k = 4 * np.pi**2 / slope_T2_x

print(f"Constante elástica k: {k:.4f} N/m")
