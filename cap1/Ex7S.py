import numpy as np
import matplotlib.pyplot as plt

# Dados fornecidos
massa = np.array([0.15, 0.20, 0.16, 0.11, 0.25, 0.32, 0.40, 0.45, 0.50, 0.55])  # kg
periodo = np.array([1.21, 1.40, 1.26, 1.05, 1.60, 1.78, 2.00, 2.11, 2.22, 2.33])  # s

# Gráfico original T vs x
plt.figure(figsize=(7,5))
plt.scatter(massa, periodo, color='b', label='Dados experimentais')
plt.xlabel("Massa (kg)")
plt.ylabel("Período (s)")
plt.title("Gráfico do Período vs Massa")
plt.grid(True)
plt.legend()
plt.show()

# Gráfico log-log
log_massa = np.log(massa)
log_periodo = np.log(periodo)
plt.figure(figsize=(7,5))
plt.scatter(log_massa, log_periodo, color='r', label='Dados log-log')
plt.xlabel("log(Massa)")
plt.ylabel("log(Período)")
plt.title("Gráfico log-log")
plt.grid(True)
plt.legend()
plt.show()

# Ajuste linear log-log
coef_log = np.polyfit(log_massa, log_periodo, 1)  # Regressão linear
slope_log, intercept_log = coef_log
r2_log = np.corrcoef(log_massa, log_periodo)[0,1]**2

print(f"Declive log-log: {slope_log:.4f}, Ordenada na origem: {intercept_log:.4f}, R²: {r2_log:.4f}")

# Transformação para relação linear (T² vs x)
periodo_quadrado = periodo**2
coef_T2_x = np.polyfit(massa, periodo_quadrado, 1)
slope_T2_x, intercept_T2_x = coef_T2_x
r2_T2_x = np.corrcoef(massa, periodo_quadrado)[0,1]**2

# Gráfico T² vs x
plt.figure(figsize=(7,5))
plt.scatter(massa, periodo_quadrado, color='g', label='Dados transformados')
plt.plot(massa, slope_T2_x * massa + intercept_T2_x, 'r-', label='Ajuste linear')
plt.xlabel("Massa (kg)")
plt.ylabel("Período² (s²)")
plt.title("Gráfico de T² vs Massa")
plt.grid(True)
plt.legend()
plt.show()

# Cálculo da constante elástica k
k = 4 * np.pi**2 / slope_T2_x
print(f"Constante elástica k: {k:.4f} N/m")