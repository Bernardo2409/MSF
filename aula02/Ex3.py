import numpy as np
import matplotlib.pyplot as plt

T = np.array([200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100]) 
E = np.array([0.6950, 4.363, 15.53, 38.74, 75.08, 125.2, 257.9, 344.1, 557.4, 690.7])  

plt.figure(figsize=(10, 5))
plt.plot(T, E, marker='o', linestyle='-', color='r')
plt.xlabel('Temperatura (T/K)')
plt.ylabel('Energia (E/J)')
plt.title('Energia Emitida com a Temperatura')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.semilogy(T, E, marker='o', linestyle='-', color='g')
plt.xlabel('Temperatura (T/K)')
plt.ylabel('Log(E/J)')
plt.title('Gráfico Log-Linear')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.loglog(T, E, marker='o', linestyle='-', color='b')
plt.xlabel('Log(T/K)')
plt.ylabel('Log(E/J)')
plt.title('Gráfico Log-Log')
plt.grid(True)
plt.show()

log_T = np.log(T)
log_E = np.log(E)

N = len(log_T)
soma_log_T = np.sum(log_T)
soma_log_E = np.sum(log_E)
soma_log_T_log_E = np.sum(log_T * log_E)
soma_log_T2 = np.sum(log_T ** 2)

declive = (N * soma_log_T_log_E - soma_log_T * soma_log_E) / (N * soma_log_T2 - soma_log_T ** 2)
b = (soma_log_E - declive * soma_log_T) / N

plt.figure(figsize=(10, 5))
plt.plot(log_T, declive * log_T + b, label=f'Ajuste Linear: E = {np.exp(b):.3f} * T^{declive:.3f}')
plt.scatter(log_T, log_E, color='r', marker='o', label='Dados')
plt.xlabel('Log(T/K)')
plt.ylabel('Log(Energia/J)')
plt.title('Ajuste Linear no gráfico Log-Log')
plt.legend()
plt.grid(True)
plt.show()

ss_total = np.sum((log_E - np.mean(log_E)) ** 2)
ss_residual = np.sum((log_E - (declive * log_T + b)) ** 2)
r2 = 1 - (ss_residual / ss_total)
print(f'r^2 = {r2:.3f}')

a = np.exp(b)
m = declive
print(f'A função de ajuste é: E = {a:.3f} * T^{m:.3f}')