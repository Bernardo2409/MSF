import numpy as np
import matplotlib.pyplot as plt

# Definição de parâmetros
v0 = 50  # m/s
g = 9.81  # m/s²
t_max = 5  # Tempo até a explosão (s)

# Vetor de tempo
t = np.linspace(0, t_max, 100)

# Equação da altura
y = v0 * t - 0.5 * g * t**2

# Criar o gráfico
plt.figure(figsize=(8, 5))
plt.plot(t, y, label="Altura $y(t)$", color="b")
plt.xlabel("Tempo (s)")
plt.ylabel("Altura (m)")
plt.title("Altura do fogo de artifício em função do tempo")
plt.axhline(0, color='black', linewidth=0.8)
plt.axvline(5, color='red', linestyle="--", label="Explosão (t=5s)")
plt.legend()
plt.grid()

# Mostrar o gráfico
plt.show()

# Calcular o tempo em que a altura máxima ocorre
t_max_altura = v0 / g

# Calcular a altura máxima
y_max = v0 * t_max_altura - 0.5 * g * t_max_altura**2

# Calcular a altura no momento da explosão (t = 5s)
t_explosao = 5
y_explosao = v0 * t_explosao - 0.5 * g * t_explosao**2

# Diferença de altura
diferenca_altura = y_max - y_explosao

# Exibir os resultados
print(f"Tempo até a altura máxima: {t_max_altura:.2f} s")
print(f"Altura máxima: {y_max:.2f} m")
print(f"Altura na explosão: {y_explosao:.2f} m")
print(f"Diferença de altura: {diferenca_altura:.2f} m")
