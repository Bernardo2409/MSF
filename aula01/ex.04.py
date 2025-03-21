import numpy as np
import matplotlib.pyplot as plt

# Parâmetros da distribuição
media_esperada = 5  # Média esperada das medições
desvio_padrao = 2    # Desvio padrão das medições

# Valores de N de 10 até 10000 de forma logarítmica
Ns = np.logspace(1, 4, num=4, dtype=int)  # 50 pontos entre 10 e 10000

# Armazenamento das médias calculadas
medias = []
incerteza_teorica = []

for N in Ns:
    dados = np.random.normal(media_esperada, desvio_padrao, N)  # Geração das medições
    media_calculada = np.mean(dados)  # Média dos valores gerados
    medias.append(media_calculada)
    incerteza_teorica.append(desvio_padrao / np.sqrt(N))  # Incerteza esperada

# Converter para arrays
medias = np.array(medias)
incerteza_teorica = np.array(incerteza_teorica)

# Gráfico
plt.figure(figsize=(8, 6))
plt.plot(Ns, medias, 'bo-', markersize=4, label="Média das medições")  # Adicionado '-' para ligar os pontos
plt.axhline(media_esperada, color='r', linestyle='--', label="Média esperada")
plt.fill_between(Ns, media_esperada - incerteza_teorica, media_esperada + incerteza_teorica, 
                 color='gray', alpha=0.3, label="Intervalo de incerteza")
plt.xscale("log")
plt.xlabel("Número de medições (N)")
plt.ylabel("Média")
plt.title("Evolução da média com o número de medições")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()

