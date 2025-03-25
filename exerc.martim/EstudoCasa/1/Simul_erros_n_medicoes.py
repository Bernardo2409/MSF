import numpy as np3
import matplotlib.pyplot as plt



media_esperada = 10
desvio_padrao_esperado = 0.5

valores_N = np.logspace(1, 4, num=50, dtype=int)

medias = []
incertezas = []

# Simulação 
for N in valores_N:
    medicoes = np.random.normal(media_esperada, desvio_padrao_esperado, N)
    media = np.mean(medicoes)
    medias.append(media)
    incerteza = desvio_padrao_esperado / np.sqrt(N)
    incertezas.append(incerteza)

fig,ax = plt.subplots()
ax.plot(valores_N, medias)
ax.axhline(y=media_esperada, color='r', linestyle='-')
ax.plot(valores_N, media_esperada + np.array(incertezas), 'r--')
ax.plot(valores_N, media_esperada + np.array(incertezas), 'r--')

ax.set_xscale('log')

ax.legend()
ax.set_xlabel('N (escala logarítmica)')
ax.set_ylabel('Média')

plt.show()