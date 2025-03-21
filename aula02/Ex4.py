import numpy as np
import matplotlib.pyplot as plt

tempo = np.arange(0, 50, 5) # 10*5 = 50
atividade = np.array([9.676, 6.355, 4.261, 2.729, 1.862, 1.184, 0.7680, 0.4883, 0.3461, 0.2119]) 

# a)
plt.figure(figsize=(10, 5))
plt.plot(tempo, atividade, 'o-', label='Atividade (mCi)')
plt.xlabel('Tempo (dias)')
plt.ylabel('Atividade (mCi)')
plt.title('Atividade do isotopo com o tempo')
plt.legend()
plt.grid(True)
plt.show()

# b) 
log_atividade = np.log(atividade)

plt.figure(figsize=(10, 5))
plt.plot(tempo, log_atividade, 'o-',color = 'g', label='Logaritmo da Atividade com o tempo')
plt.xlabel('Tempo (dias)')
plt.ylabel('Logaritmo da Atividade')
plt.title('Logaritmo da Atividade do isotopo com o tempo')
plt.legend()
plt.grid(True)  
plt.show()

# c) 
coef = np.polyfit(tempo, log_atividade, 1)
a, b = coef

A0 = np.exp(b) # valor inicial

def atividade_func(t):
    return A0 * np.exp(a * t)

# constantes
print(f"Taxa de decaimento radioativo = {a}")
print(f"Atividade inicial = {A0}")

metade = -np.log(2) / a
print(f"A semivida do isótopo é: {metade:.2f} dias")

tempo_ajustado = np.linspace(0, 45, 100)
atividade_ajustada = atividade_func(tempo_ajustado)

plt.figure(figsize=(10, 5))
plt.plot(tempo, atividade, 'o', label='Dados experimentais')
plt.plot(tempo_ajustado, atividade_ajustada, '-', color = 'r', label='Função ajustada')
plt.xlabel('Tempo (dias)')
plt.ylabel('Atividade (mCi)')
plt.title('Ajuste da Atividade do isotopo com o tempo')
plt.legend()
plt.grid(True)
plt.show()