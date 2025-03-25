import numpy as np
import matplotlib.pyplot as plt

T = np.linspace(5,50,10) #de 5 a 50 espaçados igualmente (50/10 = 5, logo é de 5 em 5)
R = np.array([9.676,6.355,4.261,2.729,1.862,1.184,0.7680,0.4883,0.3461,0.2119])

plt.plot(T,R,'.')
plt.show()
#Não é linear

# Aplicar log nos valores de R para linearizar a equação
log_R = np.log(R)

# Ajuste linear no log de R
m, b = np.polyfit(T, log_R, 1)  

# Criar os valores ajustados
R_fit = np.exp(m * T + b)  # Reverter o logaritmo para obter R ajustado

# Plot
plt.semilogy(T, R, '.', label="Dados semilogy")  # Dados reais
plt.semilogy(T, R_fit, 'r-', label="Ajuste linear")  # Reta ajustada

plt.xlabel("Temperatura (K)")
plt.ylabel("Resistência (log scale)")
plt.legend()
plt.show()