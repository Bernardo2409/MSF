import numpy as np
import matplotlib.pyplot as plt

v_A = 70 * (1000 / 3600)  
a_P = 2.0  

t_captura = (2 * v_A) / a_P

x_captura = 0.5 * a_P * t_captura**2

# Print dos resultados
print(f"Instante de captura: {t_captura:.2f} segundos")
print(f"Distância percorrida pelo carro patrulha: {x_captura:.2f} metros")

t = np.linspace(0, t_captura, 100)

x_A = v_A * t

x_P = 0.5 * a_P * t**2

plt.figure(figsize=(10, 5))
plt.plot(t, x_A, label='Carro A (MRU)', linestyle='--', color='blue')
plt.plot(t, x_P, label='Carro Patrulha (MRUA)', linestyle='-', color='red')
plt.scatter([t_captura], [x_captura], color='black', label='Momento da captura')
plt.xlabel('Tempo (s)')
plt.ylabel('Distância (m)')
plt.title('Perseguição do Carro Patrulha')
plt.legend()
plt.grid()
plt.show()  

