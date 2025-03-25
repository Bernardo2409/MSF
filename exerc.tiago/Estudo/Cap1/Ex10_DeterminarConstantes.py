import numpy as np
import matplotlib.pyplot as plt

R = np.array([6.37, 7.02, 7.61, 8.02, 8.43, 8.92, 9.31, 9.78, 10.25, 10.74])
a = np.array([9.8, 8.0, 6.6, 6.3, 5.5, 5.1, 4.6, 4.1, 3.8, 3.6])

G = 6.67 * 10**(-11)
Mt = 5.98 * 10**(24)

log_R = np.log(R)
log_a = np.log(a)

m, b = np.polyfit(log_R, log_a, 1)

K =np.exp(b)

#Plot dos logaritmos
plt.scatter(log_R, log_a, label='Data points')
plt.plot(log_R, m * log_R + b, color='red', label=f'Fit: y = {m:.2f}x + {b:.2f}')
plt.xlabel('ln(R)')
plt.ylabel('ln(a)')
plt.legend()
plt.show()

print(f"declive: {m}")
print(f"Ordenada na origem: {b}")
print(f"K: {K}")


K1 = G * Mt

K2 = a * R**2

print("K1: ",K1)
print("K2: ",K2)