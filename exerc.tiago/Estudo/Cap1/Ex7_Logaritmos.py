import numpy as np
import matplotlib.pyplot as plt

M = np.array([0.15,0.20,0.16,0.11,0.25,0.32,0.40,0.45,0.50,0.55])
T = np.array([1.21,1.40,1.26,1.05,1.60,1.78,2.00,2.11,2.22,2.33])

plt.plot(M,T,label="Dados")
plt.show()
log_M = np.log(M)
log_T = np.log(T)

m,b = np.polyfit(log_M,log_T,1)

plt.scatter(log_M,log_T)
plt.plot(log_M,log_M*m+b, label=f'y = {m:.3f}x + {b:.2f}',color='red')
plt.show()

