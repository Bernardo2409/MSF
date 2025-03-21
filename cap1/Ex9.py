import numpy as np
import matplotlib.pyplot as plt


D = np.array( [5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
V = np.array( [9.676, 6.355, 4.261, 2.729, 1.862, 1.184, 0.768, 0.4883, 0.3461, 0.2119])

plt.figure(figsize=(13, 5))
plt.subplot(1,2,1)
plt.xlabel('Dias')
plt.ylabel('Valores(mCi)')

plt.scatter(D, V, color = 'blue', label = 'valores')
plt.grid(True)
plt.legend()


plt.subplot(1, 2, 2)
plt.xlabel('Dias')
plt.ylabel('Valores (mCi)')
plt.semilogy(D, V, 'o', color='blue', label='valores')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()