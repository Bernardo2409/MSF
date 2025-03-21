import numpy as np
import matplotlib.pyplot as plt


T = np.array( [ 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100])
E = np.array( [0.6950, 4.363, 15.53, 38.74, 75.08, 125.2, 257.9, 344.1, 556.4, 690.7])

plt.figure(figsize=(13, 5))
plt.subplot(1,2,1)
plt.scatter( T, E, color = "blue",)
plt.plot (T, E)
plt.grid(True)



T_log = np.log(T)
E_log = np.log(E)
plt.subplot(1,2,2)
plt.scatter( T_log , E_log, color = 'red')
plt.plot (T_log, E_log, color = 'red')


plt.tight_layout()
plt.show()