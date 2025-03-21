import numpy as np
import matplotlib.pyplot as plt

x = np.array( [ 0 , 1, 2, 3, 4, 5, 6, 7, 8, 9]) #tempo
y = np.array( [0, 0.735, 1.363, 1.739, 2.805, 3.814, 4.458, 4.955, 5.666, 6.329 ]) # distancia




N = len(x)
soma_xy = np.sum(x*y)
soma_x = np.sum(x)
soma_y = np.sum(y)
soma_xx = np.sum(x**2)
soma_yy = np.sum(y**2)


m = (N * soma_xy - (soma_x * soma_y) ) / ( N * soma_xx - ( (soma_x)**2 ) )

b = ( (soma_xx * soma_y) - (soma_x * soma_xy) ) / (N * soma_xx - ( soma_xx **2) )

r2 = ( ( (N * soma_xy)- (soma_x * soma_y) ) **2 ) / ( (N * soma_xx - (soma_x)**2 ) * ( (N * soma_yy) - (soma_y **2))  )  


m2, b2 = np.polyfit(x, y, 1)

y_pred = m2 * x + b2
ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r22 = 1 - (ss_res / ss_tot)

print(f"y = {m2:.3f}x + {b2:.2f} E o r² é: {r22:.5f}")

plt.scatter(x,y, color = 'blue', label = 'Data Points')
x_line = np.linspace(min(x), max(x), 50)
y_line = m * x_line + b
plt.plot(x_line, y_line, color='red', label=f'y = {m:.3f}x + {b:.2f}')
plt.grid(True)
plt.xlabel("Tempo(min)")
plt.ylabel("Distância(Km)")
plt.legend()
plt.show()


velocidade =  m *60 
print(velocidade)