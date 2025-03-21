import numpy as np
import sympy as sy
import matplotlib.pyplot as plt

x = np.array([ 222, 207.5, 194, 171.5, 153, 133, 113, 92])
y = np.array([ 2.3, 2.2, 2, 1.8, 1.6, 1.4, 1.2, 1])

N = len(x)
soma_xy = np.sum(x*y)
soma_x = np.sum(x)
soma_y = np.sum(y)
soma_xx = np.sum(x**2)
soma_yy = np.sum(y**2)


m = (N * soma_xy - (soma_x * soma_y) ) / ( N * soma_xx - ( (soma_x)**2 ) )

b = ( (soma_xx * soma_y) - (soma_x * soma_xy) ) / (N * soma_xx - ( soma_xx **2) )

r2 = ( ( (N * soma_xy)- (soma_x * soma_y) ) **2 ) / ( (N * soma_xx - (soma_x)**2 ) * ( (N * soma_yy) - (soma_y **2))  )               


print(f"soma de xy = {soma_xy}. A soma de x = {soma_x}")
print(f"y = {m:.3f}x + {b:.2f} E o r^2 é: {r2}")


plt.figure(figsize = (10, 5))

plt.scatter(x, y, color='blue', label='Data points')

x_line = np.linspace(min(x), max(x))
y_line = m * x_line + b
plt.plot(x_line, y_line, color='red', label=f'y = {m:.3f}x + {b:.2f}')

plt.xlabel("L (cm)")
plt.ylabel("X (cm)")
plt.title("Exercício 5")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()

#e)

descobrir_valor = m * 165 + b
print(descobrir_valor)
