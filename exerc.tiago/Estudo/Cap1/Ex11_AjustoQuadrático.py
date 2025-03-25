import numpy as np
import matplotlib.pyplot as plt

x = np.array([1080, 1044, 1008, 972, 936, 900, 864, 828, 792, 756, 720, 540, 360, 180, 0])
y1 = np.array([0.0, 2.25, 5.25, 7.5, 8.75, 12.0, 13.75, 14.75, 15.5, 17.0, 17.5, 19.5, 18.5, 13.0, 0.0])
y2 = np.array([0.0, 3.25, 6.5, 7.75, 9.25, 12.25, 16.0, 15.25, 16.0, 17.0, 18.5, 20.0, 18.5, 13.0, 0.0])
y3 = np.array([0.0, 4.5, 6.5, 8.25, 9.5, 12.5, 16.0, 15.5, 16.6, 17.5, 18.5, 20.25, 19.0, 13.0, 0.0])
y4 = np.array([0.0, 6.5, 8.75, 9.25, 10.5, 14.75, 16.5, 17.5, 16.75, 19.25, 19.0, 20.5, 19.0, 13.0, 0.0])

y_avg = np.mean([y1, y2, y3, y4], axis = 0)

plt.plot(x,y_avg,label='Média dos Dados Originais')
plt.legend()
plt.show()

#Ajusto Linear
coef_linear = np.polyfit(x,y_avg,1)
m = coef_linear[0]
b = coef_linear[1]
plt.scatter(x, y_avg, label="Dados experimentais")
plt.plot(x, m*x+b, label=f"Ajuste Linear: y = {m:.5f}x + {b:.2f}",color='red')
plt.legend()
plt.show()

#Ajusto Quadrático
coef_quadratic = np.polyfit(x, y_avg, 2)  # Ajuste de 2º grau (parábola)
y_quadratic = np.polyval(coef_quadratic, x)

plt.scatter(x, y_avg, label="Dados experimentais")
plt.plot(x, y_quadratic, label=f"Ajuste Quadrático: y = {coef_quadratic[0]:.5f}x² + {coef_quadratic[1]:.2f}x + {coef_quadratic[2]:.2f}",color='red')
coeffs = np.polyfit(x, y_avg, 2)
poly1 = np.poly1d(coeffs)
x_fit = np.linspace(min(x), max(x), 500)
plt.plot(x_fit, poly1(x_fit), label='Curva', linestyle='--', color = 'red')
plt.legend()
plt.show()





print("Coeficientes do ajuste linear:", coef_linear)
print("Coeficientes do ajuste quadrático:", coef_quadratic)