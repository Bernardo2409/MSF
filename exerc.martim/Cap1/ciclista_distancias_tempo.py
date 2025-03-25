import numpy as np
import matplotlib.pyplot as plt

def minimos_quadrados(x, y):
    # Número de pontos
    N = x.size
    
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_x2 = np.sum(x ** 2)
    sum_y2 = np.sum(y ** 2)
    sum_xy = np.sum(x * y)

    # Coeficientes da reta
    m = (N * sum_xy - sum_x * sum_y) / (N * sum_x2 - sum_x ** 2)
    b = (sum_x2 * sum_y - sum_x * sum_xy) / (N * sum_x2 - sum_x ** 2)

    # Coeficiente de determinação
    r2 = ((N * sum_xy - sum_x * sum_y) ** 2) / ((N * sum_x2 - sum_x ** 2) * (N * sum_y2 - sum_y ** 2))

    # Incertezas
    dm = np.abs(m) * np.sqrt((1 / r2 - 1) / (N - 2))
    db = dm * np.sqrt(sum_x2 / N)

    return m, b, r2, dm, db



# 6  - Problemas Capitulo 1
d = np.array([0.00, 0.735, 1.363, 1.739, 2.805, 3.814, 4.458, 5.666, 6.329])
t = np.array([1,2,3,4,5,6,7,8,9])

# a)
plt.scatter(t,d)
plt.ylabel("Distância percorrida (m)")
plt.xlabel("Tempo (m)")


#b)
m,b,r2,dm,db = minimos_quadrados(t,d)
print(f"Declive (m): {m}")
print(f"Ordenada na origem (b): {b}")
print(f"Coeficiente de determinação (r²): {r2}")
print(f"Erro no declive (dm): {dm}")
print(f"Erro na ordenada na origem (db): {db}")
plt.show()

#c)

vm = ((np.size(d)-1)-d[0])/((np.size(t)-1)-t[0])
print("Velocidade média do ciclista: {:.2f} ".format(vm))

#d) 

m2, b2 = np.polyfit(t, d, 1)

d_pred = m2 * t + b2
ss_res = np.sum((d - d_pred) ** 2)
ss_tot = np.sum((d - np.mean(d)) ** 2)
r22 = 1 - (ss_res / ss_tot)

print(f"y = {m2:.3f}x + {b2:.2f} E o r² é: {r22:.5f}")

plt.scatter(t,d, color = 'blue', label = 'Data Points')
x_line = np.linspace(min(t), max(t))
y_line = m * x_line + b
plt.plot(x_line, y_line, color='red', label=f'y = {m:.3f}x + {b:.2f}')
plt.grid(True)
plt.xlabel("Tempo(min)")
plt.ylabel("Distância(Km)")
plt.legend()
plt.show()

#e)
velocidade_kmh = m * 60  
print(f"Velocidade do ciclista: {velocidade_kmh:.2f} km/h")
