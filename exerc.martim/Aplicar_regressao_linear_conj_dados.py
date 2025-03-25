import matplotlib.pyplot as plt
import numpy as np
import calcular_regressao_linear as c

l = np.array([222.0,207.5,194.0,171.5,153.0,133.0,113.0,92.0])
x_data = np.array([2.3,2.2,2.0,1.8,1.6,1.4,1.2,1.0])

m, b, r2, dm, db = c.minimos_quadrados(l, x_data)

print("m= ", m)
print("b= ", b)
print("r2= ", r2)
print("dm= ", dm)
print("db=", db)

x_line = np.array([90.0, 250.0])
y_line = m * x_line + b

plt.scatter(l, x_data)
plt.xlabel("Distância da fonte de luz ao alvo, L (cm)")
plt.ylabel("Distância entre máximos luminosos consecutivos, X (cm)")

plt.plot(x_line, y_line)
print("X = {0:.4f} cm".format(m*165.0+b)) #L=165.0cm

x_data = np.array([2.3,2.2,2.0,1.8,1.8,1.4,1.2,1.0])
m2, b2, r22, dm2, db2 = c.minimos_quadrados(l, x_data)

print("m= ", m2)
print("b= ", b2)
print("r2= ", r22)
print("dm= ", dm2)
print("db=", db2)

x_line = np.array([90.0, 250.0])
y_line = m2 * x_line + b2

plt.scatter(l, x_data)
plt.xlabel("Distância da fonte de luz ao alvo, L (cm)")
plt.ylabel("Distância entre máximos luminosos consecutivos, X (cm)")

# Linha de regressão
plt.plot(x_line, y_line)
plt.show()








