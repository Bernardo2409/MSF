#AVISO: Rever programa!!!!!!


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

m = np.array([0.15,0.20,0.16,0.11, 0.25, 0.32, 0.40, 0.45, 0.50, 0.55])
t = np.array([1.21, 1.40, 1.26, 1.05, 1.60, 1.78, 2.00, 2.11, 2.22, 2.33])


#a) - Apresente as medições num gráfico.
plt.scatter(m,t)
plt.ylabel("T(s)")
plt.xlabel("M(kg)")
plt.show()
# R: Não é linear;



# b) - Apresente as medições num gráfico loglog
plt.loglog(m, t, 'o')
plt.ylabel("T(s) (log scale)")
plt.xlabel("M(kg) (log scale)")

# T = CM^0.5

mlog = np.log(m)
tlog = np.log(t)
md, b, r2, dm, db = minimos_quadrados(mlog, tlog)
y = md * mlog + b  # Equação da reta no espaço log-log

plt.plot(m, np.exp(y), label=f"Ajuste: T = {np.exp(b):.2f} M^{md:.2f}", color='red')
plt.show()

"""c) Considerando  a  relação  entre  o  período  e  a  massa  descoberta  na  alínea  anterior, 
transforme as quantidades de modo a obter um gráfico que apresente uma relação linear. 
Encontre  o  declive,  a  ordenada  na  origem,  os  erros  respetivos  e  o  coeficiente  de 
determinação. É um bom ajuste?"""

t2 = t**2
plt.scatter(m,t2)
plt.xlabel("M(kg)")
plt.ylabel("T**2(s**2)")
md2, b2, r22,dm2,db2 = minimos_quadrados(m,t2)
y2 = md2*t2+b2

print("m = {}+- {} s2/kg".format(md2, dm2))
print("b = {} +- {} s2".format(b2, db2))
print("r2 = ", r22)

#Sim é um bom ajuste (ver r2)

# d) - Calcule a constante elástica

K = 4 * np.pi**2 * (m / t**2)  
k2 = np.mean(K)  # Média de K
std_k = np.std(K, ddof=1)  
print("K = {:.2f} ± {:.2f} kg/s2".format(k2, std_k))