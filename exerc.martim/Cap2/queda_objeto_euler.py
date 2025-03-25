import numpy as np
import matplotlib.pyplot as plt


"""a) A aceleração é constante e igual a g. A velocidade varia linearmente com o tempo, pois o objeto está em movimento uniformemente acelerado. 
Portanto, a aceleração é a taxa de variação da velocidade no tempo e a velocidade é proporcional ao tempo devido à aceleração constante."""

#b) - Determinar a velocidade do objeto usando o método de Euler.

g = 9.80
t0 = 0
tf = 4

def euler(dt):

    t = np.arange(t0,tf,dt)
    v = np.empty(np.size(t))
    y = np.empty(np.size(t))

    v[0] = 0
    y[0] = 0

    for i in range(1,len(t)):
        v[i] = v[i-1] + g*dt
        y[i] = y[i-1] + v[i-1]*dt


    tempo_desejado = 3
    indice = int(tempo_desejado/dt)
    v_3s = v[indice]
    t2 = 2
    indice2=int(t2/dt)
    y_2s=y[indice2]
    
    print(f"A velocidade do objeto em {tempo_desejado} segundos é {v_3s:.2f} m/s.")
    print(f"A posição do objeto em {t2} segundos é {y_2s:.2f} m.")
    print(("O desvio é {}".format(np.abs(y[t2]-19.11))))

euler(0.1)
euler(0.01)
euler(0.001)
euler(0.0001)


# d) neste caso o valor não altera

# h)
passo = np.array([0.1,0.01, 0.001, 0.0001])
desvio = np.array([0.98, 0.098, 0.0098, 0.00098])


plt.loglog(passo, desvio, 'o-')
plt.xlabel("Passo (s)")
plt.ylabel("Desvio da Posição, dx (m)")
plt.show()