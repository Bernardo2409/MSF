import numpy as np
import matplotlib.pyplot as plt

def minimos_quadrados(X, Y, N):
    sumX = np.sum(X)
    sumY = np.sum(Y)
    sumX2 = np.sum(X ** 2)
    sumY2 = np.sum(Y ** 2)
    sumXY = np.sum(X * Y)

    m = (N * sumXY - sumX * sumY) / ((N * sumX2) - ((sumX)**2)) #declive

    b = (sumX2 * sumY - sumX * sumXY) / (N * sumX2 - (sumX**2)) #ordenada na origem

    r2 = (N * sumXY - sumX * sumY)**2 / ((N* sumX2 - (sumX)**2)*(N * sumY2 - (sumY)**2)) #coeficiente de correlação linear

    dm = abs(m)*(np.sqrt((( 1 / r2 ) - 1)/(N - 2))) 

    db = dm*np.sqrt(sumX2 / N)
    return m,b,r2,dm,db

X = np.array([0,1,2,3,4,5,6,7,8,9]) #tempo em mins
Y = np.array([0.00,0.735,1.363,1.739,2.805,3.814,4.458,4.955,5.666,6.329]) #Distância em kms
N = len(X)

plt.scatter(X,Y,label='Dados Originais')
plt.xlabel("Tempo (mins)")
plt.ylabel("Distância Percorrida (kms)")

m,b,r2,dm,db = minimos_quadrados(X,Y,N)
x = np.array([np.min(X),np.max(X)])
y = m*x+b
plt.plot(x,y,label='Minimos_Quadrados',color='green')

vmedia = (np.sum(Y)/np.sum(X))*1000/60 #Velocidade média = Total de distancia percorrida / total de tempo 
print("A velocidade média do ciclista foi de {:.2f}(m/s)".format(vmedia))

coeficientes = np.polyfit(X, Y, 1) #Ajustar reta de grau 1
declive = coeficientes[0] #Primeiro elemento da reta = declive
ordenadaNaOrigem = coeficientes[1] #Segundo elemento da reta = OrdenadaNaOrigem
tempo_ajustado = np.linspace(0,9,100) #Calcular tempo de 0 a 9 mns com 100 passos
distancia_ajustada = declive*tempo_ajustado + ordenadaNaOrigem

plt.plot(tempo_ajustado, distancia_ajustada, label='Reta Ajustada', color='red')

vhora = vmedia *3600/1000
print("A velocidade média do ciclista foi de {:.2f}(km/h)".format(vhora))

plt.show()
