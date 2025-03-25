import numpy as np
import matplotlib.pyplot as plt

n=10
X = np.random.normal(4.5,0.5,size=n) #gerar 10 valores de x com media de 4.5 e desvio 0.5
Xmedia = np.mean(X) #encontrar a média dos valores
Xerro = np.std(X)/np.sqrt(N) #encontrar o valor

Y = np.random.normal(10,1,size=n) #gerar 10 valores de y com media de 1 e desvio 1
Ymedia = np.mean(Y) #encontra a media dos valores
Yerro = np.std(Y)/np.sqrt(N)

Z = X + Y  #soma de cada par de valores
Zmedia = np.mean(Z)
Zerro1 = Xerro + Yerro  #calculo da incerteza diretamente do desvio padrao dos valores de Z
Zerro2 = np.std(Z)/np.sqrt(N) #calculo da incerteza diretamente do desvio padrao dos valores de Z

W = X*Y #produto de cada par de valores
Wmedia = np.mean(W)
Werro1 = Xerro*Yerro
Wincerteza2 = ((Yerro/Ymedia)+ (Xerro/Xmedia))*Wmedia


