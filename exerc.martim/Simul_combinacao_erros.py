import numpy as np
import matplotlib.pyplot as plt


#1.
n = 10
x = np.random.normal(4.5,0.5, size = n)
Xmedia = np.mean(x)
Xerro = np.std(x)/np.sqrt(n)

print("x",x)
print("Xmedia",Xmedia)
print("Xerro",Xerro)

#2.
Y = np.random.normal(10, 1, size = n)
Ymedia = np.mean(Y)
Yerro = np.std(Y)/np.sqrt(Y)

print(Y)
print(Ymedia)
print(Yerro)

#3.
Z = x + Y 
Zmedia = np.mean(Z)
print(Z)
print(Zmedia)
Zerro1 = np.std(Z) / np.sqrt(n)  
print("Zerro1",Zerro1)
Zerro2 = Xerro + Yerro
print("Zerro2: ", Zerro2)

#4
W = x*Y
Wmedia = np.mean(W)
Wincerteza1 = np.std(W)/np.sqrt(W)
Wincerteza2 = ((Yerro/Ymedia)+ (Xerro/Xmedia))*Wmedia


print("W Incerteza 1", Wincerteza1)
print("W incerteza 2:", Wincerteza2)