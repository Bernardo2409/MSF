import numpy as np
import matplotlib as plt

N = 10

#X--------------------
X = np.random.normal(4.5,0.5,size=N)
Xmedia = np.mean(X)
Xerro = np.std(X)/np.sqrt(N)

print("X = ", ["{:.1f}".format(Xi) for Xi in X])
print("Xmedia = {:.1f}".format(Xmedia))
print("Xerro = {:.1f}".format(Xerro))

#Y--------------------
Y = np.random.normal(10.0,1.0,size=N)
Ymedia = np.mean(Y)
Yerro = np.std(Y)/np.sqrt(N)

print("Y = ", ["{:.1f}".format(Yi) for Yi in Y])
print("Ymedia = {:.1f}".format(Ymedia))
print("Yerro = {:.1f}".format(Yerro))

#Z--------------------
Z = X + Y
Zmedia= np.mean(Z)
Zerro_frm = Xerro + Yerro
Zerro = np.std(Z)/np.sqrt(N)

print("Z = ", ["{:.1f}".format(Zi) for Zi in Z])
print("Zmedia = {:.1f}".format(Zmedia))
print("Zerro = {:.1f}".format(Zerro))
print("Zerro_frm = {:.1f}".format(Zerro_frm))

#W--------------------
W = X * Y
Wmedia = np.mean(W)
Werro = np.std(W)/np.sqrt(W)
Werro_frm = (np.abs(Xerro/Xmedia) + np.abs(Yerro/Ymedia)) * Xmedia * Ymedia