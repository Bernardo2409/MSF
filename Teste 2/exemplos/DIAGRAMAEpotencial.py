import numpy as np
import matplotlib.pyplot as plt

xeq = 2
x0 = -5
xf = 5
dt = 0.001
n = int((xf-x0)/dt+0.1)

x = np.linspace(x0,xf,n)

def Ep(x):
    return 0.5 * (np.abs(x)-xeq)**2

y = Ep(x)

plt.title("Energia Potencial")
plt.ylabel("Energia [J]")
plt.xlabel("x [m]")
plt.plot(x,y,"-r", label = "Energia Potencial")
plt.ylim(top = 4.5)
plt.legend()
plt.show()