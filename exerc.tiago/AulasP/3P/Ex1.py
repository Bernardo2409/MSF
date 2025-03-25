import sympy as sp;
import numpy as np;
import matplotlib.pyplot as plt;

def Euler(x0,v0,t0,tf,dt,g):

    #N = Número de passos 
    N = int((tf - t0) / (dt) + 0.1) #(0.1 para arredondar para cima)

    #Iniciar variáveis
    x = np.zeros(N+1)
    v = np.zeros(N+1)
    t = np.zeros(N+1)

    #Valores iniciais
    v[0] = v0
    t[0] = t0
    x[0] = x0

    #Preencher vetores
    for i in range(N):
        x[i+1] = x[i] + dt*v[i] #Distância percorrida
        t[i+1] = t[i] + dt
        v[i+1] = g*t[i+1]
    
    return t,x,v

x,y,m,b = sp.symbols("x, y, m, b,")
y=m*x**2 + b
y2 = y.subs([(m,0.01),(b,0.0)])

y_em_1 = y2.evalf(subs={x:1}) 
y_lam = sp.lambdify(x,y2,'numpy')

#2
v0=0
a = 3
x0 = 0
vf =250
# 250 km h = 250 *1000 /3600 m/s**2
vm = vf*1000/3600 #69.44
tdesc = vm/a #instante em que descola

xdesc = x0 + v0*tdesc +1/2*a*tdesc**2  #posição em que descola

print("Instante de descolagem: ",tdesc)
print("Posição de descolagem: ",xdesc)

#3

tin = sp.symbols("t")
vin = sp.integrate(a,tin)
xin = sp.integrate(vin,tin)

print("Integral (v) [3*t]: ",vin)
print("Integral (x) [(3/2) * (t**2)]",xin)

#Representação gráfica
dt = 1
t0 = 0
t,x,v = Euler(x0,v0,t0,tdesc,dt,a)



plt.plot(t,x,'g')
plt.plot(t,v,'r')
plt.xlabel("Tempo (s)")
plt.ylabel("Distancia (m)")
plt.show()
