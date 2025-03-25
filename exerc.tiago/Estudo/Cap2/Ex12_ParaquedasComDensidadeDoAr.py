import numpy as np
import matplotlib.pyplot as plt

def EulerComResistencia(y0, v0, t0, tf, dt, g, v_terminal):
    # Número de passos
    N = int((tf - t0) / dt + 0.5)  # Arredonda para o inteiro mais próximo
    
    # Inicializa arrays
    t = np.zeros(N+1)
    y = np.zeros(N+1)
    v = np.zeros(N+1)
    a = np.zeros(N+1)
    # Condições iniciais
    t[0] = t0
    y[0] = y0
    v[0] = v0
    a[0] = 0

    k = g / v_terminal**2
    
    for i in range(N):
        t[i+1] = t[i] + dt
        
        acceleration = -g - k * v[i] * abs(v[i])
        
        a[i+1] = acceleration
        v[i+1] = v[i] + acceleration * dt
        y[i+1] = y[i] + v[i] * dt
        
        if y[i+1] < 0: #Quando Bate no chão para
            break
    
    return t[:i+2], y[:i+2], v[:i+2]

rho = 1225 #kg/m^3
vT_livre = 60 #m/s
vT_aberto = 5 #m/s

y0 = 1000 #m
v0 = 0
t0 = 0
tf = 1000
dt = 0.001
g = 9.8

#a)
print("a)")
t_vals_livre,y_vals_livre,v_vals_livre = EulerComResistencia(y0, v0, t0, tf, dt, g, vT_livre)
plt.plot(t_vals_livre,y_vals_livre,label="Paraquedas fechado",color="red")
plt.ylabel("Distância (m)")
plt.xlabel("Tempo (s)")
plt.legend()
plt.show()

t_chegada = t_vals_livre[-1]
print("O Paraquediasta demora {:.2f} segundos para chegar ao chão com o paraquedas fechado".format(t_chegada))

v_chegada = abs(v_vals_livre[-1]) *3600 /1000
print("O Paraquediasta chega ao chão com a velocidade de {:.2f} km/h com o paraquedas fechado".format(v_chegada))

#b)
print("\nb)")
t_vals_aberto,y_vals_aberto,v_vals_aberto = EulerComResistencia(y0, v0, t0, tf, dt, g, vT_aberto)
plt.plot(t_vals_aberto,y_vals_aberto,label="Paraquedas Aberto",color="red")
plt.ylabel("Distância (m)")
plt.xlabel("Tempo (s)")
plt.legend()
plt.show()

t_chegada = t_vals_aberto[-1]/60
print("O Paraquediasta demora {:.2f} minutos para chegar ao chão com o paraquedas aberto".format(t_chegada))

v_chegada = abs(v_vals_aberto[-1]) *3600 /1000
print("O Paraquediasta chega ao chão com a velocidade de {:.2f} km/h com o paraquedas aberto".format(v_chegada))

#c) Considere a alínea anterior, considerando que o paraquedas fica aberto 20 s depois do salto do avião.
print("\nc)")
def EulerComResistencia2(y0, v0, t0, tf, dt, g,vT_aberto,vT_livre):
    # Número de passos
    N = int((tf - t0) / dt + 0.5)  # Arredonda para o inteiro mais próximo
    
    # Inicializa arrays
    t = np.zeros(N+1)
    y = np.zeros(N+1)
    v = np.zeros(N+1)
    a = np.zeros(N+1)
    # Condições iniciais
    t[0] = t0
    y[0] = y0
    v[0] = v0
    a[0] = 0

    
    for i in range(N):
        t[i+1] = t[i] + dt
        if t[i+1]>=20:
            k = g/vT_aberto**2
        else:
            k= g/vT_livre**2
        acceleration = -g - k * v[i] * abs(v[i])
        
        a[i+1] = acceleration
        v[i+1] = v[i] + acceleration * dt
        y[i+1] = y[i] + v[i] * dt
        
        if y[i+1] < 0: #Quando Bate no chão para
            break
    
    return t[:i+2], y[:i+2], v[:i+2]

t_vals_fechado_aberto,y_vals_fechado_aberto,v_vals_fechado_aberto = EulerComResistencia2(y0, v0, t0, tf, dt, g, vT_aberto, vT_livre)

plt.plot(t_vals_fechado_aberto,y_vals_fechado_aberto,label="Paraquedas fechado durante 20 segundos",color="red")
plt.ylabel("Distância (m)")
plt.xlabel("Tempo (s)")
plt.legend()
plt.show()

t_chegada = t_vals_fechado_aberto[-1]
print("O Paraquediasta demora {:.2f} segundos para chegar ao chão com o paraquedas fechado durante 20 segundos".format(t_chegada))

v_chegada = abs(v_vals_fechado_aberto[-1]) *3600 /1000
print("O Paraquediasta chega ao chão com a velocidade de {:.2f} km/h com o paraquedas fechado durante 20 segundos".format(v_chegada))

#d) Resolva novamente a questão anterior tendo em consideração que a densidade do ar varia de acordo com   𝜌(ℎ) = 1.225 𝑒^(-0.1234*h) kg/m3 para alturas até 90 km, sendo a altura medida em quilómetros. Considere que o coeficiente de resistência do ar é proporcional à densidade do ar.
print("\nd)")
def EulerComResistenciaVariavel(y0, v0, t0, tf, dt, g, vT_aberto, vT_livre,rho0):
    # Número de passos
    N = int((tf - t0) / dt + 0.5)  # Arredonda para o inteiro mais próximo
    
    # Inicializa arrays
    t = np.zeros(N+1)
    y = np.zeros(N+1)
    v = np.zeros(N+1)
    a = np.zeros(N+1)
    # Condições iniciais
    t[0] = t0
    y[0] = y0
    v[0] = v0
    a[0] = 0
    
    h_factor = 0.1234  # Fator de variação da densidade com a altura
    
    for i in range(N):
        t[i+1] = t[i] + dt
        
        # Calcula a densidade do ar dependendo da altura
        rho = rho0 * np.exp(-h_factor * (y[i] / 1000))  # h em km
        
        # Escolhe a velocidade terminal com base no tempo
        if t[i+1] >= 20:
            vT = vT_aberto
        else:
            vT = vT_livre
        
        k = g / vT**2 * (rho / rho0)  # Ajuste do coeficiente de resistência do ar
        acceleration = -g - k * v[i] * abs(v[i])
        
        a[i+1] = acceleration
        v[i+1] = v[i] + acceleration * dt
        y[i+1] = y[i] + v[i] * dt
        
        if y[i+1] < 0:  # Quando bate no chão, para
            break
    
    return t[:i+2], y[:i+2], v[:i+2]

t_vals_var,y_vals_var,v_vals_var = EulerComResistenciaVariavel(y0, v0, t0, tf, dt, g, vT_aberto, vT_livre,rho)

plt.plot(t_vals_var,y_vals_var,label="Paraquedas fechado por 20s e densidade variável",color="red")
plt.ylabel("Distância (m)")
plt.xlabel("Tempo (s)")
plt.legend()
plt.show()

t_chegada = t_vals_var[-1]
print("O Paraquedista demora {:.2f} segundos para chegar ao chão com densidade do ar variável".format(t_chegada))

v_chegada = abs(v_vals_var[-1]) * 3600 / 1000
print("O Paraquedista chega ao chão com a velocidade de {:.2f} km/h com densidade do ar variável".format(v_chegada))
