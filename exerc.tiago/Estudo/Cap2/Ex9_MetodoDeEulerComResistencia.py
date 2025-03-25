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
m = 0.058
y0 = 5
v0 = 0
t0 = 0
tf = 100
dt = 0.001
g = 9.8
vT_tenis = 100 * (1000 / 3600) 
vT_badmington = 6.8 
t_tenis,y_tenis,v_tenis = EulerComResistencia(y0,v0,t0,tf,dt,g,vT_tenis)
t_badmington,y_badmington,v_badmington = EulerComResistencia(y0,v0,t0,tf,dt,g,vT_badmington)

plt.plot(t_tenis,y_tenis,label="Téns",color="Green")
plt.plot(t_badmington,y_badmington,label="Badmington",color="yellow")
plt.ylabel("Posição (m)")
plt.xlabel("Tempo (s)")
plt.legend()
plt.show()

t_soloTenis = t_tenis[-1]
t_soloBadmington = t_badmington[-1]

print("Instante em que a bola de ténis toca no chão: {:.2f}".format(t_soloTenis))
print("Instante em que o volante de badmington toca no chão: {:.2f}".format(t_soloBadmington))

#O Volante demora mais a tocar no solo.