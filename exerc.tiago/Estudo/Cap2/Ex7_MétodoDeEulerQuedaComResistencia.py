import numpy as np
import matplotlib.pyplot as plt

#a)
def euler_projectile_with_drag(x0, v0, t0, tf, dt, g, v_terminal):
    # Número de passos
    N = int((tf - t0) / dt + 0.5)  # Arredonda para o inteiro mais próximo
    
    # Inicializa arrays
    t = np.zeros(N+1)
    x = np.zeros(N+1)
    v = np.zeros(N+1)
    a = np.zeros(N+1)
    # Condições iniciais
    t[0] = t0
    x[0] = x0
    v[0] = v0
    a[0] = 0
    # Coeficiente de resistência do ar (k) calculado a partir de v_terminal
    k = g / v_terminal**2
    
    # Método de Euler
    for i in range(N):
        t[i+1] = t[i] + dt
        
        # Aceleração: dv/dt = -g - k * v * |v| (considerando arrasto quadrático)
        acceleration = -g - k * v[i] * abs(v[i])
        
        a[i+1] = acceleration
        v[i+1] = v[i] + acceleration * dt
        x[i+1] = x[i] + v[i] * dt
        
        if x[i+1] < 0:
            break
    
    return t[:i+2], x[:i+2], v[:i+2]

# Parâmetros
v0 = 10.0 
x0 = 0.0  
t0 = 0.0  
tf = 20.0  
dt = 0.01 
g = 9.8   
v_terminal = 100 * 1000 / 3600  

t, x, v = euler_projectile_with_drag(x0, v0, t0, tf, dt, g, v_terminal)

plt.plot(t,x,label="Posição",color="blue")
plt.legend()
plt.grid(True)
plt.show()

#b)
h_max = np.max(x)
print("Altura máxima: {:.2f}".format(h_max))
idx_max = np.argmax(x)
t_max = t[idx_max]
print("Intante de altura máxima: {:.2f}".format(t_max))

#c)
#Em que instante volta a passar pela posição inicial? 
crossing_idx = np.where(np.diff(np.sign(x)) < 0)[0]
if len(crossing_idx) > 0:
    t_return = t[crossing_idx[0]]
    print(f"c) Volta à posição inicial em t = {t_return:.2f} s")
else:
    print("c) A bola não retornou à posição inicial no tempo simulado.")