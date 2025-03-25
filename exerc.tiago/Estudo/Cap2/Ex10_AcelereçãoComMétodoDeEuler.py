import numpy as np
import matplotlib.pyplot as plt



v0 = 200 * (1000 / 3600) 
v_terminal = 6.8  
altura_inicial = 200  
g = 9.8  #m/s²

def queda_volante(y0, v0, t0, tf, dt, g, v_terminal, m):
    k = (m * g) / (v_terminal ** 2)  # Coeficiente de resistência do ar
    
    N = int((tf - t0) / dt) + 1
    t = np.zeros(N)
    y = np.zeros(N)
    v = np.zeros(N)
    a = np.zeros(N)
    
    t[0] = t0
    y[0] = y0
    v[0] = v0  # Velocidade inicial para baixo (positiva)
    a[0] = g - (k/m) * v[0]**2  # Aceleração inicial
    
    t_half = None  # Tempo para reduzir a velocidade em 50%
    t_4m = None #Tempo para precorrer 4 metros
    
    for i in range(N - 1):
        a[i] = g - (k/m) * v[i]**2  # Aceleração instantânea
        v[i+1] = v[i] + a[i] * dt
        y[i+1] = y[i] - v[i] * dt  # Movimento para baixo (y diminui)
        t[i+1] = t[i] + dt
        
        # Verifica se a velocidade reduziu em 50% pela primeira vez
        if v[i+1] <= v0 / 2 and t_half is None:
            t_half = t[i+1]
        if (y0 - y[i+1]) >= 4 and t_4m is None:
            t_4m = t[i+1]
        if y[i+1] <= 0:  # Atingiu o solo
            break
    
    return t[:i+2], y[:i+2], v[:i+2], a[:i+2], t_half, t_4m

# Simulação
t_vals, y_vals, v_vals, a_vals, t_half,t_4m = queda_volante(altura_inicial, v0, 0, 10, 0.001, g, v_terminal, 0.058)

plt.plot(t_vals,y_vals,label="Posição",color="blue")
plt.legend()
plt.show()
plt.plot(t_vals,v_vals,label="Velocidade",color="red")
plt.legend()
plt.show()
plt.plot(t_vals,a_vals,label="Aceleração",color="yellow")
plt.legend()
plt.show()

#c) Em quanto tempo tem o volante reduzida a sua velocidade em 50%?

print(f"Tempo para reduzir a velocidade em 50%: {t_half:.4f} s")
print(f"Tempo para precorrer 4 metros: {t_4m:.4f} s")
print(f"Velocidade nesse instante: {v0/2:.2f} m/s")
print(f"Velocidade terminal: {v_terminal} m/s")