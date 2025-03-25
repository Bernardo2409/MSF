import numpy as np
import matplotlib.pyplot as plt

massa = 0.058  # kg (58 g)
v0 = 0  # m/s 
v_terminal = 6.80  # m/s
g = 9.8  # m/s²
# v_T = g/C
C = g / v_terminal  # Coeficiente de resistência linear
y0 = 0
def euler_velocidade(y0, v0, t0, tf, dt, g, C):
    N = int((tf - t0) / dt) + 1
    t = np.zeros(N)
    v = np.zeros(N)
    y = np.zeros(N)
    
    t[0] = t0
    v[0] = v0
    y[0] = y0
    
    for i in range(N - 1):
        a = g - C * v[i]  # Aceleração instantânea
        v[i+1] = v[i] + a * dt
        y[i+1] = y[i] - v[i] * dt
        t[i+1] = t[i] + dt
    
    return t, v, y

# Simulação
t, v, y = euler_velocidade(y0, v0, 0, 5, 0.01, g, C)

# Gráfico
plt.plot(t, v, label="Velocidade (m/s)", color="blue")
plt.axhline(y=v_terminal, color='red', linestyle='--', label=f"$v_T$ = {v_terminal} m/s")
plt.xlabel("Tempo (s)")
plt.ylabel("Velocidade (m/s)")
plt.title("Evolução da Velocidade com Resistência Linear")
plt.legend()
plt.grid(True)
plt.show()

plt.plot(t, y, label="Posição (m)", color="green")
plt.xlabel("Tempo (s)")
plt.ylabel("Altura (m)")
plt.title("Queda do Volante: Posição vs Tempo")
plt.legend()
plt.grid(True)
plt.show()