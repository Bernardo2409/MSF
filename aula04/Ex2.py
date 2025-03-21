import numpy as np
import matplotlib.pyplot as plt

g = -9.8
v0 = 10
y0 = 0
v_terminal = 100 / 3.6  #km/h  pra m/s

# a) 
def movimento_sem_resistencia(t, v0, g):
    return v0 * t + 0.5 * g * t**2

# b)
t_max = -v0 / g
y_max = movimento_sem_resistencia(t_max, v0, g)
print(f"altura máxima sem resistência de ar: {y_max:.2f} m em {t_max:.2f} s")

# c) 
t_voltar = 2 * t_max
print(f"tempo de retorno à posição inicial sem resistência de ar: {t_voltar:.2f} s")

# d)  Euler Ex.1
def movimento_com_resistencia(v0, g, v_terminal, dt, tf):
    t = [0]
    y = [0]
    v = [v0]
    
    while y[-1] >= 0:
        dv = g * (1 + v[-1] / v_terminal) * dt
        v.append(v[-1] + dv)
        y.append(y[-1] + v[-1] * dt)
        t.append(t[-1] + dt)
    
    return np.array(t), np.array(y)

# e) 
dt = 0.01
tf = 5
t, y = movimento_com_resistencia(v0, g, v_terminal, dt, tf)

altura_max = np.max(y)
t_max_euler = t[np.argmax(y)]
t_voltar_euler = t[-1]

print(f"Altura máxima (com resistência): {altura_max:.2f} m em {t_max_euler:.2f} s")
print(f"Tempo de retorno à posição inicial (com resistência): {t_voltar_euler:.2f} s")


plt.figure(figsize=(10, 5))
plt.plot(t, y, label="Com resistência do ar")

t_continuo = np.linspace(0, t_voltar, 100)
y_continuo = movimento_sem_resistencia(t_continuo, v0, g)

plt.plot(t_continuo, y_continuo, 'o', label="Sem resistência do ar")
plt.xlabel("Tempo(s)")
plt.ylabel("Altura(m)")
plt.legend()
plt.grid()
plt.title("Movimento vertical com e sem resistência do ar")
plt.show()


