import numpy as np
import matplotlib.pyplot as plt

def movimento_analitico(v0, g=9.81):
    t_subida = v0 / g  
    t_total = 2 * t_subida  
    
    def y(t):
        return v0 * t - 0.5 * g * t**2
    
    return y, t_subida, t_total

def metodo_de_euler(dt=0.01, v0=10, g=9.81, v_terminal=100/3.6):
    k = g / v_terminal**2 
    
    t = [0]
    v = [v0]
    y = [0]
    
    while y[-1] >= 0 or v[-1] > 0:
        a = -g * (1 + (v[-1]/v_terminal)**2)
        v_new = v[-1] + a * dt
        y_new = y[-1] + v[-1] * dt
        
        v.append(v_new)
        y.append(y_new)
        t.append(t[-1] + dt)
    
    return np.array(t), np.array(y), np.array(v)

y_analitico, t_max, t_total = movimento_analitico(10)
print(f"Altura máxima (sem resistência): {y_analitico(t_max):.2f} m, Tempo até altura máxima: {t_max:.2f} s")
print(f"Tempo total (sem resistência): {t_total:.2f} s")

t, y, v = metodo_de_euler()
altura_max_numerica = np.max(y)
tempo_max_numerico = t[np.argmax(y)]
tempo_total_numerico = t[-1]

print(f"Altura máxima (com resistência): {altura_max_numerica:.2f} m, Tempo até altura máxima: {tempo_max_numerico:.2f} s")
print(f"Tempo total (com resistência): {tempo_total_numerico:.2f} s")

tl = np.linspace(0, t_total, 100)
yl = [y_analitico(ti) for ti in tl]

plt.figure(figsize=(10, 5))
plt.plot(tl, yl, label='Sem resistência (Analítico)', linestyle='dashed')
plt.plot(t, y, label='Com resistência (Euler)', linestyle='solid')
plt.axhline(altura_max_numerica, linestyle='dotted', color='red', label='Altura Máxima')
plt.xlabel('Tempo (s)')
plt.ylabel('Altura (m)')
plt.legend()
plt.title('Movimento da Bola')
plt.grid()
plt.show()