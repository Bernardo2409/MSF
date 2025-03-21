import numpy as np
import matplotlib.pyplot as plt

# Constantes
g = 9.8  # Aceleração gravitacional (m/s²)
t_final = 4.0  # Tempo final (s)
v0 = 0.0  # Velocidade inicial (m/s)
y0 = 0.0  # Posição inicial (m)

# Função exata para velocidade e posição
def velocidade_exata(t):
    return v0 + (g * t)

def posicao_exata(t):
    return y0 + v0 * t + 0.5 * g * t**2

# Método de Euler para velocidade
def euler_velocidade(dt):
    t = np.arange(0, t_final + dt, dt)
    v = np.zeros_like(t)
    for i in range(len(t) - 1):
        v[i + 1] = v[i] + g * dt
    return t, v

# Método de Euler para posição
def euler_posicao(dt):
    t = np.arange(0, t_final + dt, dt)
    y = np.zeros_like(t)
    v = np.zeros_like(t)
    for i in range(len(t) - 1):
        v[i + 1] = v[i] + g * dt
        y[i + 1] = y[i] + v[i] * dt
    return t, y

# b) Velocidade com passo dt
dt_b = 0.1
t_b, v_b = euler_velocidade(dt_b)
v_3s_b = v_b[int(3/dt_b)]
print(f"Velocidade em 3s (passo {dt_b}): {v_3s_b:.2f} m/s")

# c) Velocidade com passo 10 vezes menor
dt_c = dt_b / 10
t_c, v_c = euler_velocidade(dt_c)
v_3s_c = v_c[int(3/dt_c)]
print(f"Velocidade em 3s (passo {dt_c}): {v_3s_c:.2f} m/s")

# d) Comparação com o valor exato
v_3s_exata = velocidade_exata(3)
print(f"Velocidade exata em 3s: {v_3s_exata:.2f} m/s")
print(f"Erro (passo {dt_b}): {abs(v_3s_b - v_3s_exata):.2e}")
print(f"Erro (passo {dt_c}): {abs(v_3s_c - v_3s_exata):.2e}")

# e) Posição com passo dt
t_e, y_e = euler_posicao(dt_b)
y_2s_e = y_e[np.abs(t_e - 2).argmin()]
print(f"Posição em 2s (passo {dt_b}): {y_2s_e:.2f} m")

# f) Posição com passo 10 vezes menor
dt_f = dt_b / 10
t_f, y_f = euler_posicao(dt_f)
y_2s_f = y_f[np.abs(t_f - 2).argmin()]
print(f"Posição em 2s (passo {dt_f}): {y_2s_f:.2f} m")

# g) Comparação com o valor exato
y_2s_exata = posicao_exata(2)
print(f"Posição exata em 2s: {y_2s_exata:.2f} m")
print(f"Erro (passo {dt_b}): {abs(y_2s_e - y_2s_exata):.2e}")
print(f"Erro (passo {dt_f}): {abs(y_2s_f - y_2s_exata):.2e}")

# h) Gráfico do desvio em função do passo
passos = [dt_b, dt_f, dt_f / 10, dt_f / 100]
erros = []

for dt in passos:
    _, y_aprox = euler_posicao(dt)
    y_2s_aprox = y_aprox[np.abs(np.arange(0, t_final + dt, dt) - 2).argmin()]
    erros.append(abs(y_2s_aprox - y_2s_exata))

plt.figure()
plt.loglog(passos, erros, marker='o')
plt.title("Erro em função do passo (posição em 2s)")
plt.xlabel("Passo de tempo (s)")
plt.ylabel("Erro (m)")
plt.grid(True, which="both", linestyle="--")
plt.show()