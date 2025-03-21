import numpy as np
import matplotlib.pyplot as plt

g = -9.80
tf = 4
v0 = 0
y0 = 100

def metodo_euler(y0, v0, g, dt, tf):
    t = np.arange(0, tf + dt, dt)  
    y = np.zeros_like(t)
    v = np.zeros_like(t)

    y[0] = y0
    v[0] = v0

    for i in range(len(t) - 1):
        v[i + 1] = v[i] + g * dt
        y[i + 1] = y[i] + v[i] * dt
    return t, y, v

dts = [1, 0.10, 0.01]
erros = []

for dt in dts:
    t, y, v = metodo_euler(y0, v0, g, dt, tf)
    
    idx_t2 = np.argmin(np.abs(t - 2))
    posicao_t2 = y[idx_t2]

    posicao_exata = y0 + v0 * 2 + 0.5 * g * 2**2
    erro = np.abs(posicao_t2 - posicao_exata)
    erros.append(erro)

    print(f"dt: {dt}, Posição aproximada em 2s é: {posicao_t2:.2f} m, Erro: {erro:.2f} m ")


idx_t3 = np.argmin(np.abs(t - 3))
velocidade = v[idx_t3]
posicao = y[idx_t3]

print(f"A velocidade no instante 3s é {velocidade:.2f} m/s.")
print(f"A posição no instante 3s é {posicao:.2f} m.")


plt.figure(figsize=(10, 5))
for dt in dts:
    t, y, ç = metodo_euler(y0, v0, g, dt, tf)
    plt.plot(t, y, label=f"dt = {dt}")

plt.xlabel("Tempo (s)")
plt.ylabel("Altura (m)")
plt.legend()
plt.grid()
plt.title("Posição ao longo do tempo para diferentes passos de tempo")
plt.show()


