import numpy as np
import matplotlib.pyplot as plt

# Dados
m = 0.25
k = 1.0
b = 0.1

#a) Calcule a lei do movimento quando a massa começa do repouso em x = 0.4m

x0 = 0.4
v0 = 0.0

t_inicio = 0
t_fim = 20
dt = 0.01

t = np.arange(t_inicio, t_fim, dt)

x = np.zeros_like(t)
v = np.zeros_like(t)

x[0] = x0
v[0] = v0

for i in range(1, len(t)):
    a = (-k*x[i-1] - b*v[i-1]) / m
    v[i] = v[i-1] + a * dt
    x[i] = x[i-1] + v[i] * dt

plt.figure
plt.plot(t, x)
plt.title('Posição x(t)')
plt.xlabel('Tempo (s)')
plt.ylabel('Posição (m)')
plt.grid()
plt.show()

# b) Encontrar maximos e minimos locais 

# Funções de interpolação fornecidas
def intlagv(xinp, xm1, xm2, xm3, ym1, ym2, ym3):
    xab = xm1 - xm2
    xac = xm1 - xm3
    xbc = xm2 - xm3

    xi1 = xinp - xm1
    xi2 = xinp - xm2
    xi3 = xinp - xm3
    
    a = xi2 * xi3 / (xab * xac)
    b = -xi1 * xi3 / (xab * xbc)
    c = xi1 * xi2 / (xac * xbc)

    yout = a * ym1 + b * ym2 + c * ym3
    return xinp, yout

def maxminv(xm1, xm2, xm3, ym1, ym2, ym3):
    xab = xm1 - xm2
    xac = xm1 - xm3
    xbc = xm2 - xm3

    a = ym1 / (xab * xac)
    b = -ym2 / (xab * xbc)
    c = ym3 / (xac * xbc)

    xmla = (b + c) * xm1 + (a + c) * xm2 + (a + b) * xm3
    xm = 0.5 * xmla / (a + b + c)

    xta = xm - xm1
    xtb = xm - xm2
    xtc = xm - xm3

    ymax = a * xtb * xtc + b * xta * xtc + c * xta * xtb
    return xm, ymax

# Encontrar índices aproximados dos extremos locais
extremos_idx = []
for i in range(1, len(x)-1):
    if (x[i] > x[i-1] and x[i] > x[i+1]) or (x[i] < x[i-1] and x[i] < x[i+1]):
        extremos_idx.append(i)

# Refinar os extremos usando a função maxminv
t_extremos = []
x_extremos = []
for idx in extremos_idx:
    # Pegar três pontos ao redor do extremo
    if idx > 0 and idx < len(x)-1:
        xm, ym = maxminv(t[idx-1], t[idx], t[idx+1], x[idx-1], x[idx], x[idx+1])
        t_extremos.append(xm)
        x_extremos.append(ym)

# Plot com os extremos marcados
plt.figure(figsize=(10, 5))
plt.plot(t, x, label='Posição')
plt.scatter(t_extremos, x_extremos, color='red', label='Extremos locais')
plt.title('Posição com extremos locais marcados')
plt.xlabel('Tempo (s)')
plt.ylabel('Posição (m)')
plt.legend()
plt.grid(True)
plt.show()

print("Extremos locais encontrados:")
for te, xe in zip(t_extremos, x_extremos):
    print(f"Tempo: {te:.2f} s, Posição: {xe:.4f} m")


# c) Plot do logaritmo das amplitudes e ajuste linear

maximos_t=[]
maximos_x=[]

for i in range(len(t_extremos)):
    if x_extremos[i] > 0:
        maximos_t.append(t_extremos[i])
        maximos_x.append(x_extremos[i])

log_amplitudes = np.log(maximos_x)

m_linear, c_linear = np.polyfit(maximos_t, log_amplitudes, 1)

y_pred = m_linear * np.array(maximos_t) + c_linear

plt.figure(figsize=(10, 5))
plt.scatter(maximos_t, log_amplitudes, label='Dados')
plt.plot(maximos_t, y_pred, 'r', label=f'Ajuste: y = {m_linear:.4f}t + {c_linear:.4f}')
plt.title('Logaritmo das amplitudes em função do tempo')
plt.xlabel('Tempo (s)')
plt.ylabel('ln(Amplitude)')
plt.legend()
plt.grid(True)
plt.show()

# Resultados
print("\nAnálise do decaimento exponencial:")
print(f"Declive (coeficiente angular): {m_linear:.6f}")
print(f"Coeficiente linear: {c_linear:.6f}")

# Comparação com a teoria
gamma = b / (2 * m)
omega0 = np.sqrt(k / m)
omega = np.sqrt(omega0**2 - gamma**2)
m_teorico = -gamma

print("\nComparação com a teoria:")
print(f"Declive experimental: {m_linear:.6f}")
print(f"Declive teórico (-γ): {m_teorico:.6f}")
print(f"Diferença: {abs(m_linear - m_teorico):.6f}")
print(f"Erro relativo: {abs((m_linear - m_teorico)/m_teorico)*100:.2f}%")