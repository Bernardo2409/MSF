import numpy as np
import matplotlib.pyplot as plt

# Parâmetros do sistema
k = 1.0
k_prime = 0.5
m = 1.0
xA_eq = 1.0
xB_eq = 1.2

# Configuração da simulação
dt = 0.01  # passo de tempo
t_max = 30.0  # tempo máximo
n_steps = int(t_max / dt)
t = np.linspace(0, t_max, n_steps)

# Função para calcular as acelerações
def accelerations(xA, xB):
    xA_dotdot = -1.5*xA + 0.5*xB + 1.1
    xB_dotdot = 0.5*xA - 1.5*xB + 1.3
    return xA_dotdot, xB_dotdot

# Função para simular o movimento
def simulate(xA0, xB0, vA0=0.0, vB0=0.0):
    # Arrays para armazenar resultados
    xA = np.zeros(n_steps)
    xB = np.zeros(n_steps)
    vA = np.zeros(n_steps)
    vB = np.zeros(n_steps)
    
    # Condições iniciais
    xA[0] = xA0
    xB[0] = xB0
    vA[0] = vA0
    vB[0] = vB0
    
    # Calcular primeira aceleração
    aA_prev, aB_prev = accelerations(xA[0], xB[0])
    
    # Integração usando Verlet
    for i in range(1, n_steps):
        # Atualizar posições
        xA[i] = xA[i-1] + vA[i-1]*dt + 0.5*aA_prev*dt**2
        xB[i] = xB[i-1] + vB[i-1]*dt + 0.5*aB_prev*dt**2
        
        # Calcular novas acelerações
        aA_new, aB_new = accelerations(xA[i], xB[i])
        
        # Atualizar velocidades
        vA[i] = vA[i-1] + 0.5*(aA_prev + aA_new)*dt
        vB[i] = vB[i-1] + 0.5*(aB_prev + aB_new)*dt
        
        # Preparar para próxima iteração
        aA_prev, aB_prev = aA_new, aB_new
    
    return xA, xB, vA, vB

# Casos a serem simulados
cases = [
    ("i) xA0 = xA_eq + 0.05, xB0 = xB_eq + 0.05", xA_eq + 0.05, xB_eq + 0.05, 0.0, 0.0),
    ("ii) xA0 = xA_eq + 0.05, xB0 = xB_eq - 0.05", xA_eq + 0.05, xB_eq - 0.05, 0.0, 0.0),
    ("iii) xA0 = xA_eq + 0.05, xB0 = xB_eq", xA_eq + 0.05, xB_eq, 0.0, 0.0)
]

# Simular e plotar cada caso
for title, xA0, xB0, vA0, vB0 in cases:
    xA, xB, vA, vB = simulate(xA0, xB0, vA0, vB0)
    
    plt.figure(figsize=(12, 6))
    plt.plot(t, xA, label='Corpo A')
    plt.plot(t, xB, label='Corpo B')
    plt.title(title)
    plt.xlabel('Tempo (s)')
    plt.ylabel('Posição (m)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Análise de frequência para casos i e ii
    if title.startswith(('i', 'ii')):
        # Calcular FFT para encontrar frequências
        fft_A = np.fft.fft(xA - np.mean(xA))
        freqs = np.fft.fftfreq(len(t), dt)
        
        # Pegar apenas frequências positivas
        mask = freqs > 0
        freqs = freqs[mask]
        fft_A = np.abs(fft_A[mask])
        
        # Encontrar picos de frequência
        peaks = np.argsort(fft_A)[-2:]
        peak_freqs = freqs[peaks]
        
        print(f"\n{title}")
        print(f"Frequências angulares observadas: {2*np.pi*peak_freqs} rad/s")
        print(f"Períodos observados: {1/peak_freqs} s")

# Frequências teóricas esperadas
# Para um sistema acoplado como este, esperamos duas frequências normais:
# ω1 = sqrt(k/m) = 1 rad/s
# ω2 = sqrt((k + 2k')/m) = sqrt(2) ≈ 1.414 rad/s
print("\nFrequências angulares teóricas esperadas:")
print("ω1 = 1.0 rad/s (T1 ≈ 6.28 s)")
print("ω2 = √2 ≈ 1.414 rad/s (T2 ≈ 4.44 s)")