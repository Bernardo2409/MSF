import numpy as np
import matplotlib.pyplot as plt

# Parâmetros do sistema
k = 1.0       # Constante elástica das molas externas (N/m)
k_prime = 0.5 # Constante elástica das molas internas (N/m)
m = 1.0       # Massa de cada partícula (kg)

# 1. Configuração da matriz dinâmica e cálculo dos modos normais
K_matrix = np.array([
    [-(k + k_prime), k_prime, 0],
    [k_prime, -2*k_prime, k_prime],
    [0, k_prime, -(k + k_prime)]
]) / m

# Cálculo dos autovalores e autovetores
eigenvalues, eigenvectors = np.linalg.eig(K_matrix)
frequencies = np.sqrt(np.abs(eigenvalues))

# Ordenar e normalizar
sorted_indices = np.argsort(frequencies)
frequencies = frequencies[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]
eigenvectors = eigenvectors / np.linalg.norm(eigenvectors, axis=0)  # Normalização

# 2. Simulação com Euler-Cromer
dt = 0.01
t_max = 20.0
n_steps = int(t_max / dt)
t = np.linspace(0, t_max, n_steps)

def simulate_mode(mode_vector):
    xA, xB, xC = np.zeros(n_steps), np.zeros(n_steps), np.zeros(n_steps)
    vA, vB, vC = np.zeros(n_steps), np.zeros(n_steps), np.zeros(n_steps)
    
    # Inicialização correta garantindo todas as massas sejam excitadas
    xA[0] = 0.1 * mode_vector[0]
    xB[0] = 0.1 * mode_vector[1] 
    xC[0] = 0.1 * mode_vector[2]
    
    for i in range(n_steps-1):
        # Forças e acelerações
        aA = -(k+k_prime)*xA[i] + k_prime*xB[i]
        aB = k_prime*xA[i] - 2*k_prime*xB[i] + k_prime*xC[i]
        aC = k_prime*xB[i] - (k+k_prime)*xC[i]
        
        # Integração
        vA[i+1] = vA[i] + aA*dt
        vB[i+1] = vB[i] + aB*dt
        vC[i+1] = vC[i] + aC*dt
        
        xA[i+1] = xA[i] + vA[i+1]*dt
        xB[i+1] = xB[i] + vB[i+1]*dt
        xC[i+1] = xC[i] + vC[i+1]*dt
    
    return xA, xB, xC

# 3. Visualização com configurações aprimoradas
plt.figure(figsize=(14, 10))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Cores distintas
line_styles = ['-', '--', '-.']  # Estilos diferentes

for mode in range(3):
    xA, xB, xC = simulate_mode(eigenvectors[:, mode])
    
    plt.subplot(3, 1, mode+1)
    
    # Plot com configurações visuais melhoradas
    plt.plot(t, xA, color=colors[0], linestyle=line_styles[0], 
             linewidth=2.5, label=f'Massa A (amp: {eigenvectors[0,mode]:.2f})')
    plt.plot(t, xB, color=colors[1], linestyle=line_styles[1], 
             linewidth=2.0, label=f'Massa B (amp: {eigenvectors[1,mode]:.2f})')
    plt.plot(t, xC, color=colors[2], linestyle=line_styles[2], 
             linewidth=2.0, label=f'Massa C (amp: {eigenvectors[2,mode]:.2f})')
    
    plt.title(f'Modo Normal {mode+1} - ω = {frequencies[mode]:.4f} rad/s', 
              fontsize=12, pad=15)
    plt.xlabel('Tempo (s)', fontsize=10)
    plt.ylabel('Deslocamento (m)', fontsize=10)
    
    # Configurações do eixo e legenda
    max_amp = max(np.max(np.abs(xA)), np.max(np.abs(xB)), np.max(np.abs(xC)))
    plt.ylim(-max_amp*1.3, max_amp*1.3)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9, loc='upper right')

plt.tight_layout()
plt.show()