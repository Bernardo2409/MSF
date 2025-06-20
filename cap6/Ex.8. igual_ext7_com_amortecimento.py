import numpy as np
import matplotlib.pyplot as plt

## Parâmetros do sistema
k = 1.0          # Constante elástica das molas externas (N/m)
k_prime = 0.5    # Constante elástica da mola de acoplamento (N/m)
m = 1.0          # Massa de cada corpo (kg)
b = 0.05         # Coeficiente de amortecimento (kg/s)
F0 = 0.005       # Amplitude da força externa (N)
xA_eq = 1.0      # Posição de equilíbrio do corpo A (m)
xB_eq = 1.2      # Posição de equilíbrio do corpo B (m)

## Configurações da simulação
dt = 0.01        # Passo de tempo (s)
t_max_a = 150.0  # Tempo máximo para parte a) (s)
t_max_b = 200.0  # Tempo máximo para parte b) (s)
t_transient = 50.0 # Tempo para descartar transiente (s)

## Funções auxiliares
def accelerations(xA, xB, vA, vB, time, omega_d):
    """Calcula as acelerações dos corpos A e B"""
    # Termos de força elástica
    force_A = -k*(xA - xA_eq) - k_prime*((xA - xA_eq) - (xB - xB_eq))
    force_B = -k*(xB - xB_eq) - k_prime*((xB - xB_eq) - (xA - xA_eq))
    
    # Termos de amortecimento
    force_A -= b*vA
    force_B -= b*vB
    
    # Forçamento externo apenas no corpo A
    force_A += F0 * np.cos(omega_d * time)
    
    return force_A/m, force_B/m

def simulate(xA0, xB0, omega_d, t_max):
    """Simula o movimento dos corpos A e B"""
    n_steps = int(t_max / dt)
    t = np.linspace(0, t_max, n_steps)
    
    xA = np.zeros(n_steps)
    xB = np.zeros(n_steps)
    vA = np.zeros(n_steps)
    vB = np.zeros(n_steps)
    
    # Condições iniciais
    xA[0] = xA0
    xB[0] = xB0
    vA[0] = 0.0
    vB[0] = 0.0
    
    # Calcular primeira aceleração
    aA_prev, aB_prev = accelerations(xA[0], xB[0], vA[0], vB[0], t[0], omega_d)
    
    # Integração usando Verlet
    for i in range(1, n_steps):
        # Atualizar posições
        xA[i] = xA[i-1] + vA[i-1]*dt + 0.5*aA_prev*dt**2
        xB[i] = xB[i-1] + vB[i-1]*dt + 0.5*aB_prev*dt**2
        
        # Calcular novas acelerações
        aA_new, aB_new = accelerations(xA[i], xB[i], vA[i-1], vB[i-1], t[i], omega_d)
        
        # Atualizar velocidades
        vA[i] = vA[i-1] + 0.5*(aA_prev + aA_new)*dt
        vB[i] = vB[i-1] + 0.5*(aB_prev + aB_new)*dt
        
        # Preparar para próxima iteração
        aA_prev, aB_prev = aA_new, aB_new
    
    return t, xA, xB

## Parte a) Simulação para ωd = 1 rad/s
print("Executando simulação para ωd = 1 rad/s...")
omega_d_a = 1.0
xA0 = xA_eq + 0.05
xB0 = xB_eq + 0.05

t_a, xA_a, xB_a = simulate(xA0, xB0, omega_d_a, t_max_a)

# Plotar resultados para parte a)
plt.figure(figsize=(12, 6))
plt.plot(t_a, xA_a, label='Corpo A')
plt.plot(t_a, xB_a, label='Corpo B')
plt.title('Posição dos corpos A e B (ωd = 1 rad/s)')
plt.xlabel('Tempo (s)')
plt.ylabel('Posição (m)')
plt.legend()
plt.grid(True)
plt.show()

## Parte b) Análise de amplitude vs frequência de forçamento
print("Executando análise de resposta em frequência...")
omega_d_values = np.linspace(0, 2.5, 100)
amplitudes_A = np.zeros_like(omega_d_values)
amplitudes_B = np.zeros_like(omega_d_values)

n_transient = int(t_transient / dt)

for idx, omega_d in enumerate(omega_d_values):
    t_b, xA_b, xB_b = simulate(xA0, xB0, omega_d, t_max_b)
    
    # Calcular amplitude no regime estacionário
    xA_steady = xA_b[n_transient:]
    xB_steady = xB_b[n_transient:]
    
    amplitudes_A[idx] = np.std(xA_steady - np.mean(xA_steady))
    amplitudes_B[idx] = np.std(xB_steady - np.mean(xB_steady))

    

# Plotar curva de resposta em frequência
plt.figure(figsize=(12, 6))
plt.plot(omega_d_values, amplitudes_A, 'b-', label='Corpo A (forçado)')
plt.plot(omega_d_values, amplitudes_B, 'r-', label='Corpo B')
plt.title('Resposta em Amplitude vs Frequência de Forçamento')
plt.xlabel('Frequência de forçamento ωd (rad/s)')
plt.ylabel('Amplitude de oscilação (m)')
plt.legend()
plt.grid(True)

# Marcar frequências naturais teóricas
omega1 = np.sqrt(k/m)
omega2 = np.sqrt((k + 2*k_prime)/m)
plt.axvline(x=omega1, color='gray', linestyle='--', label=f'ω1 = {omega1:.2f} rad/s')
plt.axvline(x=omega2, color='black', linestyle='--', label=f'ω2 = {omega2:.2f} rad/s')
plt.legend()

plt.show()

## Frequências naturais teóricas
print("\nFrequências naturais teóricas do sistema:")
print(f"ω1 (modo simétrico) = {omega1:.3f} rad/s")
print(f"ω2 (modo antissimétrico) = {omega2:.3f} rad/s")