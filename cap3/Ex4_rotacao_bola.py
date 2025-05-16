import numpy as np
import matplotlib.pyplot as plt

# Parâmetros do problema (MANTIDOS)
t0 = 0.0
tf = 5.0  # Aumentado para garantir alcance
dt = 0.0001  # Reduzido para maior precisão
r0 = np.array([-10.0, 1.0, 0.0])
v0 = 130.0 * 1000 / 3600  
theta = np.deg2rad(10)
m = 0.057
R = 0.067/2
g = 9.8
vT = 100 * 1000 / 3600
dAr = 1.225
A = np.pi * R ** 2
D = (m * g) / (vT**2)  #(coeficiente de arrasto linear)

def f(w):
    t = np.arange(t0, tf, dt)
    n_steps = len(t)
    
    a = np.zeros([3, n_steps])
    v = np.zeros([3, n_steps])
    r = np.zeros([3, n_steps])
    
    v[0,0] = v0 * np.cos(theta)
    v[1,0] = v0 * np.sin(theta)
    v[2,0] = 0.0
    r[:,0] = r0
    
    for i in range(n_steps - 1):
        v_norm = np.linalg.norm(v[:,i])
        
        # Força de arrasto CORRIGIDA (modelo linear)
        F_drag = -D * v_norm * v[:,i]
        
        # Força Magnus CORRIGIDA
        F_magnus = 0.5 * dAr * A * R * np.cross(w, v[:,i])
        
        a[0,i] = F_drag[0]/m + F_magnus[0]/m
        a[1,i] = -g + F_drag[1]/m + F_magnus[1]/m
        a[2,i] = F_drag[2]/m + F_magnus[2]/m
        
        v[:,i+1] = v[:,i] + a[:,i]*dt
        r[:,i+1] = r[:,i] + v[:,i]*dt
        
        if r[1,i+1] <= 0:
            n_steps = i+2
            break

    t = t[:n_steps]
    r = r[:, :n_steps]
    v = v[:, :n_steps]
    
    # Cálculos CORRIGIDOS para coincidir com a tabela
    altura_max = np.max(r[1,:])
    alcance = r[0,-1]  
    
    print(f"Altura máxima: {altura_max:.2f} m")
    print(f"Alcance: {alcance:.2f} m\n")
    
    # Plot (opcional)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(r[0,:], r[2,:], r[1,:])
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    plt.show()

# Casos com rotação (MANTIDOS)
w0 = np.array([0.0, 0.0, 0.0])    # a) Sem rotação
w1 = np.array([0.0, 0.0, 100.0])  # b) Rotação positiva
w2 = np.array([0.0, 0.0, -100.0]) # c) Rotação negativa

print("Caso a) Rotação nula:")
f(w0)  # Deve retornar ~2.70 m e ~27.21 m

print("Caso b) Rotação +100 rad/s:")
f(w1)  # Deve retornar ~3.62 m e ~39.29 m

print("Caso c) Rotação -100 rad/s:")
f(w2)  # Deve retornar ~2.24 m e ~19.62 m