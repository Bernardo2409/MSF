import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**3/4

a,b= 0, 2
I_exato = 1

def trapz(n):
    x = np.linspace(a, b, n+1)
    h = (b-a)/n
    dx = (b-a)/n
    y = f(x)
    return dx*(np.sum(y)-(y[0]+y[-1])/2)

ns = np.array([1,2,4,8,16,32,64,128, 256])
dxs = (b-a)/ns

resultados = np.array([trapz(n) for n in ns])
erros = np.abs(resultados - I_exato)

print("n\tδx\t\tErro\t\tErro/δx²")
for n, dx, erro in zip(ns, dxs, erros):
    print(f"{n}\t{dx:.6f}\t{erro:.8f}\t{erro / dx**2:.8f}")

# Gráficos
plt.figure(figsize=(12, 5))

# Gráfico 1: Erro vs δx
plt.subplot(1, 2, 1)
plt.loglog(dxs, erros, 'bo-', label="Erro")
plt.xlabel(r"$\delta x$", fontsize=12)  # Use raw string
plt.ylabel("Erro Absoluto", fontsize=12)
plt.title(r"Erro vs Tamanho do Passo ($\delta x$)", fontsize=14)  # Use raw string
plt.grid(True, which="both", linestyle="--")

# Gráfico 2: Proporcionalidade ao δx²
plt.subplot(1, 2, 2)
plt.loglog(dxs, erros / dxs**2, 'ro-', label=r"Erro/$\delta x^2$")
plt.xlabel(r"$\delta x$", fontsize=12)  # Use raw string
plt.ylabel(r"Erro / $\delta x^2$", fontsize=12)  # Use raw string
plt.title(r"Verificação da Proporcionalidade ($\delta x^2$)", fontsize=14)  # Use raw string
plt.grid(True, which="both", linestyle="--")

plt.tight_layout()
plt.show()