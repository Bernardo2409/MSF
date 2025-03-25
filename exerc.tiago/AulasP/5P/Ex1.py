import numpy as np
import matplotlib.pyplot as plt

x, y, theta = 0.0, 0.0, 0.0
positions = [[x, y, theta]]
plt.figure(figsize=(6.5, 4.5))
plt.axis([-5, 5, -5, 5])
plt.arrow(x,y,np.cos(-theta * np.pi / 180), np.sin(-theta * np.pi /180),width=0.01,head_width=0.1)

# Passo 1: (45,3)
theta += 45
x += 3 * np.cos(-theta * np.pi / 180)
y += 3 * np.sin(-theta * np.pi / 180)
positions.append([x, y, theta])
plt.arrow(x,y,np.cos(-theta * np.pi / 180), np.sin(-theta * np.pi /180),width=0.01,head_width=0.1)

# Passo 2: (90,2)
theta += 90
x += 2 * np.cos(-theta * np.pi / 180)
y += 2 * np.sin(-theta * np.pi / 180)
positions.append([x, y, theta])
plt.arrow(x,y,np.cos(-theta * np.pi / 180), np.sin(-theta * np.pi /180),width=0.01,head_width=0.1)

# Passo 3: (45,3)
theta += 45
x += 3 * np.cos(-theta * np.pi / 180)
y += 3 * np.sin(-theta * np.pi / 180)
positions.append([x, y, theta])
plt.arrow(x,y,np.cos(-theta * np.pi / 180), np.sin(-theta * np.pi /180),width=0.01,head_width=0.1)

# Passo 4: (45,2)
theta += 45
x += 2 * np.cos(-theta * np.pi / 180)
y += 2 * np.sin(-theta * np.pi / 180)
positions.append([x, y, theta])
plt.arrow(x,y,np.cos(-theta * np.pi / 180), np.sin(-theta * np.pi /180),width=0.01,head_width=0.1)

# Passo 5: (90,3)
theta += 90
x += 3 * np.cos(-theta * np.pi / 180)
y += 3 * np.sin(-theta * np.pi / 180)
positions.append([x, y, theta])
plt.arrow(x,y,np.cos(-theta * np.pi / 180), np.sin(-theta * np.pi /180),width=0.01,head_width=0.1)

theta += 45
x += 0-x * np.cos(-theta * np.pi / 180)
y += 0-y * np.sin(-theta * np.pi / 180)
positions.append([x, y, theta])
plt.arrow(x,y,np.cos(-theta * np.pi / 180), np.sin(-theta * np.pi /180),width=0.01,head_width=0.1)

# Convertendo para array NumPy
positions = np.array(positions)

# Plot da trajetória
plt.plot(positions[:, 0], positions[:, 1], 'g-o', label="Trajetória")
plt.legend()
plt.show()

c_final = positions[-1]
x_final = x
y_final = y
t_final = theta
print("Coordenadas Finais: {:.2f} {:.2f}".format(x_final,y_final))
print("Ângulo final: {}".format(theta))

