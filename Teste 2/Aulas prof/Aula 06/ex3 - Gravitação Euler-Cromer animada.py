from matplotlib.animation import FuncAnimation
from IPython.display import HTML

import numpy as np
import matplotlib.pyplot as plt

t0 = 0.0 # condição inicial, tempo [ano]
tf = 10.0 # limite do domínio, tempo final [ano]
dt = 0.001 # passo [ano]

v0 = 2.0 * np.pi # condição inicial, módulo da velocidade inicial [AU/ano]
x0 = 1.0 # condição inicial, coordenada x da posição inicial [AU]
G = 4.0 * np.pi ** 2 # constante de gravitação [AU^3 / M ano^2]

# inicializar domínio [ano]
t = np.arange(t0, tf, dt)

# inicializar solução, aceleração a 2D [AU / ano^2]
a = np.zeros([np.size(t),2 ])

# inicializar solução, velocidade [m / s]
v = np.zeros([np.size(t), 2])
v[0, :] = np.array([0, v0]) # velocidade [AU / ano] para t = 0 ano

# inicializar solução, posição [m]
r = np.zeros([np.size(t), 2])
r[0, :] = np.array([x0, 0.0]) # posição [AU] para t = 0 ano

for i in range(np.size(t) - 1):
    a[i, :] = - G * r[i, :] / np.linalg.norm(r[i, :]) ** 3 # aceleração instantânea
    v[i + 1, :] = v[i, :] + a[i, :] * dt
    r[i + 1, :] = r[i, :] + v[i + 1, :] * dt # Metodo de Euler-Cromer

fig = plt.figure()
ax = plt.axes(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5))

terra = ax.plot([], [], 'o')[0] # terra, posição inicial
def update(frame):
    # atualizar o plot da posição da Terra
    terra.set_xdata([r[frame, 0]])
    terra.set_ydata([r[frame, 1]])
    return terra
nframes = 1000
total_frames = np.size(t)
iframes = np.arange(0, total_frames, total_frames // nframes)
ani = FuncAnimation(fig=fig, func=update, frames=iframes, interval=100)
HTML(ani.to_jshtml())