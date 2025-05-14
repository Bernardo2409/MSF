import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

d = 0.1           
l = 1.0           
m = 0.3           
g = 9.8           
N = 5 #numero de esferas            
k = 10**7           

t_max = 5.0
dt = 0.0001
n_steps = int(t_max / dt)


fix_points_x = np.array([(i - N//2) * d for i in range(N)])
fix_points_y = np.zeros(N)


pos = np.zeros((N, 2, n_steps))  
vel = np.zeros((N, 2, n_steps))  


initial_angle = 1.2 
for i in range(N):
    if i == 3 or i == 4:  
        pos[i, :, 0] = [fix_points_x[i] + l * np.sin(initial_angle), -l * np.cos(initial_angle)]
    else:
        pos[i, :, 0] = [fix_points_x[i], -l]



def collision_force(dx, dy):
    dist = np.sqrt(dx**2 + dy**2)
    if dist < d:
        direction_x = dx / dist
        direction_y = dy / dist
        overlap = d - dist
        force = k * overlap
        return force * direction_x, force * direction_y
    # se nao ocorrer a colisão
    return 0, 0

for t in range(n_steps - 1):
    for i in range(N):
        ax, ay = 0, -g
        xi, yi = pos[i, :, t]
        
        dx = xi - fix_points_x[i]
        dy = yi - fix_points_y[i]
        dist = np.sqrt(dx**2 + dy**2)
        if dist > 0:
            v2 = vel[i, 0, t]**2 + vel[i, 1, t]**2
            tension = m * v2 / dist + m * g * (-dy) / dist
            ax += -tension * dx / dist
            ay += -tension * dy / dist
        
        for j in range(N):
            if i != j:
                dx = pos[i, 0, t] - pos[j, 0, t]
                dy = pos[i, 1, t] - pos[j, 1, t]
                fx, fy = collision_force(dx, dy)
                ax += fx / m
                ay += fy / m

        vel[i, 0, t+1] = vel[i, 0, t] + ax * dt
        vel[i, 1, t+1] = vel[i, 1, t] + ay * dt

        new_x = pos[i, 0, t] + vel[i, 0, t+1] * dt
        new_y = pos[i, 1, t] + vel[i, 1, t+1] * dt
        new_dx = new_x - fix_points_x[i]
        new_dy = new_y - fix_points_y[i]
        new_dist = np.sqrt(new_dx**2 + new_dy**2)
        if new_dist > 0:
            scale = l / new_dist
            pos[i, 0, t+1] = fix_points_x[i] + new_dx * scale
            pos[i, 1, t+1] = fix_points_y[i] + new_dy * scale

fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(-1.5 * l, 1.5 * l)
ax.set_ylim(-1.5 * l, 0.5 * l)
ax.grid()

lines = [ax.plot([], [], 'k-', lw=2)[0] for _ in range(N)]
circles = [Circle((0, 0), d/2, fc='blue', ec='black') for _ in range(N)]
for circ in circles:
    ax.add_patch(circ)


def update(frame):
    for i in range(N):
        x, y = pos[i, 0, frame], pos[i, 1, frame]
        circles[i].center = (x, y)
        lines[i].set_data([fix_points_x[i], x], [fix_points_y[i], y])
    return lines + circles

ani = FuncAnimation(fig, update, frames=range(0, n_steps, 50),
                    interval=20, blit=True)

plt.xlabel("Posição X (m)")
plt.ylabel("Posição Y (m)")
plt.show()