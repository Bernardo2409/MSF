import numpy as np
import matplotlib.pyplot as plt

t0 = 0.0
tf = 0.5
dt = 0.001

r0 = np.array([23.8, 0, 0])
v0 = np.array([-50, 25, 5])
w = 390.0

g = 9.8
m = 0.45
R = 0.11
A = np.pi*R**2
rho = 1.225

# inicializar dominio
t = np.arange(t0, tf, dt)

# inicializar solucao, aceleracao
a = np.zeros([np.size(t),3])
# inicializar solucao, velocidade
v = np.zeros([np.size(t),3])
v[0] = v0
# inicializar solucao, posicao
r = np.zeros([np.size(t),3])
r[0] = r0
# calcular pontos
for i in range(np.size(t)-1):
    a[i,0] = - A * rho * R * w * v[i,1] / (2 * m)
    a[i,1] = A * rho * R * w * v[i,0] / (2 * m)
    a[i,2] = - g
    v[i+1] = v[i] + a[i] * dt
    r[i+1] = r[i] + v[i+1] * dt
    
w*=0.6
# inicializar solucao, aceleracao
a1 = np.zeros([np.size(t),3])
# inicializar solucao, velocidade
v1 = np.zeros([np.size(t),3])
v1[0] = v0
# inicializar solucao, posicao
r1 = np.zeros([np.size(t),3])
r1[0] = r0
# calcular pontos
for i in range(np.size(t)-1):
    a1[i,0] = - A * rho * R * w * v1[i,1] / (2 * m)
    a1[i,1] = A * rho * R * w * v1[i,0] / (2 * m)
    a1[i,2] = - g
    v1[i+1] = v1[i] + a1[i] * dt
    r1[i+1] = r1[i] + v1[i+1] * dt
    
larguraDoCampo = 48
comprimentoDoCampo = 70

# ax = plt.axes(projection='3d')
# ax.plot3D(larguraDoCampo/2, 0, 0, 'ko')
# ax.plot3D(-larguraDoCampo/2/8, comprimentoDoCampo/2, 30, 'ko')
# ax.plot3D(r[:,0], r[:,1], r[:,2], 'pink')
# ax.plot3D(r1[:,0], r1[:,1], r1[:,2], 'r:')
# ax.set_xlabel("x")
# ax.set_ylabel("y")

# # Desenhar linha central
# ax.plot3D([0, 0], [0, comprimentoDoCampo/2], [0, 0], 'g-')

# # Desenhar linha de fundo do campo
# ax.plot3D([0, larguraDoCampo/2], [0, 0], [0, 0], 'g-')

# # Desenhar baliza
# ab = 2.44
# lb = 7.32
# ax.plot3D([-lb/2, lb/2], [0, 0], [0, 0], 'b-')  # Largura da baliza
# ax.plot3D([-lb/2, lb/2], [0, 0], [ab, ab], 'b-')  # Altura da baliza
# ax.plot3D([-lb/2, -lb/2], [0, 0], [0, ab], 'b-')  # Lado esquerdo
# ax.plot3D([lb/2, lb/2], [0, 0], [0, ab], 'b-')  # Lado direito
# ax.plot3D(0, 0, ab/2, 'r.')  # Centro da baliza
plt.figure(figsize=(8,8))
ax = plt.axes(projection='3d')
#criar eixos 3D
#desenhar a baliza
goalx = [0,0,0,0]
goaly = [0,2.4,2.4,0]
goalz = [-3.66,-3.66,3.66,3.66]
ax.plot3D(goalx,goalz,goaly, 'k') #trajetória da bola
ax.plot3D(r[:,1][r[:,0]>=0], -r[:,0][r[:,0]>=0],r[:,2][r[:,0]>=0], 'r')
#ajustar eixos
ax.set_xlim3d(0, 5)
ax.set_ylim3d(
-25, 5)
ax.set_zlim3d(0, 5)
ax.set_box_aspect((2,6,2))
ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_zlabel('y')
plt.show()
