import numpy as np
import matplotlib.pyplot as plt

t0 = 0.0
tf = 0.5
dt = 0.001
r0 = np.array([0.0, 0.0, 23.8])
v0 = np.array([25.0, 5.0, -50.0])
w = 390.0
g = 9.8
R = 0.11
A = np.pi*R**2
m=0.45
dAr = 1.225
vT = 100*1000/3600
D = g/vT**2

t = np.arange(t0,tf, dt)
a = np.zeros([3, np.size(t)])
v = np.zeros([3, np.size(t)])

v[:,0] = v0

r=np.zeros([3, np.size(t)])
r[:,0] = r0

for i in range(np.size(t)-1):
    v_norm = np.linalg.norm(v[:,i])
    a[0,i] = -D*v[0,i]*v_norm + A*dAr*R*w*v[2,i]/(2*m)
    a[1,i] = -g - D*v[1,i]*v_norm
    a[2,i] = -D*v[2,i]*v_norm - A*dAr*R*w*v[0,i]/(2*m)
    v[:,i+1] = v[:,i] + a[:,i]*dt
    r[:,i+1] = r[:,i] + v[:,i]*dt

fig = plt.figure()
ax =fig.add_subplot(111, projection='3d')
ax.plot(r[0,:], r[2,:], r[1,:], 'r--')
ax.set_xlabel('X')
ax.set_ylabel('Z')
ax.set_zlabel('Y')
plt.show()

ixzero = np.size(r[0, r[0,:]>=0])
txzero = t[ixzero]
print("txzero = ", txzero, "s")
print("   x = ", r[0,ixzero], "m")
print("   y = ", r[1,ixzero], "m")
print("   z = ", r[2,ixzero], "m")

def golo(r, ixzero):
    if r[0, ixzero] < 0 and 0 < r[1, ixzero] < 2.4 and 0 < r[2, ixzero] < 7.3:
        return True
    else:
        return False

if golo(r, ixzero):
    print("Golo!")
else:
    print("Não é golo!")