# imports
import numpy as np
import matplotlib.pyplot as plt


def main():
    # Condições Iniciais 

    # Posição
    x0 = 0
    y0 = 0

    # ângulo de lançamento
    alpha = np.deg2rad(10)

    # Velocidade
    v0 = 100 / 3.6
    v0x = v0 * np.cos(alpha)
    v0y = v0 * np.sin(alpha)

    # aceleração gravítica
    g = 9.8

    # massa da bola de ténis
    m = 0.057

    # Velocidade Terminal
    vt = 100 / 3.6

    # Cálculo do D para a Resistência do ar
    D = g/vt**2

    # Passo temporal e Número de Etapas para o Método de Euler #
    dt = 0.001
    t0 = 0
    tf = 1.0
    n = np.int((tf - t0)/dt)

    # Preparação dos Arrays para o Método de Euler :
    t = np.zeros(n+1)

    ax = np.zeros(n+1)
    ay = np.zeros(n+1)
    ayrar = np.zeros(n+1)

    vx = np.zeros(n+1)
    vy = np.zeros(n+1)

    x = np.zeros(n+1)
    y = np.zeros(n+1)

    # array das forças não conservativas
    fx = np.zeros(n+1)
    fy = np.zeros(n+1)
    Em0 = m*g*y0 + 0.5*m*v0**2
    Emecanica = np.zeros(t.size)
    Emecanica[0] = Em0

    # Inicializar os arrays com os valores iniciais (t=0) :
    vx[0] = v0x
    vy[0] = v0y

    x[0] = x0
    y[0] = y0

    # Método de Euler :
    for i in range(n):
        t[i + 1] = t[i] + dt  # alterar o instante

        vv = np.sqrt(vx[i]**2 + vy[i]**2) # intensidade do vetor velocidade

        ay[i] = -( g + (D*vy[i]*vv)) # alterar a aceleração eixo y
        ax[i] = -D*vx[i]*vv # alterar a aceleração eixo x
        ayrar[i] = -(D*vy[i]*vv) # alterar a aceleraçao no eixo y só com a resistência do ar

        vy[i+1] = vy[i] + ay[i]*dt # alterar a velocidade eixo x
        vx[i+1] = vx[i] + ax[i]*dt # alterar a velocidade eixo y

        x[i + 1] = x[i] + vx[i] * dt  # alterar a posição eixo x
        y[i + 1] = y[i] + vy[i] * dt  # alterar a posição eixo y

        fx[i] = m*ax[i]*vx[i]   # alterar as forças não conservativas eixo x
        fy[i] = m*ayrar[i]*vy[i]    # alterar as forças não conservativas eixo y
        Emecanica[i+1] = m*g*y[i+1] + 0.5*m*vv**2

    for i in range(n):
        if (0 - dt < t[i] < 0 + dt) or (0.4 - dt < t[i + 1] < 0.4 + dt) or (0.8 - dt < t[i + 1] < 0.8 + dt):
            # Calcular a Energia Mecânica (Em = Ec + Ep) :
            vv = np.sqrt(vx[i]**2 + vy[i]**2) # instensidade do vetor velocidade em cada instante

            Ep = m*g*y[i+1] # Energia Potencial
            Ec = 0.5*m*(vv**2) # Energia Cinética

            Em = Ec + Ep # Energia Mecânica

            print("A Energia Mecânica no instante {:.1f}s é {:.2f}J".format(t[i], Em))

            # Calcular o trabalho realizado pela força da resistência do ar (Aproximação Trapezoidal) :
            Wx = dt*((fx[0] + fx[i+1])*0.5 + np.sum(fx[1:i]))
            Wy = dt*((fy[0] + fy[i+1])*0.5 + np.sum(fy[1:i]))
            print("O trabalho realizado pela resistência do ar foi {:.2f}J.".format(Wx+Wy))

    plt.plot(t, Emecanica, label="Energia Mecânica")
    plt.xlabel("Tempo")
    plt.ylabel("Energia Mecanica")
    
    
    plt.grid()
    plt.legend()
    plt.show()

main()