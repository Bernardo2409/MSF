import numpy as np
import matplotlib.pyplot as plt

def calcular_trajetoria(instrucoes):

    #posições iniciais 
    x, y = 0, 0  
    angulo = 0   
    posicoes = [(x, y)]
    
    for ang, dist in instrucoes:
        angulo -= ang  #sentido horário
        rad = np.radians(angulo)  #grau --> radiano
        x += dist * np.cos(rad)
        y += dist * np.sin(rad)
        posicoes.append((x, y))
    
    return posicoes

def desenhar_trajetoria(posicoes):
    plt.figure(figsize=(8,8))
    
    for i in range(len(posicoes) - 1):

        # calcular para onde vai o robô
        x0, y0 = posicoes[i]
        x1, y1 = posicoes[i+1]
        deslcx = x1 - x0 
        deslcy = y1 - y0

        plt.arrow(x0, y0, deslcx, deslcy, head_width = 0.1, head_length = 0.1, length_includes_head = True, color = 'r')

    x_coords = [posicao[0] for posicao in posicoes ]
    y_coords = [posicao[1] for posicao in posicoes] 
    plt.scatter(x_coords, y_coords, color='blue')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Movimento/Trajetória do Robô')
    plt.grid()
    plt.show()


instrucoes = np.array([(45,3), (90,2), (45,3), (45,2), (90,3)])
trajetoria = calcular_trajetoria(instrucoes) # retorna array das posicoes
desenhar_trajetoria(trajetoria)

# Coordenadas finais
x_final, y_final = trajetoria[-1] #ultimo elemento da lista 
print("Coordenadas finais do robô: x:", x_final, "y:", y_final)

angulo = 0
for i in range(len(instrucoes) - 1 ):
    angulo += (instrucoes[i][0])
angulo_retorno = 360 - angulo
print(angulo_retorno)

distancia = np.sqrt(x_final**2 + y_final**2)

print("Instrução: (" , angulo_retorno, "," , distancia ,")" )


