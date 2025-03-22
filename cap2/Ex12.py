import math

# Constantes
g = 9.81  # Aceleração gravitacional (m/s^2)
rho_0 = 1.225  # Densidade do ar ao nível do solo (kg/m^3)
h_inicial = 1000  # Altura inicial (m)
v_terminal_livre = 60.0  # Velocidade terminal em queda livre (m/s)
v_terminal_paraquedas = 5.0  # Velocidade terminal com paraquedas aberto (m/s)

# Função para calcular o tempo de queda com velocidade terminal constante
def tempo_de_queda(h, v_terminal):
    return h / v_terminal

# a) Tempo e velocidade com paraquedas fechado
tempo_livre = tempo_de_queda(h_inicial, v_terminal_livre)
velocidade_livre_kmh = v_terminal_livre * 3.6  # Converter para km/h
print(f"Tempo de queda com paraquedas fechado: {tempo_livre:.1f} s")
print(f"Velocidade ao atingir o solo: {velocidade_livre_kmh:.0f} km/h")

# b) Tempo com paraquedas aberto desde o início
tempo_paraquedas = tempo_de_queda(h_inicial, v_terminal_paraquedas)
velocidade_paraquedas_kmh = v_terminal_paraquedas * 3.6  # Converter para km/h
print(f"Tempo de queda com paraquedas aberto: {tempo_paraquedas / 60:.1f} minutos")
print(f"Velocidade ao atingir o solo: {velocidade_paraquedas_kmh:.0f} km/h")

# c) Paraquedas aberto após 20 segundos
tempo_livre_20s = 20  # Tempo em queda livre
distancia_livre = v_terminal_livre * tempo_livre_20s  # Distância percorrida em queda livre
h_restante = h_inicial - distancia_livre  # Altura restante
tempo_restante = tempo_de_queda(h_restante, v_terminal_paraquedas)
tempo_total = tempo_livre_20s + tempo_restante
print(f"Tempo total com paraquedas aberto após 20 s: {tempo_total:.1f} s")
print(f"Velocidade ao atingir o solo: {velocidade_paraquedas_kmh:.0f} km/h")

# d) Considerando a densidade do ar variável
# Função para calcular a densidade do ar em função da altura (h em metros)
def densidade_ar(h):
    return rho_0 * math.exp(-0.1378 * h / 1000)

# Função para calcular a velocidade terminal em função da altura
def velocidade_terminal(h):
    rho = densidade_ar(h)
    return v_terminal_livre * math.sqrt(rho / rho_0)

# Função para calcular o tempo de queda com densidade variável usando o método de Euler
def tempo_de_queda_variavel(h_inicial, delta_t=0.1):
    h = h_inicial  # Altura inicial
    tempo = 0  # Tempo inicial

    while h > 0:
        v_terminal = velocidade_terminal(h)  # Velocidade terminal na altura atual
        dh = -v_terminal * delta_t  # Variação da altura
        h += dh  # Atualiza a altura
        tempo += delta_t  # Incrementa o tempo

        # Evita valores negativos de altura
        if h < 0:
            break

    return tempo

# d) Tempo de queda considerando densidade variável
tempo_total_variavel = tempo_de_queda_variavel(h_inicial)
print(f"Tempo total de queda com densidade variável: {tempo_total_variavel:.2f} s")