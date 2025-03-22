import numpy as np

def euler_falling(h, v_terminal, dt=0.01):
    g, t, v, y = 9.8, 0, 0, h
    while y > 0:
        v += g * (1 - v / v_terminal) * dt
        y -= v * dt
        t += dt
    return t

# Dados
h = 5  # altura inicial (m)
v_terminal_tennis = 100 / 3.6  # m/s
v_terminal_badminton = 6.80  # m/s

# Cálculo
t_tennis = euler_falling(h, v_terminal_tennis)
t_badminton = euler_falling(h, v_terminal_badminton)

print(f"Tempo de queda da bola de ténis: {t_tennis:.2f} s")
print(f"Tempo de queda do volante de badmington: {t_badminton:.2f} s")
print(f"Diferença de tempo: {abs(t_tennis - t_badminton):.2f} s")