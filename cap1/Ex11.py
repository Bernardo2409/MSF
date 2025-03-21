import numpy as np
import matplotlib.pyplot as plt


x = np.array([1080, 1044, 1008, 972, 936, 900, 864, 828, 792, 756, 720, 540, 360, 180, 0])
y1 = np.array([0, 2.25, 5.25, 7.5, 8.75, 12, 13.75, 14.75, 15.5, 17, 17.5, 19.5, 18.5, 13, 0])
y2 = np.array([0, 3.25, 6.5, 7.75, 9.25, 12.25, 16, 15.25, 16, 17, 18.5, 20, 18.5, 13, 0])
y3 = np.array([0, 4.5, 6.5, 8.25, 9.5, 12.5, 16, 15.5, 16.6, 17.5, 18.5, 20.25, 19, 13, 0])
y4 = np.array([0, 6.5, 8.75, 9.25, 10.5, 14.75, 16.5, 17.5, 16.75, 19.25, 19, 20.5, 19, 13, 0])


plt.figure(figsize=(11,7))
plt.subplot(2, 2, 1)
plt.scatter(x, y1, label='y1')
plt.title('Plot 1: y1')

plt.subplot(2, 2, 2)
plt.scatter(x, y2, label='y2')
plt.title('Plot 2: y2')

plt.subplot(2, 2, 3)
plt.scatter(x, y3, label='y3')
plt.title('Plot 3: y3')

plt.subplot(2, 2, 4)
plt.scatter(x, y4, label='y4')
plt.title('Plot 4: y4')


coeffs1 = np.polyfit(x, y1, 2)
coeffs2 = np.polyfit(x, y2, 2)
coeffs3 = np.polyfit(x, y3, 2)
coeffs4 = np.polyfit(x, y4, 2)


poly1 = np.poly1d(coeffs1)
poly2 = np.poly1d(coeffs2)
poly3 = np.poly1d(coeffs3)
poly4 = np.poly1d(coeffs4)


x_fit = np.linspace(min(x), max(x), 500)


plt.subplot(2, 2, 1)
plt.plot(x_fit, poly1(x_fit), label='Polyfit y1', linestyle='--', color = 'red')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(x_fit, poly2(x_fit), label='Polyfit y2', linestyle='--')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(x_fit, poly3(x_fit), label='Polyfit y3', linestyle='--')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(x_fit, poly4(x_fit), label='Polyfit y4', linestyle='--')
plt.legend()



plt.show()



