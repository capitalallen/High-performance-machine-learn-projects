from turtle import color
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(0, (200-5)/30, 1)
y = 30*x+5
x = np.append(x, [(200-5)/30,10,12,14])
y = np.append(y, [200,200,200,200])
x_benchmark = [0.25,7/32,0.5,1/8,0.7]
y_benchmark=[]
for i in x_benchmark:
    y_benchmark.append(30*i+5)

plt.plot(x, y, '-r')
# print(y_benchmark)
plt.scatter(x_benchmark,y_benchmark,color='blue')
# plt.title('Graph of y=30x+5')
plt.axvline(x=6.5)
plt.xlabel('Arith Intensity', color='#1C2833')
plt.ylabel('Actual Flops', color='#1C2833')

plt.legend(loc='upper left')
plt.grid()
plt.show()
