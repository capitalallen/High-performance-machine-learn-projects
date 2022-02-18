from turtle import color
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(0, (200-5)/30, 1)
y = 30*x+5
x = np.append(x, [(200-5)/30,10,12,14])
y = np.append(y, [200,200,200,200])
x_benchmark = [114849.81/45957.924, 34324022.346/13729608.9, 452196.96/452196.95, 135032967.03 /
               13503296.703, 6.33/2.533, 3.18/1.271, 0.0069/0.00278, 0.0069/0.00277, 7.64/3.064, 6.156/24.0625]

y_benchmark=[114.81,34.3346,45.296,13.503,6.33,3.18,0.0069,0.0069, 7.64,6.156]

for i in range(len(x_benchmark)):
    print((x_benchmark[i],y_benchmark[i]))
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
