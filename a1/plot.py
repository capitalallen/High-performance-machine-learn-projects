import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(0,(200-5)/30,5)
y = 30*x+5
x=np.append(x,[8])
y=np.append(y,[200])
plt.plot(x, y, '-r', label='y=30x+5')
plt.title('Graph of y=30x+5')
plt.axvline(x=6.5)
plt.xlabel('x', color='#1C2833')
plt.ylabel('y', color='#1C2833')

plt.legend(loc='upper left')
plt.grid()
plt.show()
195