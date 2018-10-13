import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-3, 3, 50)
x = 1
plt.figure()
plt.plot(0, 1, 'ks', color='red')
plt.xlim((0, 1))
plt.ylim((0, 1))
plt.xlabel('False positive rate, FPR')
plt.ylabel('True positive rate, TPR')
plt.show()