from matplotlib import pyplot as plt
import numpy as np
with open('x.csv', 'r') as f:
    y1 = f.readlines()
with open('y.csv', 'r') as fl:
    y2 = fl.readlines()


x = range(10,301)
plt.plot(x,y1[: -1])
plt.show()
