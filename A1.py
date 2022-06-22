
import matplotlib.pyplot as plt
import numpy as np

dt= 0.01
startingX = [-7,-0.2,8]
steps = 1000

def nextXByEulerFunc(x):
    return x + dt*(x-(x**3))

for i in startingX:
    Xpair = [0 for i in range(steps)]
    Ypair = [i*0.01 for i in range( steps)]
    for j in range(steps):
        Xpair[j] = i
       # print(i,nextXByEulerFunc(i))
        i = nextXByEulerFunc(i)
    print("----")
    plt.plot(Ypair, Xpair)
    plt.ylabel(i)
    plt.show()

