import numpy as np
import math

class NN:
    def tan(self, x):
        return (2/ 1 + math.exp(-2) * x)-1

    def __init__(self) -> None:
        self.o1 = 0.0
        self.o2 = 0.0

        self.W = [[-1,1.5],[-1.5,0]]
        #self.W = [[-1.2,0.45],[-0.45,0]] ergebnisse von ü6
        self.bias1 = -3.37
        self.bias2 = 0.125

    def activate(self):
        O1Inputs = [self.W[0][0]*self.o1, self.W[0][1]*self.o2, self.bias1]
        O2Inputs = [self.W[1][0]*self.o1, self.bias2]

        newO1 = math.tanh(sum(O1Inputs))
        newO2 = math.tanh(sum(O2Inputs))

        print(newO1,newO2)

        self.o1, self.o2 = newO1,newO2
        

n = NN()
n.activate()
n.activate()
n.activate()
n.activate()
    
