import random
import gnnLayer as gnn
import numpy as np

sampleSize = 400*400


def createRandomSet(n):
    return [[random.random()*3-1,random.random()*3-1] for x in range(n)]

def inTarget(x,y):
    if (x**2 + y **2)**(1/2) < 1:
        return True
    else:
        return False


nn = gnn.NeuralNetwork()

set = createRandomSet(sampleSize)
#print(createRandomSet(10))

set = createRandomSet(sampleSize)
for x in set:
    a,b = x
    nn.activate(x)
    if inTarget(a,b):
        nn.backPropagation(0.01, [0.8])
    else:
        nn.backPropagation(0.01, [0])


nn.activate([0.75,0.75])
print(nn.activate([0.75,0.75]) , inTarget(0.75,0.75))
print(nn.activate([1,1]), inTarget(1,1))
print(nn.activate([0.5,0.5]), inTarget(0.5,0.5))
print(nn.activate([0,0]), inTarget(0,0))
print(nn.activate([-1,-1]), inTarget(-1,-1))
