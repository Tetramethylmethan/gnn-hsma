import numpy as np
import random

def transferFunction(x):
        return 1.0 / (1.0 + np.exp(-x))

class RBM:
    
    def __init__(self, size = (28*28)+10) -> None:
        self.wght = np.random.uniform(-1, 1, size = (size,size))
        #self.wght =np.ones(shape=(size,size))
        
        
    def activate(self,v0):    
        hOut = (self.wght @ v0)

        #------- sig
        h = transferFunction(hOut)   

        #------- prop
        #s = np.random.uniform(0, 1, hOut.shape)
        #h = np.where(hOut > s, 1, 0)

        return h


    def reActivate(self,v0):    
        hOut = (self.wght.T @ v0)

        #------- sig
        h = transferFunction(hOut)   

        #------- prop
        #s = np.random.uniform(0, 1, hOut.shape)
        #h = np.where(hOut > s, 1, 0)

        return h


    def cD(self,learningRate, v0, h, v1):

        self.wght += learningRate * (np.outer(h , v0-v1))   # Eta * (h * v0 - h * v1) linear /sig.
        return self.wght
        
    def train(self,lerningrate, v0):
        
        #v0[0] = 1 # bias
        h = self.activate(v0) # aktivierung
        #h[0] = 1 # bias

        v1 = self.reActivate(h) # rückaktivierung
        self.cD(lerningrate, v0,h,v1) # anpassen von wght
        return h,v1
    
    def run(self, v0):
        v0[0] = 1 # bias
        h = self.activate(v0) # aktivierung
        h[0] = 1 # bias
        v1 = self.rekonstructA(h) # rückaktivierung

        return h, v1

    def checkOrthogonal(self):
        e = self.wght @ self.wght.T
        return e
    




