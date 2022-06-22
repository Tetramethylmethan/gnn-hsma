import numpy as np
import random

def transferFunction(x):
    return 1.0 / (1.0 + np.exp(-x))

def derivativeTransferFunction(x):
    x = transferFunction(x)
    return x * (1 - x)

class NeuralNetwork:
    
    def __init__(self) -> None:

        self.inputSize = 2
        self.hiddenSize = 4
        self.outputSize = 1

        self.biasHidden = np.zeros((self.hiddenSize,1))
        self.wghtHidden = np.random.rand(self.hiddenSize,self.inputSize)

        self.biasOutput = np.zeros((self.outputSize,1))
        self.wghtOutput = np.random.rand(self.outputSize,self.hiddenSize)

    def activate(self,input):
        self.input = input

        self.outMtrxHidden = input * self.wghtHidden + self.biasHidden
        self.actMtrxHidden = [transferFunction(x.sum()) for x in self.outMtrxHidden]
        self.outMtrxOutput = self.actMtrxHidden @ self.wghtOutput.T + self.biasOutput
        self.actMtrsOutput = transferFunction(self.outMtrxOutput)
    
        return self.actMtrsOutput
        

    def backPropagation(self,learningRate,target):

        epsilonOutput = (self.actMtrsOutput - target) * derivativeTransferFunction(self.outMtrxOutput)
        self.wghtOutput -= learningRate * epsilonOutput * self.actMtrxHidden
        
        epsilonHidden = (epsilonOutput @ self.wghtOutput) * [derivativeTransferFunction(x.sum()) for x in self.outMtrxHidden] # ek * wkj * f'(sum(wji *oi))
        epsilonHidden = np.array([[x.sum() for x in epsilonHidden]]) # sumk (ek * wkj * f'(sum(wji *oi)))
        self.wghtHidden -= learningRate * (epsilonHidden.T * self.input)



