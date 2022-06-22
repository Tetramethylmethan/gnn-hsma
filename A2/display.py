import numpy as np
import matplotlib.pyplot as plt
import pygame
import random
import gnnLayer as gnn

sampleSize = 200
xmax = 200
ymax = 200
pygame.init()
screen = pygame.display.set_mode((xmax, ymax))


def inTarget(x,y):
        if (x**2 + y **2)**(1/2) < 1:
                return True
        else:
                return False


def createRandomSet(n):
    return [[random.randint(0,1),random.randint(0,1)] for x in range(n)]


def main():
        nn = gnn.NeuralNetwork()
        
        set = createRandomSet(sampleSize)
        '''
        for x in set:
                a,b = x 
                nn.activate([a,b])
                if inTarget(a,b):
                        nn.backPropagation([a,b], 0.8)
                else:
                        nn.backPropagation([a,b], 0)
        '''
        endlos = True
        while endlos:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                            endlos = False

                pressed = pygame.key.get_pressed()

               
                screen.fill((0, 0, 0))
                for x in range(100):
                        a =random.randint(0,200)
                        a =1-(a/(xmax/2))
                        
                        b = random.randint(0,200)
                        b =1-(b/(xmax/2))
                        
                        output=nn.activate([abs(a),abs(b)])
                


                        if inTarget(a,b):
                                nn.backPropagation(0.5, [0.8])
                                #output = 0.8
                        else:
                                nn.backPropagation(0.5, [0]) 
                                #output = 0  

                for y in range(0,ymax):
                        for x in range(0,xmax):
                                a = 1-(x/(xmax/2)) 
                                b = 1-(y/(ymax/2)) 
                                
                                
                                output=nn.activate([abs(a),abs(b)])
                                
                                             
                                pygame.draw.rect(screen, (255*output, 0, 0), pygame.Rect(x, y, 1, 1))
                        #pygame.display.flip()
                pygame.display.flip()
                print("done")

# --------------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------------
main()