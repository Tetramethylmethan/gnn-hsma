import numpy as np
import math
import matplotlib.pyplot as plt
import pygame

xmax = 400
ymax = 200
pygame.init()
screen = pygame.display.set_mode((xmax, ymax))

def main():

    image_size = 28 # width and length
    no_of_different_labels = 10 #  i.e. 0, 1, 2, 3, ..., 9
    image_pixels = image_size * image_size
    data_path = ""#"data/mnist/"
    #train_data = np.loadtxt(data_path + "mnist_train.csv", delimiter=",")
    test_data = np.loadtxt("A4\mnist.csv", delimiter=",") 
    test_data[:10]
    test_data[test_data==255]
    test_data.shape
    test_labels = np.asfarray(test_data[:, :1])
    lr = np.arange(no_of_different_labels)

    pattern = 0
    endlos = True
    while endlos:
            for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                            endlos = False

            pressed = pygame.key.get_pressed()

            in_vec = np.zeros(28*28)
            maxPool = np.random.rand(28*28)

            # --- draw input activation ---
            ii=0
            screen.fill((0, 0, 0))
            for y in range(0,28):
                for x in range(0,28):
                    in_vec[ii] = val = test_data[pattern,ii]/255.0              
                    ii=ii+1
                    pygame.draw.rect(screen, (255*val, 0, 0), pygame.Rect(x*5, y*5, 5, 5))
 
            # --- calculate maxPooling 2x2 -> 1x1
            ii=0
            for y in range(0,28,2):
                for x in range(0,28,2):
                    maxPool[ii] = np.max(in_vec[y*28+x:y*28+x+2])
                           
                    #maxPool[ii] = in_vec[y*28+x] # nur die linke obere Ecke von den 4 Werten
                    ii=ii+1                                # aus denen das Maximum rausgesucht werden muss
                    

            # --- draw maxPoolingLayer ---
            jj=0
            for y in range(0,14):
                for x in range(0,14):
                    val = maxPool[jj]             
                    jj=jj+1
                    pygame.draw.rect(screen, (0,255*val, 0), pygame.Rect(200+x*5, (y+10)*5, 5, 5))

            pygame.time.delay(1000)    # may be reduced in training time     
            pattern=pattern+1
            pygame.display.flip()

# --------------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------------
main()
