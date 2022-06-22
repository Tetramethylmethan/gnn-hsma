import numpy as np
import math
import matplotlib.pyplot as plt
import pygame
import rbm

xmax = 600
ymax = 200
pygame.init()
screen = pygame.display.set_mode((xmax, ymax))

def main():

    image_size = 28 # width and length
    no_of_different_labels = 10 #  i.e. 0, 1, 2, 3, ..., 9
    image_pixels = image_size * image_size
    data_path = ""#"data/mnist/"
    #train_data = np.loadtxt(data_path + "mnist_train.csv", delimiter=",")
    test_data = np.loadtxt("A3\mnist.csv", delimiter=",") 
    test_data[:10]
    test_data[test_data==255]
    test_data.shape
    test_labels = np.asfarray(test_data[:, :1])
    lr = np.arange(no_of_different_labels)

    # transform labels into one hot representation
    test_labels_one_hot = (lr==test_labels).astype(np.float)

    # we don't want zeroes and ones in the labels neither:
    test_labels_one_hot[test_labels_one_hot==0] = 0.01
    test_labels_one_hot[test_labels_one_hot==1] = 0.99



    # create RBM
    RBM = rbm.RBM()

    pattern = 0
    endlos = True
    while endlos:
            for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                            endlos = False

            pressed = pygame.key.get_pressed()

            in_vec = np.zeros(28*28+10)
            hid_vec = np.random.rand(28*28+10)
            out_vec = np.random.rand(28*28+10)

            

            # --- draw input activation ---
            ii=0
            screen.fill((0, 0, 0))
            for y in range(0,28):
                for x in range(0,28):
                        in_vec[ii] = val = test_data[pattern,ii]/255.0   
                        ii=ii+1
                        pygame.draw.rect(screen, (255*val, 0, 0), pygame.Rect(x*5, y*5, 5, 5))
            for x in range(0,10):
                    in_vec[ii] = val = test_labels_one_hot[pattern,x]               
                    ii=ii+1
                    pygame.draw.rect(screen, (255*val, 0, 0), pygame.Rect(x*5, 28*5, 5, 5))    

            #  insert activation here using in_vec[] as input vector
            #hid_vec = RBM.activate(in_vec)
            
            hid_vec,out_vec = RBM.train(0.1,in_vec)

            # --- draw hidden activation ---
            jj=0
            for y in range(0,28):
                for x in range(0,28):
                    val = hid_vec[jj]
                    #print(val)            
                    jj=jj+1
                    pygame.draw.rect(screen, (0,255*val, 0), pygame.Rect(200+x*5, y*5, 5, 5))
            for x in range(0,10):
                    val = hid_vec[jj]              
                    jj=jj+1
                    pygame.draw.rect(screen, (0,255*val, 0), pygame.Rect(200+x*5, 28*5, 5, 5)) 

            #  insert back-activation here
            #out_vec = RBM.reconstruct()

             # --- draw output activation ---
            kk=0
            for y in range(0,28):
                for x in range(0,28):
                    val = out_vec[kk]             
                    kk=kk+1
                    #print(val)
                    pygame.draw.rect(screen, (0,0,255*val), pygame.Rect(400+x*5, y*5, 5, 5))
            for x in range(0,10):
                    val = out_vec[kk]              
                    kk=kk+1
                    pygame.draw.rect(screen, (0,0,255*val), pygame.Rect(400+x*5, 28*5, 5, 5)) 

            # insert learning here
            if pattern %100 == 0:
                 pygame.time.delay(1000)    # may be reduced in training time
            

        
                
            pattern=pattern+1
            pygame.display.flip()

# --------------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------------
main()
