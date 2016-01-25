import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

startTime = time.time();

particleSize = 10;
position = [[0 for x in range(3)] for x in range(particleSize)];
velocity = [[0 for x in range(3)] for x in range(particleSize)];
gbest = [0 for x in range(3)];
pbest = [[0 for x in range(3)] for x in range(particleSize)];

c1 = 0.5;
c2 = 0.5;

def compute_fitness(ind):
    #return position[ind][0]*position[ind][0] + position[ind][1]*position[ind][1];
    #return (1-position[ind][0])**2 + (position[ind][1] - position[ind][0]*position[ind][0])**2;
    return (position[ind][0]**2 + position[ind][1] - 11)**2 + (position[ind][0] + position[ind][1]**2 - 7)**2;

 
def initialize_particles():
    # random numbers
    for i in range(particleSize):
        for j in range(2):
            position[i][j] = np.random.uniform(-5,5);
            velocity[i][j] = 0;
        position[i][2] = compute_fitness(i);
        pbest[i] = position[i][0:1] + position[i][1:2] + position[i][2:3];
    for k in range(3):
        gbest[k] = 0;   
            
def find_gbest():
    position2 = sorted(position, key=lambda x: x[2], reverse=True); 
    gbest[0] = position2[0][0];
    gbest[1] = position2[0][1];
    gbest[2] = position2[0][2];
   
       
def update_velocity(indi, indj):
    velocity[indi][indj] = velocity[indi][indj] + c1 * (gbest[indj] - position[indi][indj]) + c2 * (pbest[indi][indj] - position[indi][indj]);
    if abs(velocity[indi][indj]) > 3:
        velocity[indi][indj] = velocity[indi][indj] / 2;
    
       
def update_position():
    for i in range(particleSize):
        for j in range(2):
            update_velocity(i,j);
            position[i][j] = position[i][j] + velocity[i][j];        
        
        if position[i][0] > 5:           
            position[i][0] = 5;
        elif position[i][0] < -5:            
            position[i][0] = -5;        
        if position[i][1] > 5:            
            position[i][1] = 5;
        elif position[i][1] < -5:            
            position[i][1] = -5;
        position[i][2] = compute_fitness(i);
        if position[i][2] > pbest[i][2]:
            pbest[i][0] = position[i][0];
            pbest[i][1] = position[i][1];
            pbest[i][2] = position[i][2];
    
    
    
#initialize_particles();
#find_gbest();
#update_position(); 
                          
iterationCount = 2000;                    
genSize = 150  ;           
xaxis = [i for i in range(genSize)];
bsf = [[0 for x in range(iterationCount)] for x in range(genSize)];

for i in range(iterationCount):                                                    
    initialize_particles();
    find_gbest();      
    for generation in range(genSize):         
        update_position();    
        find_gbest();        
        bsf[generation][i] = gbest[2];


np_bsf = np.array(bsf);
avg_bsf = np_bsf.mean(axis=1)        
plt.plot(xaxis,avg_bsf);
#plt.ylabel('Avg Best-so-Far');
#plt.xlabel('Generations');
#plt.show(); 
print time.time() - startTime;