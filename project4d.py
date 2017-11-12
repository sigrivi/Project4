import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import linalg
import sys
import time
import random
from multiprocessing import Pool,cpu_count
from project4b2 import calculate_energy # input: (spin matrix, index, index), returns: (energy)
from project4b2 import monte_carlo # input: (spin matrix, temperature, number of MC cycles), returns: (energy_mean, energy_squared, magnetization_mean, magnetization_squared, state_energy) (vectors)

dimension = 20
MC_cycles = 10000000
J = 1

S1 = np.ones((dimension,dimension)) #lattice with spins. All spins pointing up
S2 = np.ones((dimension,dimension)) #lattice with spins. All spins pointing up

pool = Pool(cpu_count())
mean_values = pool.starmap(monte_carlo,[(S1,1,MC_cycles),(S2,2.4,MC_cycles)]) #calculating <E> , <E**2>, <M>, <M**2> and the energy of each state 

# mean values and variance:
#(mean_values[0][1][MC_cycles-1]) is the last element of a vector(of length MC_cycles) of mean energies when the temperature is 1. 
sigma_E_1 = mean_values[0][1][MC_cycles-1] - mean_values[0][0][MC_cycles-1]**2 # variance when T=1.
print('energy mean',mean_values[0][0][MC_cycles-1]) 
print('sigma_E_1',sigma_E_1)

sigma_E_24 = mean_values[1][1][MC_cycles-1] - mean_values[1][0][MC_cycles-1]**2 #variance when T=2.4
print('energy mean',mean_values[1][0][MC_cycles-1]) 
print('sigma_E_2_4',sigma_E_24) 

#for the histograms, I discard the firts 10%
plt.hist(mean_values[0][4][MC_cycles//10:MC_cycles],normed=1)
plt.xlabel('E')
plt.ylabel('P(E)')
plt.title('Probability distribution of the energy. T = 1')
plt.savefig('prob_dist_1',dpi=225)
plt.show()


plt.hist(mean_values[1][4][MC_cycles//10:MC_cycles],normed=1)
plt.xlabel('E')
plt.ylabel('P(E)')
plt.title('Probability distribution of the energy. T = 2.4')
plt.savefig('prob_dist_2_4',dpi=225)
plt.show()

