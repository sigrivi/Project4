import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import linalg
import sys
import time
import random
from multiprocessing import Pool,cpu_count
from project4b2 import calculate_energy # input: (spin matrix, index, index), returns: (energy)
from project4b2 import monte_carlo # input: (spin matrix, temperature, number of MC cycles), returns: (energy_mean, energy_squared, magnetization_mean, magnetization_squared) (vectors)

pool = Pool(cpu_count())

dimension = 20
MC_cycles = 1000000
J = 1

S = np.ones((dimension,dimension)) #lattice with spins. All spins pointing up

S_random = S.copy() # lattice with spins. Random configuration
for j in range(dimension):
		for k in range(dimension):
			r = np.random.randint(2)
			S_random[j,k] = S_random[j,k]*(-1)**r


mean_values = pool.starmap(monte_carlo,[(S,1,MC_cycles),(S,2.4,MC_cycles)]) #run the simulation

energy1 = np. asarray(mean_values[0][0])/dimension**2 # mean energy per particle for T = 1
energy24 = np. asarray(mean_values[1][0])/dimension**2 # mean energy per prticle for T = 2.4
magnetization1 = np. asarray(mean_values[0][2])/dimension**2 # mean magnetization per particle for T = 1
magnetization24 = np. asarray(mean_values[1][2])/dimension**2 # mean magnetization per particle for T = 2.4

plt.plot(energy1, label = 'T = 1')
plt.xlabel('MC cycles')
plt.ylabel('mean energy')
plt.title('Mean energy per particle as a function MC cycles' )
plt.legend()
plt.savefig('energy_mean_MC_1',dpi=225)
plt.show()


plt.plot(energy24, label = 'T = 2.4')
plt.xlabel('MC cycles')
plt.ylabel('mean magnetization')
plt.title('Mean energy per particle as a function of MC cycles')
plt.legend()
plt.savefig('energy_mean_MC_2_4',dpi=225)
plt.show()

plt.plot(magnetization1, label = 'T = 1')
plt.xlabel('MC cycles')
plt.ylabel('mean magnetization')
plt.title('Mean magnetization per particle as a function MC cycles' )
plt.legend()
plt.savefig('magnetization_mean_MC_1',dpi=225)
plt.show()

plt.plot(magnetization24, label = 'T = 2.4')
plt.xlabel('MC cycles')
plt.ylabel('mean magnetization')
plt.title('Mean magnetization per particle as a function of MC cycles')
plt.legend()
plt.savefig('magnetization_mean_MC_2_4',dpi=225)
plt.show()
