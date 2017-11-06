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

# parameters:
T = 1
dimension = 20
MC_cycles = 100000
J = 1

S = np.ones((dimension,dimension)) #lattice with spins. All spins pointing up

S_random = S.copy() # lattice with spins. Random configuration
for j in range(dimension):
		for k in range(dimension):
			r = np.random.randint(2)
			S_random[j,k] = S_random[j,k]*(-1)**r


energy_mean, energy_squared, magnetization_mean, magnetization_squared = monte_carlo(S,1,MC_cycles)
magnetization_mean = magnetization_mean/dimension**2
#plt.plot(energy_mean)
#plt.show()

plt.plot(energy_mean)
plt.xlabel('MC_cycles')
plt.ylabel('mean energy')
plt.title('Mean energy as a function MC_cycles' )
plt.savefig('energy_mean__MC_1',dpi=225)
plt.show()

plt.plot(magnetization_mean)
plt.xlabel('MC_cycles')
plt.ylabel('mean magnetization')
plt.title('Mean magnetization as a function of MC_cycles')
plt.savefig('magnetization_mean_MC_1',dpi=225)
plt.show()
