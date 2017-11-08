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
from project4b2 import make_functions_of_T # input: (S, MC_cycles, T_max, T_step), returns: (energy_mean, magnetization_mean, C_V, chi,temperature, MC_cycles) 

T_min = 2.
T_max = 2.3
T_step = 6
temperature = np.linspace(T_min,T_max, num = T_step)

MC_cycles = 10000000
J = 1
pool = Pool(cpu_count())

S1 = np.ones((40,40)) #lattice with spins. All spins pointing up
S2 = np.ones((60,60))
S3 = np.ones((80,80))
S4 = np.ones((100,100))
mean_values = pool.starmap(make_functions_of_T,[(S1,MC_cycles,T_min,T_max,T_step),(S2,MC_cycles,T_min,T_max,T_step),(S3,MC_cycles,T_min,T_max,T_step),(S4,MC_cycles,T_min,T_max,T_step)])
#energy_mean, magnetization_mean, C_V, chi,temperature, MC_cycles = make_functions_of_T(S1,MC_cycles,T_min,T_max,T_step)

plt.plot(temperature, mean_values[0][0]/40**2, label = 'L = 40')
plt.plot(temperature, mean_values[1][0]/60**2, label = 'L = 60')
plt.plot(temperature, mean_values[2][0]/80**2, label = 'L = 80')
plt.plot(temperature, mean_values[3][0]/100**2, label = 'L = 100')
plt.legend()
plt.xlabel('temperature')
plt.ylabel('mean energy')
plt.title('Mean energy per particle as a function of temperature')
plt.show()

plt.plot(temperature, mean_values[0][1]/40**2, label = 'L = 40')
plt.plot(temperature, mean_values[1][1]/60**2, label = 'L = 60')
plt.plot(temperature, mean_values[2][1]/80**2, label = 'L = 80')
plt.plot(temperature, mean_values[3][1]/100**2, label = 'L = 100')
plt.legend()
plt.xlabel('temperature')
plt.ylabel('mean magnetization')
plt.title('Mean magnetization per particle as a function of temperature')
plt.show()

plt.plot(temperature, mean_values[0][2]/40**2, label = 'L = 40')
plt.plot(temperature, mean_values[1][2]/60**2, label = 'L = 60')
plt.plot(temperature, mean_values[2][2]/80**2, label = 'L = 80')
plt.plot(temperature, mean_values[3][2]/100**2, label = 'L = 100')
plt.title('Heat capcity per particle as a function of temperature')
plt.legend()
plt.xlabel('temperature')
plt.ylabel('C_V')
plt.show()

plt.plot(temperature, mean_values[0][3]/40**2, label = 'L = 40')
plt.plot(temperature, mean_values[1][3]/60**2, label = 'L = 60')
plt.plot(temperature, mean_values[2][3]/80**2, label = 'L = 80')
plt.plot(temperature, mean_values[3][3]/100**2, label = 'L = 100')
plt.legend()
plt.xlabel('temperature')
plt.ylabel('chi')
plt.title('Susceptibility per particle as a function of temperature')
plt.show()