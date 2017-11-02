import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import linalg
import sys
import time
import random

temperature = 3  # T=1 is equivalent to  (k_B * T[in Kelvin])/J[in Joules]
J = 1
spins  = np.ones(4)

energy = 0
energy_sum = 0
energy_squared_sum = 0
spin_sum = 0
spin_squared_sum = 0
MC_cycles = 100000

w = {-16:math.exp(-(-16)/temperature),-8:math.exp(-(-8)/temperature), 0:math.exp(-(0)/temperature), 8:math.exp(-(8)/temperature),16:math.exp(-(16)/temperature)}

Z = 12 + 2*math.exp(8/temperature) + 2*math.exp(-8/temperature)
E = 2*8*J/Z * (- math.exp(8/temperature) + math.exp(-8/temperature))
E_squared = 2*8**2*J**2/Z * (math.exp(8/temperature) + math.exp(-8/temperature))
M = 0
M_squared = 2*4**2 *(1+(math.exp(8/temperature)))/Z
print(E)
print(E_squared)
print(M_squared)
T_max = 4
energy_mean = np.zeros(T_max)
magnetization_mean = np.zeros(T_max)
temperature = [1,2,3,4]
for T in temperature:
	energy_sum = 0
	spin_sum = 0
	w = {-16:math.exp(-(-16)/T),-8:math.exp(-(-8)/T), 0:math.exp(-(0)/T), 8:math.exp(-(8)/T),16:math.exp(-(16)/T)}

	for i in range(MC_cycles):
		RN = random.randint(0,len(spins)-1)
		spins[RN] = -spins[RN]	#flips one random spin

		temp_energy = -J*(spins[0]*spins[1] + spins[1]*spins[0] + spins[0]*spins[2] + spins[2]*spins[0] + spins[2]*spins[3] + spins[3]*spins[2] + spins[1]*spins[3] + spins[3]*spins[1])
		if temp_energy > energy:
			RN2 = np.random.random()
			delta_energy = temp_energy-energy
			if RN2 >= w[delta_energy]:
				spins[RN] = -spins[RN] # Undo the flip/reject the move
			else:
				energy = temp_energy		
		else :
			energy = temp_energy
			
		energy_sum += energy 				# mean energy
		energy_squared_sum += energy**2

		spin_sum += abs(sum(spins)) 		# mean magnetization 				
		spin_squared_sum += sum(spins)**2
	energy_mean[T-1] = energy_sum/MC_cycles
	magnetization_mean[T-1] =  spin_sum/MC_cycles
plt.plot(temperature,energy_mean)
plt.show()
plt.plot(temperature,magnetization_mean)
plt.show()
#print ( energy_sum/MC_cycles)
#print ( energy_squared_sum/MC_cycles)
#print( spin_sum/MC_cycles)
#print( spin_squared_sum/MC_cycles)