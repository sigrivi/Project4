import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import linalg
import sys
import time
import random

J = 1

def calculate_energy(S,j,k): # calculates the energy contribution of element S[jk]
	dimension=S.shape[0]
	energy = -J*(S[j,k]*S[j-1,k] + S[j,k]*S[(j+1)%dimension,k] + S[j,k]*S[j,k-1] + S[j,k]*S[j,(k+1)%dimension]) 
	return(energy)

def monte_carlo(S,T,MC_cycles):
	dimension = S.shape[0] 
	energy_sum = 0
	spin_sum = 0
	energy = 0
	spin = 0 #the total spin of the system
	
	energy_mean = np.zeros(MC_cycles)
	energy_squared = np.zeros(MC_cycles)
	magnetization_mean = np.zeros(MC_cycles)
	magnetization_squared = np.zeros(MC_cycles)

	w = {-8:math.exp(-(-8)/T),-4:math.exp(-(-4)/T), 0:math.exp(-(0)/T), 4:math.exp(-(4)/T),8:math.exp(-(8)/T)} #list with delta_energies and calculated w's 
	RN = np.random.random(size = MC_cycles) #random numbers for the MC-cycle
	
	# calculate the total energy of the system:
	for j in range(dimension):
		for k in range(dimension):
			energy += 0.5*calculate_energy(S,j,k)
			spin += S[j,k]
	
	energy_mean[0] = energy
	magnetization_mean[0] = spin
	energy_squared[0] = energy**2
	magnetization_squared[0] = spin**2

	for i in range(1,MC_cycles):
		j = random.randint(0,dimension-1)
		k = random.randint(0,dimension-1)
		old_energy = calculate_energy(S,j,k)
		S[j,k] = -S[j,k] #flip spin j,k
		temp_energy = calculate_energy(S,j,k)
		delta_energy=temp_energy-old_energy				
		
		if temp_energy > old_energy:
			if RN[i] >= w[delta_energy]: # Metropolis algorithm: 
				S[j,k] = -S[j,k] # Undo the flip/reject the move
				
			else:
				energy += delta_energy # accept move
				spin += 2*S[j,k]
		else :
			energy += delta_energy # 100% acceptance when temp_energy < old_energy
			spin += 2*S[j,k]
	
		energy_sum += energy 
		spin_sum += np.abs(spin)
		energy_mean[i] = energy_sum/i
		magnetization_mean[i] = spin_sum/i
		energy_squared[i] = energy_squared[i-1] + energy**2
		magnetization_squared[i] = magnetization_squared[i-1]+spin**2
	
	for i in range(1,MC_cycles):	
		energy_squared[i] = energy_squared[i]/i
		magnetization_squared[i] = magnetization_squared[i]/i

	return(energy_mean, energy_squared, magnetization_mean, magnetization_squared)

def make_functions_of_T(S, MC_cycles, T_max, T_step):
	temperature = np.linspace(1,T_max, num = T_step)
	i = 0
	energy_mean = np.zeros((T_step,MC_cycles))
	energy_squared = np.zeros((T_step,MC_cycles)) 
	magnetization_mean = np.zeros((T_step,MC_cycles))
	magnetization_squared = np.zeros((T_step,MC_cycles))
	
	for T in temperature:
		energy_mean[i], energy_squared[i], magnetization_mean[i], magnetization_squared[i] = monte_carlo(S,T,MC_cycles)
		np.save('data/state_{0}_{1}.npz'.format(T,dimension),S)
		i += 1

	np.save('data/energy_mean_{0}.npy'.format(dimension),energy_mean)
	np.save('data/energy_squared_{0}.npy'.format(dimension),energy_squared)
	np.save('data/magnetization_mean_{0}.npy'.format(dimension),magnetization_mean)
	np.save('data/magnetization_squared_{0}.npy'.format(dimension),magnetization_squared)
	np.save('data/temperature_{0}.npy'.format(dimension),temperature)

	C_V = energy_squared[:,MC_cycles-1]-energy_mean[:,MC_cycles-1]**2
	chi = magnetization_squared[:,MC_cycles-1]-magnetization_mean[:,MC_cycles-1]**2
	return(energy_mean[:,MC_cycles-1], magnetization_mean[:,MC_cycles-1], C_V, chi, temperature, MC_cycles)

def plot_functions_of_T(energy_mean, magnetization_mean, C_V, chi,temperature, MC_cycles):
	plt.plot(temperature ,energy_mean)
	plt.xlabel('temperature')
	plt.ylabel('mean energy')
	plt.title('Mean energy as a function of temperature after %d MC_cycles' %MC_cycles)
	plt.savefig('energy_mean',dpi=225)
	plt.show()

	plt.plot(temperature ,magnetization_mean)
	plt.xlabel('temperature')
	plt.ylabel('mean magnetization')
	plt.title('Mean magnetization as a function of temperature after %d MC_cycles' %MC_cycles)
	plt.savefig('magnetization_mean',dpi=225)
	plt.show()

	plt.plot(temperature ,C_V)
	plt.xlabel('temperature')
	plt.ylabel('C_V')
	plt.title('Heat capacity as a function of temperature after %d MC_cycles' %MC_cycles)
	plt.savefig('C_V',dpi=225)
	plt.show()

	plt.plot(temperature ,chi)
	plt.xlabel('temperature')
	plt.ylabel('chi')
	plt.title('Susceptibility as a function of temperature after %d MC_cycles' %MC_cycles)
	plt.savefig('chi',dpi=225)
	plt.show()

#plot_functions_of_T(*make_functions_of_T(S, 1000, 2, 10))


#C_V = energy_squared - energy_mean**2
#chi = magnetization_squared- magnetization_mean**2


#print(energy_mean[MC_cycles-1])
#print(energy_squared[MC_cycles-1])
#print(C_V[MC_cycles-1])
#print(magnetization_mean[MC_cycles-1])
#print(magnetization_squared[MC_cycles-1])
#print(chi[MC_cycles-1])
#energy_target = np.ones(MC_cycles)*(-7.98392834374676)
#plt.plot(C_V[1000:MC_cycles-1])
#plt.plot(energy_target, 'g')


if __name__ == "__main__":
	# parameters:
	dimension = 20
	MC_cycles = 100000
	T = 1
	S = np.ones((dimension,dimension)) #lattice with spins. Can use the previously generated as a starting point for the next temeprature
	#S[0,1]=S[1,0] =-1
	#S[0,0]=S[1,1] =-1
	energy_mean, energy_squared, magnetization_mean, magnetization_squared = monte_carlo(S,1,MC_cycles)
	plt.plot(energy_mean)
	plt.show()
	plt.plot(magnetization_mean)
	plt.show()
	#print(energy_mean,type(energy_mean),energy_squared,type(energy_squared))

	pass