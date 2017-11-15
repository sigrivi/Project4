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

def metropolis(S,RN,w, energy, spin):
	j = random.randint(0,dimension-1)
	k = random.randint(0,dimension-1)
	old_energy = calculate_energy(S,j,k)
	S[j,k] = -S[j,k] #flip spin j,k
	temp_energy = calculate_energy(S,j,k)
	delta_energy = temp_energy-old_energy				
	
	if temp_energy > old_energy:
		if RN >= w[delta_energy]: # Metropolis algorithm: 
			S[j,k] = -S[j,k] # Undo the flip/reject the move
				
		else:
			energy += delta_energy # accept move
			spin += 2*S[j,k]
	else :
		energy += delta_energy # 100% acceptance when temp_energy < old_energy
		spin += 2*S[j,k]
	return(S, energy, spin)

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
	energy_of_state = np.zeros(MC_cycles)

	w = {-8:math.exp(-(-8)/T),-4:math.exp(-(-4)/T), 0:math.exp(-(0)/T), 4:math.exp(-(4)/T),8:math.exp(-(8)/T)} #list with delta_energies and calculated w's 
	RN = np.random.random(size = MC_cycles) #random numbers for the MC-cycle
	
	# calculate the total energy of the system:
	for j in range(dimension):
		for k in range(dimension):
			energy += 0.5*calculate_energy(S,j,k)
			spin += S[j,k]
	
	energy_mean[0] = energy
	magnetization_mean[0] = spin
	energy_of_state[0] = energy
	energy_squared[0] = energy**2
	magnetization_squared[0] = spin**2


	for i in range(1,MC_cycles):
		# proposes one spint to flip. Either accepts or rejects the move. Updates energy and spin of the system:
		rn = RN[i]
		S, energy, spin = metropolis(S ,rn ,w , energy, spin) 
		
		# adds values to the solution vectors:
		energy_of_state[i] = energy # stores the energy of each cycle
		energy_sum += energy # adds the enegy of this cycle to the energy of the previous one
		spin_sum += np.abs(spin) # adds the magnetizatio of this cycle to the magnetization of the previous one
		energy_mean[i] = energy_sum/i
		magnetization_mean[i] = spin_sum/i
		energy_squared[i] = energy_squared[i-1] + energy**2
		magnetization_squared[i] = magnetization_squared[i-1]+spin**2
	
	for i in range(1,MC_cycles):	# calculates the mean energy squared and the mean magnetization squared
		energy_squared[i] = energy_squared[i]/i
		magnetization_squared[i] = magnetization_squared[i]/i

	return(energy_mean, energy_squared, magnetization_mean, magnetization_squared, energy_of_state)

def make_functions_of_T(S, MC_cycles, T_min, T_max, T_step):
	temperature = np.linspace(T_min,T_max, num = T_step)
	i = 0
	energy_mean = np.zeros((T_step,MC_cycles))
	energy_squared = np.zeros((T_step,MC_cycles)) 
	magnetization_mean = np.zeros((T_step,MC_cycles))
	magnetization_squared = np.zeros((T_step,MC_cycles))
	energy_of_state = np.zeros((T_step,MC_cycles))
	C_V = np.zeros((T_step,MC_cycles))
	chi = np.zeros((T_step,MC_cycles))
	dimension = S.shape[0]
	for T in temperature:
		energy_mean[i], energy_squared[i], magnetization_mean[i], magnetization_squared[i], energy_of_state[i] = monte_carlo(S,T,MC_cycles)
		C_V[i] = (energy_squared[i] - energy_mean[i]**2)/T**2
		chi[i] = (magnetization_squared[i]- magnetization_mean[i]**2)/T
		np.save('data/state_{0}_{1}.npz'.format(T,dimension),S)
		i += 1

	np.save('data/energy_mean_{0}.npy'.format(dimension),energy_mean)
	np.save('data/energy_squared_{0}.npy'.format(dimension),energy_squared)
	np.save('data/magnetization_mean_{0}.npy'.format(dimension),magnetization_mean)
	np.save('data/magnetization_squared_{0}.npy'.format(dimension),magnetization_squared)
	np.save('data/temperature_{0}.npy'.format(dimension),temperature)

	return(energy_mean[:,MC_cycles-1], magnetization_mean[:,MC_cycles-1], C_V[:,MC_cycles-1], chi[:,MC_cycles-1], temperature, MC_cycles)

def plot_functions_of_T(energy_mean, magnetization_mean, C_V, chi,temperature, MC_cycles):
	E_exact, C_V_exact, M_exact, chi_exact = exact_expressions(temperature)
	
	plt.plot(temperature ,energy_mean,label ="Monte Carlo ( %d cycles)" %MC_cycles)
	plt.plot(temperature, E_exact, label = "Exact")
	plt.xlabel('temperature')
	plt.ylabel('mean energy')
	plt.title('Mean energy as a function of temperature ')
	plt.legend()
	plt.savefig('energy_mean',dpi=225)
	plt.show()

	plt.plot(temperature ,magnetization_mean, label ="Monte Carlo ( %d cycles)" %MC_cycles)
	plt.plot(temperature, M_exact, label = "Exact")
	plt.xlabel('temperature')
	plt.ylabel('mean magnetization')
	plt.title('Mean magnetization as a function of temperature')
	plt.legend()
	plt.savefig('magnetization_mean',dpi=225)
	plt.show()

	plt.plot(temperature ,C_V,label ="Monte Carlo ( %d cycles)" %MC_cycles)
	plt.plot(temperature, C_V_exact, label = "Exact")
	plt.xlabel('temperature')
	plt.ylabel('C_V')
	plt.title('Heat capacity as a function of temperature')
	plt.legend()
	plt.savefig('C_V',dpi=225)
	plt.show()

	plt.plot(temperature ,chi,label ="Monte Carlo ( %d cycles)" %MC_cycles)
	plt.plot(temperature, chi_exact, label = "Exact")
	plt.xlabel('temperature')
	plt.ylabel('chi')
	plt.title('Susceptibility as a function of temperature')
	plt.legend()
	plt.savefig('chi',dpi=225)
	plt.show()

def exact_expressions(temperature):
	#temperature = np.linspace(1,T_max, num = T_step)
	J = 1
	beta = np.zeros(len(temperature))
	Z = np.zeros(len(temperature))
	E =np.zeros(len(temperature))
	E_squared = np.zeros(len(temperature))
	C_V = np.zeros(len(temperature))
	M = np.zeros(len(temperature))
	M_squared = np.zeros(len(temperature))
	chi = np.zeros(len(temperature))
	i=0
	for T in temperature:
	
		beta[i] = 1/(T*J)
		Z[i] = 12 + 2*math.exp(8/T) + 2*math.exp(-8/T)
		E[i] = 2*8*J/Z[i] * (- math.exp(8/T) + math.exp(-8/T))
		E_squared[i] = 2*8**2*J**2/Z[i] * (math.exp(8/T) + math.exp(-8/T))
		C_V[i] = beta[i]*(E_squared[i] - E[i]**2)/T
		M[i] = 4*2*(math.exp(8/T)+2)/Z[i]
		M_squared[i] = 2*4**2 *(1+(math.exp(8/T)))/Z[i]
		chi[i] = beta[i]*(M_squared[i] - M[i]**2)
		i+=1
	return(E, C_V, M, chi)


if __name__ == "__main__":
	# parameters:
	dimension = 2
	MC_cycles = 1000000
	T = 1.0
	S = np.ones((dimension,dimension)) #lattice with spins. Can use the previously generated as a starting point for the next temeprature
	#S[0,1]=S[1,0] =-1
	#S[0,0]=S[1,1] =-1

# To make the plot of mean energy as a functon of MC cycles
	#energy_mean, energy_squared, magnetization_mean, magnetization_squared, energy_of_state = monte_carlo(S,T,MC_cycles)
	#energy_target = np.ones(MC_cycles)*(-7.98392834374676)
	#plt.plot(energy_mean)
	#plt.plot(energy_target)
	#plt.xlabel('MC cycles')
	#plt.ylabel('Mean energy')
	#plt.title('Mean energy as a functon of MC cycles')
	#plt.savefig('1000000MCcycles',dpi=225)
	#plt.show()

# To make the plots E(T), M(T), C_V(T), chi(T)
	#plot_functions_of_T(*make_functions_of_T(S, 1000000,1, 3, 30))

# To explore the excaxt expressions
	#temperature = np.linspace(1,5, num = 40)
	#E, C_V, M, chi = exact_expressions(temperature)
	#plt.plot(temperature,C_V, 'g',temperature, chi, 'r')
	#plt.show()

	pass