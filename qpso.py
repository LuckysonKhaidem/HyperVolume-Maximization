import numpy as np

def random_uniform(l,u):
	return max(10e-8, np.random.uniform(l,u))

class QPSO:
	def __init__(self, objective, swarm_size, n_dimensions, lb,ub,g, max_iterations = 500):
		self.objective = objective
		self.swarm_size = swarm_size
		self.n_dimensions = n_dimensions
		self.ub = ub
		self.lb = lb
		self.pos = np.zeros((swarm_size, n_dimensions))
		self.g = g 
		self.max_iterations = max_iterations


		if(self.n_dimensions != len(lb) or self.n_dimensions != len(ub)):
			raise Exception

	def __initialize_swarm_particles(self):

		for i in range(self.swarm_size):
			for j in range(self.n_dimensions):
				self.pos[i][j] = np.random.uniform(self.lb[j], self.ub[j])
		
		self.lbest = self.pos.copy()

		func_values = [self.objective(x) for x in self.pos]

		ix_best = np.argmin(func_values)

		self.gbest = self.lbest[ix_best].copy()

	def __update_positions(self):
		for i in range(self.swarm_size):
			for j in range(self.n_dimensions):
				psi_1 = random_uniform(0,1)
				psi_2 = random_uniform(0,1)
				P = (psi_1*self.lbest[i][j] + psi_2 * self.gbest[j])/(psi_1 + psi_2)
				u = random_uniform(0,1)
				# self.g = random_uniform(0.5,0.99)
				L = 1/self.g * np.abs(self.pos[i][j] - P)
				chi = (self.ub[j] - self.lb[j]) / 1000.0
				if random_uniform(0,1) > 0.5:
					self.pos[i][j] = P - chi*L*np.log(1/u)
				else:
					self.pos[i][j] = P + chi*L*np.log(1/u)

	def __update_best_positions(self):
		func_values = []
		for i in range(self.swarm_size):
			f1 = self.objective(self.pos[i])
			f2 = self.objective(self.lbest[i])
			
			if(f1 < f2):
				self.lbest[i] = self.pos[i].copy()
				func_values.append(f1)
			else:
				func_values.append(f2)

		ix_best = np.argmin(func_values)
		self.gbest = self.lbest[ix_best].copy()


	def run(self):
		self.__initialize_swarm_particles()
		for i in range(self.max_iterations):
			print(self.objective(self.gbest))
			self.__update_positions()
			self.__update_best_positions()

		return self.gbest
