import numpy as np

class hmm_algorithms():
	def __init__(self,transition,emission,initial,obs):
		self.transition = transition
		self.emission = emission
		self.initial = initial
		self.obs = obs
		
		# alpha is for forward algorithm. beta is for backward algorithm
		# obs.shape[0]: number of observation channels (e.g., affect node, task node)
		# obs.shape[1]: # of timestamps for a given data sequence (e.g., t0,t1,t2...)
		# obs.shape[2]: number of data sequences (data instances) (e.g., person1, person2,person3)
		self.alpha = np.zeros((self.initial.shape[0],self.obs.shape[1],self.obs.shape[2]))
		self.beta = np.ones((self.initial.shape[0],self.obs.shape[1],self.obs.shape[2]))

	def ForwardBackward(self):
		final_alpha=self.forward_algo()
		final_beta = self.backward_algo()

		print("-----------final alpha")
		print(final_alpha)

		print("-----------final beta")
		print(final_beta)

		print("----------posterior probs")
		posterior_probs =  self.__scaling(final_alpha*final_beta)
		print(posterior_probs)
		
	def backward_algo(self):
		for k in range(0,self.obs.shape[2]):
			# k refers to data sequence

			# calculate beta value at t
			for t in range(self.obs.shape[1]-2,-1,-1):
				# t refers to the timestamp for a given data sequnce k

				# get beta value at t + 1
				tmp_beta = self.beta[:,t+1,k]

				# russel + norvig: 15.9
				# multiple beta at t + 1 with the emission prob at t  for each of the observation channel
				for r in range(0,self.obs.shape[0]):
					# r refers to observation channel (e.g., affect, task)
					obs_value = self.obs[r,t+1,k]
					tmp_beta *= self.emission[:,obs_value,r]

				# tmp_beta at this point is: emission_probs(x at t + 1) * beta at t+1
				# get beta at t
				self.beta[:,t,k] = np.dot(self.transition,tmp_beta) 
				
				print('beta for each t '+str(t))
				print(self.beta[:,t,k])

		return self.beta

	def forward_algo(self):
		# forward algorithm equation:
		# alpha at t (x at t) = p( emission prob at t | x at t) sum( p(transition from x t-1 to x_t) * alpha at t -1 (x at t-1))
		
		# need a constant scaling factor here
		for k in range(0,self.obs.shape[2]):
			# k refers to number of data instances
			# set the intial value of alpha_t0 (x_t0) to be the initial prob
			self.alpha[:,0,k] = self.initial
			

			# get intial alpha_t0(x_t0)
			''' 
			example: if we have two observation channel (affect, task)
			r = 2 
			alpha_t0 = P_initial * P(affect_t0 | hidden state t0) * P(task_t0 | hidden state t0)
			'''
			for r in range(0,self.obs.shape[0]):
				# r refers to number of observation channels (nodes)
				obs_value = self.obs[r,0,k]

				# alpha_t0 *= p(observation channel at t0 | hidden state x at t0)
				# emission probability for the first obervation value for observation channel r
				self.alpha[:,0,k] *=self.emission[:,obs_value,r]

			self.alpha[:,0,k] = self.__scaling(self.alpha[:,0,k])

			# calculate alpha at t using alpha at t-1
			for t in range(1,self.obs.shape[1]):
				# t refers to the timestamps in a given data sequence (instance)
				
				# formular: P (x at t | x at t-1) * alpha at t-1
				self.alpha[:,t,k] = np.dot(np.transpose(self.transition) ,self.alpha[:,t-1,k]) 

				# multiple the value above with emission probability for each observation channel
				# to get alpha at t
				for r in range(0,self.obs.shape[0]):
					obs_value = self.obs[r,t,k]
					self.alpha[:,t,k] *= self.emission[:,obs_value,r]
				
				self.alpha[:,t,k] = self.__scaling(self.alpha[:,t,k])

				print("alpha for t: "+str(t))
				print(self.alpha[:,t,k])

		return self.alpha

	def __scaling(self,arr):
		total = sum(arr)
		return [ i/total for i in arr]



		
