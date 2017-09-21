'''
this file contains two classes, which are build_hmm for building hmm models 
and build_bkt for building bkt models
'''

import numpy as np
#from hmmtest import *


class build_hmm:
	def __init__(self,channels_):
		'''
		transition probs: square matrix
		initial probs: vector
		'''
		self.channel_keys = channels_  # observation channels (nodes)
		self.n_channels = len(self.channel_keys) # number of observation channels (nodes)
		self.length_of_sequences=0 # length of timestamp sequence 
		self.n_sequence=0 # number of observation sequence

	def set_model_parameters(self,obs,trans,init_probs, emission_probs,state_names = [] ):
		'''
		set 
		'''
		self.raw_observations = obs # multiple channels: marriage.seq, child.seq, affect.seq, 
		self.n_states = trans.shape[0] 
		self.transition = trans # a list of transition matrixes A1, A2, A3
		self.initial_probs = init_probs # equals to # of states
		self.raw_emission = emission_probs # list ( list (B1 channels), list( B2 channels), list( B3 channels))
		self.state_names  = state_names

		self.length_of_sequences= obs[self.channel_keys[0]].shape[0] # length of the timestamps for a given seuqnce
		self.n_sequence = obs[self.channel_keys[0]].shape[1] #number of sequences

		# convert the input data to a required data frame format
		self.transformed_emission=self.__transform_emissions(emission_probs)
		self.transformed_obs=self.__transform_obs(obs)
		

	

	def hidden_paths_viterbix(self):
		'''
		use viterbix algorithm to calculate the hidden path with highest prob
		'''
		#check whether there are multiple channels
		init_probs = np.log(self.init_probs)
		trans_probs = np.log(self.transition)

		obsArray = self.transformed_obs
		emissionArray = np.log(self.transformed_emission)

		# call viterbi
		viterbi_model = viterbi()
		viterbi_model.run_viterbi(trans_probs,emissionArray,init_probs,obsArray)
		q = viterbi_model.get_q()
		print("q is ...")
		print(q)

		logp= viterbi_model.get_logp()
		return q,logp


		

	def parameter_estimation(self):
		'''
		train HMM model and estimate parameters  
		'''

		# set parameters for EM training
		itermax = 10000
		tol = 1e-10
		trace = 0
		threads = 1

		self.__set_untrained_parameters()

		para_estimator = parameter_estimation()

		para_estimator.EM(self.untrained_transition,self.untrained_emission,self.untrained_init,self.transformed_obs,self.nSymbols,itermax,tol,trace,threads)
		
		self.transformed_emission= para_estimator.get_emission()
		self.transition = para_estimator.get_transition()
		self.sumlogLik = para_estimator.get_sumlogLik()
		self.initial_probs = para_estimator.get_init()

		print("================Parameter Estimation: EM====================")
		print("Total iterations: "+str(para_estimator.get_iterations()))
		print("sum+log lik: "+str(self.sumlogLik))
		print("emission_probs:")
		print(self.transformed_emission)
		print("transition probs:")
		print(self.transition)
		print("initial probs:")
		print(self.initial_probs)

		self.__decode_EM_emission_matrix(self.transformed_emission)

	def get_posterior_probs_old(self,obs_):
		'''
		obs_ should be a single sequence. 
		the function gets posterior probs for a single sequence
		'''
		init_probs = self.initial_probs
		trans_probs = self.transition

		emissionArray=self.transformed_emission
		obsArray=self.__transform_obs(obs_)

		print("obs array...")
		print(obsArray)
		print(obsArray.shape)
		print('------------------')
		
		fb = ForwardBackward()
		
		fb.run(trans_probs,emissionArray,init_probs,obsArray,False,1)
		forward_probs=fb.get_forward_probs()
		print("======forward probs...===")
		print(forward_probs)
		backward_probs=fb.get_backward_probs()
		print("===get backward probs ====")
		print(backward_probs)
		scales=fb.get_scales()
		
		posterior_probs= forward_probs * backward_probs / np.tile(scales,(np.sum(self.n_states),1,1))
		print('===============possterior probs====================')
		print(posterior_probs)
		print('.....................done.............................')

	def get_posterior_probs(self,obs_):
		'''
		obs_ should be a single sequence. 
		the function gets posterior probs for a single sequence
		'''
		from forward_backward import hmm_algorithms as hmm_algo
		init_probs = self.initial_probs
		trans_probs = self.transition

		emissionArray=self.transformed_emission
		obsArray=self.__transform_obs(obs_)

		
		fb = hmm_algo(trans_probs,emissionArray,init_probs,obsArray)
		alpha=fb.ForwardBackward()
		
		
	def __set_observations(self,obs_):
		'''
		reset raw observation data
		'''
		self.raw_observations = obs_
		self.length_of_sequences= obs_[self.channel_keys[0]].shape[0]
		self.n_sequence = obs_[self.channel_keys[0]].shape[1]

		self.transformed_obs=self.__transform_obs(obs_)

	def __set_untrained_parameters(self):
		'''
		set untrained parameters for EM algorithm
		'''
		self.untrained_transition = self.transition
		self.untrained_emission = self.transformed_emission
		self.untrained_init = self.initial_probs

	def __transform_obs(self,obs_):
		'''
		transform raw obs into a format that is compatiable with our model
		raw obervations is a dictionary with channel names as its keys
		'''

		self.n_sequence = obs_[self.channel_keys[0]].shape[1]
		self.length_of_sequences = obs_[self.channel_keys[0]].shape[0]
		obs  = np.zeros((len(self.channel_keys),self.length_of_sequences,self.n_sequence),order='F',dtype=np.uint32)
		for index in range(0,len(self.channel_keys),1) :
			value = obs_[self.channel_keys[index]]
			obs[index,:,:]=value
		return obs
		

	def __transform_emissions(self,emission_):
		'''
		transform raw obs into a format that is compatiable with our model
		raw emission matrix is a dictionary with channel names as its keys
		'''
		self.emission_dict = emission_
		self.max_n_symbols = np.max([ value.shape[1] for value in emission_.values()])
		self.n_states = np.max([ value.shape[0] for value in emission_.values()])
		emission = np.ones((self.n_states,self.max_n_symbols,len(emission_.keys())),order='F')
		
		self.nSymbols = []
		
		for index in range(0,len(self.channel_keys),1):
			value = emission_[self.channel_keys[index]]
			emission[:,0:value.shape[1],index]=value
			self.nSymbols.append(value.shape[1])
		
		self.nSymbols = np.array(self.nSymbols,dtype=np.uint32)
		return emission
		
			
	def __decode_EM_emission_matrix(self,e_mat):
		'''
		create a human readable dictionary for the transformed emission matrix 
		'''
		self.emission_dict={}
		print (e_mat.shape)
		for i in range(e_mat.shape[2]):
			print(i)
			self.emission_dict.update({self.channel_keys[i]:e_mat[:,:,i]})
		np.set_printoptions(formatter={'float_kind':'{:f}'.format})
		print ("emission dictionary")
		print(self.emission_dict)



class build_bkt(build_hmm):
	def __init__(self,channels_):

		'''standard bayesian knowledge tracing '''
		self.channel_keys = channels_
		self.channel_keys.append('task_node')
		self.n_channels = len(self.channel_keys)

		# prob of known 
		self.p_init= 0.2
		# transition: from unknown to known
		self.p_transit= 0.2
		# known skill -> task failure
		self.p_slip = 0.1
		# unknown skill -> task success
		self.p_guess = 0.2

		# init prob: [known, unknown]
		self.bkt_init_probs = np.array([self.p_init, 1-self.p_init],order='F')

		# transitaion matrix: [[known->known,known->unknown] , [unknown -> known, unknown -> unknown]]
		self.bkt_transition = np.array([[1.0,0.0],[self.p_transit,1-self.p_transit]],order ='F')
		
		# emission probs: []
		self.bkt_raw_emission = dict()

		# for the task node, the emission matrix is: [ [known->correct, known->incorrect], [unkown ->correct, unkown->incorrect] ]
		task_node_emission_matrix = np.array([[1-self.p_slip,self.p_slip],[self.p_guess,1-self.p_guess]],order='F')
		self.bkt_raw_emission.update({'task_node':task_node_emission_matrix})

		self.bkt_raw_observations = dict() # multiple channels: marriage.seq, child.seq, affect.seq, 
	

		
	def add_node_emission(self,node_name,emission_matrx):
		'''
		add emission matrix for a given observation node
		'''
		if not emission_matrx.shape[0] == 2:
			print("input "+node_name+"'s emission matrix does not have a right number of rows and does not match the number concept states")
		else:
			self.bkt_raw_emission.update({node_name:emission_matrx})

	def add_node_observation_data(self,node_name,obs_data):
		'''
		add raw observation data for a given node
		'''
		self.bkt_raw_observations.update({node_name:obs_data})

	def add_task_node_obs_data(self,obs_data):
		'''
		add the task node to the modal. task node: correctly answer the task, incorrectly answer the task
		'''
		self.bkt_raw_observations.update({'task_node':obs_data})
	
	def commit_bkt_model(self):
		'''
		after setting parameters for the bkt model, commit the model before runniiing any parameter estination, verterbi, poster probs
		'''
		self.set_model_parameters(self.bkt_raw_observations,self.bkt_transition,self.bkt_init_probs,self.bkt_raw_emission)
		

