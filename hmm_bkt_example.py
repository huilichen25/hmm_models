from hmm_bkt_models_final import *
import numpy as np

def dummy_hmm_test():
	'''
	the dummy example comes from russel and norvig's book
	'''
	em_umbrella = np.array([[0.9,0.2],
		[0.2,0.9]],order='F')

	# observation node: umbrella
	# 0 = 
	umbrella_arr = np.array([[0],[0]])

	obs ={ "umbrella":umbrella_arr}
	emission_probs ={ "umbrella":em_umbrella }

	trans = np.array([[0.7,0.3],[0.3,0.7]])

	init_probs = np.array([0.5,0.5])

	test =build_hmm(["umbrella"] )



	test.set_model_parameters(obs,trans,init_probs, emission_probs )

	test.get_posterior_probs(obs)


def test_hmm():
	'''
	test the correctness of hmm model
	show how to construct a hmm model
	'''
	## load emission probs for each channel into a 2D array. 
	## row = # of hidden states 
	## 1: childless sing,with parents 2: childless single, left 3: xxx 4: xx
	## col = emission probs for channel states
	em_marr = np.array([[0.000997009, 0.003988036, 0.9950150],
		[0.001620140, 0.104561316, 0.8938185],
		[0.010343968, 0.413010967, 0.5766451],
		[0.031156530, 0.672357926, 0.2964855]],order='F')
	em_child = np.array([[0.9975050, 0.00249501],
		[0.9540918, 0.04590818],
		[0.8066367, 0.19336327],
		[0.5999251, 0.40007485]],order='F')
	em_left = np.array([[0.06524451, 0.9347555],
		[0.32372754, 0.6762725],
		[0.62761976, 0.3723802],
		[0.78393214, 0.2160679]],order='F')

	# observation data frame
	child_arr = np.loadtxt('example_data/Array2.csv',delimiter=',',dtype = np.uint32)

	left_arr = np.loadtxt('example_data/Array3.csv',delimiter=',',dtype = np.uint32)

	marr_arr = np.loadtxt('example_data/Array1.csv',delimiter=',',dtype = np.uint32)


	# first element is obs. sec element is emission probs
	obs ={ "marr":marr_arr,"child":child_arr,"left":left_arr}
	emission_probs ={ "marr":em_marr,"child":em_child,"left":em_left}

	# transition probs:
	# row = col = # hidden states
	trans = np.array([[0.90, 0.06, 0.03, 0.01],
       [0, 0.90, 0.07, 0.03],
       [0,    0, 0.90, 0.10],
       [0,    0,    0,    1]],order='F')

	# initial probs
	init_probs = np.array([0.9, 0.07, 0.02, 0.01],order='F')

	single_observation={ "marr":np.array([marr_arr[:,0]]).transpose(),"child":np.array([child_arr[:,0]]).transpose(),"left":np.array([left_arr[:,0]]).transpose()}

	test =build_hmm(["marr","child","left"] )

	test.set_model_parameters(obs,trans,init_probs, emission_probs )


	#test.set_parameters(trans,emission_probs,init_probs,single_observation)
	#test.parameter_estimation()

	test.get_posterior_probs(single_observation)

	# print('shape................child..')
	# print(child_arr[:,0])
	# print('marr')
	# print(marr_arr[:,0])
	# print('left')
	# print(left_arr[:,0])
	# print('......')
	test.get_posterior_probs()
	# viterbix=test.hidden_paths_viterbix()
	# # print("...................hidden paths viterbix .................")
	# with open('hidden_path.csv','wb') as out:
	# 	csv_out = csv.writer(out)
	# print(viterbix)
	# print(type(viterbix[0].shape))
	# print(viterbix[0][0,:])

def test_bkt():
	'''
	show how to construct a bkt model
	'''
	# observation node: answer correctness
	obs_task = np.array([[1,1,1,0],[1,1,0,0],[1,1,0,0],[0,0,1,0],[1,1,1,1],[1,1,0,1]],order='F')
	# observation node: affect node
	obs_aff = np.array([[0,0,1,0],[0,1,0,1],[0,1,1,0],[0,1,1,0],[1,1,1,1],[1,1,0,1]],order='F')

	single_task=[obs_task[0,:]]
	single_aff =[obs_aff[0,:]]

	test = build_bkt(['affect'])
	test.add_task_node_obs_data(obs_task)
	test.add_node_observation_data("affect",obs_aff)
	test.add_node_emission("affect",np.array([[0.23,0.44],[0.33,0.87]]))
	test.commit_bkt_model()

	# get the posterior probs without training the bkt model
	test.get_posterior_probs({'task_node':np.array(single_task).transpose(),'affect':np.array(single_aff).transpose()})
	

#test_bkt()
dummy_hmm_test()