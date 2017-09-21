from hmmtest import *
import numpy as np

m= viterbi()
m.test()
print "||||"
out=m.test2(np.array( [ [1.,2.], [4.,5.] ], order="F" ),np.array( [ [3.,2.], [2.,3.] ], order="F" ))
print(out)
print "|||"
out2= m.test3(np.array([1.,2.]))
print(out2)

print "test 4"
out3= m.test4(np.array([[[2,1],[0.1,0.9]],
	[[0.1,0.5],[2,8]]
	],order='F',dtype=np.uint32))
print(out3)

g = parameter_estimation()
g.test()



transition = np.array([[0.3,0.7],[0.2,0.8]],order='F')
emission = np.array([[[0.2,0.1],[0.1,0.9]],
	[[0.1,0.5],[0.2,0.8]]
	],order='F')
init = np.array([0.25,0.75],order='F')
obs = np.array([[[1,1,0,1],[1,1,1,1]],
	[[1,1,1,0],[1,0,0,1]]
	],order='F',dtype=np.uint32)
nSymbols = np.array([1,2],dtype=np.uint32)
itermax = 100
tol=0.5
trace = 2
threads =1
print "run EM"
g.EM(transition,emission,init,obs,nSymbols,itermax,tol,trace,threads)
print "after EM"

print(g.get_init())
print(g.get_emission())
print(g.get_transition())
print(g.get_sumlogLik())
print(g.get_iterations())
#void EM(const arma::mat& transition_, const arma::cube& emission_, const arma::vec& init_,
#  const arma::ucube& obs, const arma::uvec& nSymbols, int itermax, double tol, 
#  int trace, unsigned int threads)


# print "run viterbi"
# m.run_viterbi(transition, emission, init, obs)

# print(m.get_q())
# print(m.get_logp())

