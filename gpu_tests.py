import unittest
import numpy as np
import spin_dynamics

#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#

#Radical parameters
nA = 5
mA = np.array([2,2,2,2,2])
A_tensorA = np.zeros([5,3,3], dtype=float)
A_tensorA[0] = np.array([[-0.0636, 0.0, 0.0],
                         [0.0, -0.0636, 0.0],
                         [0.0, 0.0, 1.0812]])
A_tensorA[1] = A_tensorA[0]
A_tensorA[2] = A_tensorA[0]
A_tensorA[3] = A_tensorA[0]
A_tensorA[4] = A_tensorA[0]

nB = 5
mB = np.array([2,2,2,2,2])
A_tensorB = np.zeros([5,3,3], dtype=float)
A_tensorB[0] = np.array([[-0.0989, 0.0, 0.0],
                         [0.0, -0.0989, 0.0],
                         [0.0, 0.0, 1.7569]])
A_tensorB[1] = A_tensorB[0]
A_tensorB[2] = A_tensorB[0]
A_tensorB[3] = A_tensorB[0]
A_tensorB[4] = A_tensorB[0]

#E-E coupling parameters
J = 0.0
D = -0.4065
D_epsilon = np.pi/4.0

#External field parameters
B0 = 50
theta = 0.0
phi = 0.0

#SymmetricUncoupledApprox convergence parameters
nlow_bins = 4000
nhigh_bins = 1000
epsilon = 100

#Rf field parameters
B1 = 50
theta_rf = 0.0
phi_rf = 0.0
w_rf = 1.317E7
phase = 0.0

#Broadband parameters
wrf_min = 1.0E6
wrf_max = 1.0E7
wrf_0 = 1.0E3

#Gamma Compute time steps
nt = 128

#Number of KMC trajectories
ntrajectories = 1000000

#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#

class TestSymmetricUncoupledExactGPU(unittest.TestCase):

    global nA, mA, A_tensorA
    global nB, mB, A_tensorB
    global B0, theta, phi

    def test_sy_calc(self):
        parameters = spin_dynamics.Parameters(
                      calculation_flag="static", kS=1.0E6, kT=1.0E6, 
                      J=0.0, D=0.0, D_epsilon=0.0, num_threads=12,
                      approx_flag="exact", gpu_flag=True)
        hA, hB = spin_dynamics.build_hamiltonians(
                  parameters, nA, nB, mA, mB, A_tensorA, A_tensorB)
        hA.transform(B0, theta, phi)
        hB.transform(B0, theta, phi)

        output = spin_dynamics.compute_singlet_yield(parameters, hA=hA, hB=hB)

        self.assertAlmostEqual(output, 0.388335592180034, 7,
                               "Symmetric Uncoupled Exact GPU failed")
                                    
#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#

if __name__ == "__main__":
    unittest.main()

