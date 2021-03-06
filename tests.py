import unittest
import numpy as np
import spin_dynamics

#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#

#Radical parameters
nA = 1
mA = np.array([2])
A_tensorA = np.zeros([1,3,3], dtype=float)
A_tensorA[0] = np.array([[-0.0636, 0.0, 0.0],
                         [0.0, -0.0636, 0.0],
                         [0.0, 0.0, 1.0812]])

nB = 1
mB = np.array([2])
A_tensorB = np.zeros([1,3,3], dtype=float)
A_tensorB[0] = np.array([[-0.0989, 0.0, 0.0],
                         [0.0, -0.0989, 0.0],
                         [0.0, 0.0, 1.7569]])

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

class TestSymmetricUncoupledExact(unittest.TestCase):

    global nA, mA, A_tensorA
    global nB, mB, A_tensorB
    global B0, theta, phi

    def test_sy_calc(self):
        parameters = spin_dynamics.Parameters(
                      calculation_flag="static", kS=1.0E6, kT=1.0E6, 
                      J=0.0, D=0.0, D_epsilon=0.0, num_threads=1,
                      approx_flag="exact")
        hA, hB = spin_dynamics.build_hamiltonians(
                  parameters, nA, nB, mA, mB, A_tensorA, A_tensorB)
        hA.transform(B0, theta, phi)
        hB.transform(B0, theta, phi)

        output = spin_dynamics.compute_singlet_yield(parameters, hA=hA, hB=hB)

        self.assertAlmostEqual(output, 0.3549952509525301, 7,
                               "Symmetric Uncoupled Exact failed")
                                    
#-----------------------------------------------------------------------------#

class TestSymmetricUncoupledApprox(unittest.TestCase):

    global nA, mA, A_tensorA
    global nB, mB, A_tensorB
    global B0, theta, phi

    def test_sy_calc(self):
        parameters = spin_dynamics.Parameters(
                      calculation_flag="static", kS=1.0E6, kT=1.0E6, 
                      J=0.0, D=0.0, D_epsilon=0.0, num_threads=1,
                      approx_flag="approx", epsilon=100,
                      nlow_bins=nlow_bins, nhigh_bins=nhigh_bins)
        hA, hB = spin_dynamics.build_hamiltonians(
                  parameters, nA, nB, mA, mB, A_tensorA, A_tensorB)
        hA.transform(B0, theta, phi)
        hB.transform(B0, theta, phi)

        output = spin_dynamics.compute_singlet_yield(parameters, hA=hA, hB=hB)

        self.assertAlmostEqual(output, 0.3550022946649809, 7,
                               "Symmetric Uncoupled Approx failed")

#-----------------------------------------------------------------------------#

class TestSymmetricCoupledExact(unittest.TestCase):

    global nA, mA, A_tensorA
    global nB, mB, A_tensorB
    global B0, theta, phi
    global J, D, D_epsilon

    def test_sy_calc(self):
        parameters = spin_dynamics.Parameters(
                      calculation_flag="static", kS=1.0E6, kT=1.0E6, 
                      J=J, D=D, D_epsilon=D_epsilon, num_threads=1,
                      approx_flag="exact")
        h = spin_dynamics.build_hamiltonians(
                  parameters, nA, nB, mA, mB, A_tensorA, A_tensorB)
        h.transform(B0, theta, phi)

        output = spin_dynamics.compute_singlet_yield(parameters, h=h)

        self.assertAlmostEqual(output, 0.3655773420988036, 7,
                               "Symmetric Uncoupled Approx failed")

#-----------------------------------------------------------------------------#

class TestAsymmetricExact(unittest.TestCase):

    global nA, mA, A_tensorA
    global nB, mB, A_tensorB
    global B0, theta, phi
    global J, D, D_epsilon

    def test_sy_calc(self):
        parameters = spin_dynamics.Parameters(
                      calculation_flag="static", kS=1.0E6, kT=1.0E5, 
                      J=J, D=D, D_epsilon=D_epsilon, num_threads=1,
                      approx_flag="exact")
        h = spin_dynamics.build_hamiltonians(
                  parameters, nA, nB, mA, mB, A_tensorA, A_tensorB)
        h.transform(B0, theta, phi)

        output = spin_dynamics.compute_singlet_yield(parameters, h=h)

        self.assertAlmostEqual(output, 0.8192001880650205, 7,
                               "Asymmetric Exact failed")

#-----------------------------------------------------------------------------#

class TestAsymmetricApprox(unittest.TestCase):

    global nA, mA, A_tensorA
    global nB, mB, A_tensorB
    global B0, theta, phi
    global J, D, D_epsilon

    def test_sy_calc(self):
        parameters = spin_dynamics.Parameters(
                      calculation_flag="static", kS=1.0E4, kT=1.0E3, 
                      J=0.0, D=0.0, D_epsilon=0.0, num_threads=1,
                      epsilon=epsilon, approx_flag="approx")
        h = spin_dynamics.build_hamiltonians(
                  parameters, nA, nB, mA, mB, A_tensorA, A_tensorB)
        h.transform(B0, theta, phi)
        h.build_degenerate_blocks(parameters)

        output = spin_dynamics.compute_singlet_yield(parameters, h=h)

        self.assertAlmostEqual(output, 0.8182861386174415, 7,
                               "Asymmetric Approx failed")

#-----------------------------------------------------------------------------#

class TestFloquetUncoupledSingleFrequency(unittest.TestCase):

    global nA, mA, A_tensorA
    global nB, mB, A_tensorB
    global B0, theta, phi
    global B1, theta_rf, phi_rf
    global w_rf, phase

    def test_sy_calc(self):
        parameters = spin_dynamics.Parameters(
                      calculation_flag='floquet', kS=1.0E3, kT=1.0E3,
                      J=0.0, D=0.0, D_epsilon=0.0, num_threads=1,
                      epsilon=epsilon, nfrequency_flag='single_frequency')
        hA, hB = spin_dynamics.build_hamiltonians(
                  parameters, nA, nB, mA, mB, A_tensorA, A_tensorB)
        hA.transform(B0, theta, phi, B1, theta_rf, phi_rf)
        hB.transform(B0, theta, phi, B1, theta_rf, phi_rf)

        hA.floquet_matrices(parameters, w_rf, phase)
        hB.floquet_matrices(parameters, w_rf, phase)

        output = spin_dynamics.compute_singlet_yield(parameters, hA=hA, hB=hB)

        self.assertAlmostEqual(output, 0.3539557463720032, 7,
                               "Floquet Uncoupled Single Frequency failed")
                                    
#-----------------------------------------------------------------------------#

class TestFloquetUncoupledBroadband(unittest.TestCase):

    global nA, mA, A_tensorA
    global nB, mB, A_tensorB
    global B0, theta, phi
    global B1, wrf_min, wrf_max

    def test_sy_calc(self):
        parameters = spin_dynamics.Parameters(
                      calculation_flag='floquet', kS=1.0E3, kT=1.0E3,
                      J=0.0, D=0.0, D_epsilon=0.0, num_threads=1,
                      nfrequency_flag='broadband')
        hA, hB = spin_dynamics.build_hamiltonians(
                  parameters, nA, nB, mA, mB, A_tensorA, A_tensorB)
        hA.transform(B0, theta, phi)
        hB.transform(B0, theta, phi)

        hA.floquet_matrices(parameters, B1, wrf_min, wrf_max, 
                            phase=phase, theta_rf=theta_rf, phi_rf=phi_rf,
                            wrf_0=wrf_0)
        hB.floquet_matrices(parameters, B1, wrf_min, wrf_max,
                            phase=phase, theta_rf=theta_rf, phi_rf=phi_rf,
                            wrf_0=wrf_0)

        output = spin_dynamics.compute_singlet_yield(parameters, hA=hA, hB=hB)

        self.assertAlmostEqual(output, 0.3539557463720032, 7,
                               "Floquet Uncoupled Broadband failed")
                                    
#-----------------------------------------------------------------------------#

class TestFloquetCoupledSingleFrequency(unittest.TestCase):

    global nA, mA, A_tensorA
    global nB, mB, A_tensorB
    global B0, theta, phi
    global B1, theta_rf, phi_rf
    global w_rf, phase

    def test_sy_calc(self):
        parameters = spin_dynamics.Parameters(
                      calculation_flag='floquet', kS=1.0E3, kT=1.0E3,
                      J=J, D=D, D_epsilon=D_epsilon, num_threads=1,
                      epsilon=epsilon, nfrequency_flag='single_frequency')
        h = spin_dynamics.build_hamiltonians(
                  parameters, nA, nB, mA, mB, A_tensorA, A_tensorB)
        h.transform(B0, theta, phi, B1, theta_rf, phi_rf)

        h.floquet_matrices(parameters, w_rf, phase)

        output = spin_dynamics.compute_singlet_yield(parameters, h=h)

        self.assertAlmostEqual(output, 0.3646169371256939, 7,
                               "Floquet Coupled Single Frequency failed")
                                    
#-----------------------------------------------------------------------------#

class TestFloquetCoupledBroadband(unittest.TestCase):

    global nA, mA, A_tensorA
    global nB, mB, A_tensorB
    global B0, theta, phi
    global B1, theta_rf, phi_rf
    global wrf_min, wrf_max

    def test_sy_calc(self):
        parameters = spin_dynamics.Parameters(
                      calculation_flag='floquet', kS=1.0E3, kT=1.0E3,
                      J=J, D=D, D_epsilon=D_epsilon, num_threads=1,
                      nfrequency_flag='broadband')
        h = spin_dynamics.build_hamiltonians(
                  parameters, nA, nB, mA, mB, A_tensorA, A_tensorB)
        h.transform(B0, theta, phi)

        h.floquet_matrices(parameters, B1, wrf_min, wrf_max,
                           phase=phase, theta_rf=theta_rf, phi_rf=phi_rf,
                           wrf_0=wrf_0)

        output = spin_dynamics.compute_singlet_yield(parameters, h=h)

        self.assertAlmostEqual(output, 0.3506744668044209, 7,
                               "Floquet Coupled Broadband failed")
                                    
#-----------------------------------------------------------------------------#

class TestGammaCompute(unittest.TestCase):

    global nA, mA, A_tensorA
    global nB, mB, A_tensorB
    global B0, theta, phi
    global B1, theta_rf, phi_rf
    global w_rf, nt

    def test_sy_calc(self):
        parameters = spin_dynamics.Parameters(
                      calculation_flag='gamma_compute', kS=1.0E3, kT=1.0E3,
                      J=0.0, D=0.0, D_epsilon=0.0, num_threads=1, nt=nt)
        h = spin_dynamics.build_hamiltonians(
                  parameters, nA, nB, mA, mB, A_tensorA, A_tensorB)
        h.transform(B0, theta, phi, B1, theta_rf, phi_rf)

        h.build_propagator(parameters, w_rf)

        output = spin_dynamics.compute_singlet_yield(parameters, h=h)

        self.assertAlmostEqual(output, 0.3539554614892267, 7,
                               "Gamma Compute failed")
                                    
#-----------------------------------------------------------------------------#

class TestKMC(unittest.TestCase):

    global nA, mA, A_tensorA
    global nB, mB, A_tensorB
    global B0, theta, phi
    global ntrajectories

    def test_sy_calc(self):
        parameters = spin_dynamics.Parameters(
                      calculation_flag='KMC', kS=1.0E3, kT=1.0E3,
                      J=0.0, D=0.0, D_epsilon=0.0, num_threads=1,
                      ntrajectories=ntrajectories)
        h = spin_dynamics.build_hamiltonians(
                  parameters, nA, nB, mA, mB, A_tensorA, A_tensorB)
        h.transform(B0, theta, phi)

        output = spin_dynamics.compute_singlet_yield(parameters, h=h)

        self.assertTrue(output < 0.36 and output > 0.35, "KMC failed")
                                    
#-----------------------------------------------------------------------------#

class TestWavepacket(unittest.TestCase):

    global nA, mA, A_tensorA
    global nB, mB, A_tensorB
    global B0, theta, phi
    global B1, theta_rf, phi_rf
    global w_rf, phase

    def test_sy_calc(self):
        parameters = spin_dynamics.Parameters(
                      calculation_flag='wavepacket', kS=1.0E6, kT=1.0E6,
                      J=0.0, D=0.0, D_epsilon=0.0, num_threads=1)
        hA, hB = spin_dynamics.build_hamiltonians(
                  parameters, nA, nB, mA, mB, A_tensorA, A_tensorB)
        hA.transform(B0, theta, phi, B1, theta_rf, phi_rf, phase)
        hB.transform(B0, theta, phi, B1, theta_rf, phi_rf, phase)

        output = spin_dynamics.compute_singlet_yield(
                    parameters, hA=hA, hB=hB, wrf=w_rf, phase=phase)

        self.assertAlmostEqual(output, 0.35472252341183663, 7,
                               "Wavepacket failed")
                                    
#-----------------------------------------------------------------------------#

class TestHeadingAccuracy(unittest.TestCase):

    def test_heading_accuracy_calc(self):
        angles, sy_values = spin_dynamics.load_test_data()
        retina_signal = spin_dynamics.RetinaSignal(40, angles=angles,
                                                   sy_values=sy_values)

        heading_accuracy = spin_dynamics.HeadingAccuracy(retina_signal, 1000)
        output = heading_accuracy.lower_bound_error(
                retina_signal, 1.0E6, num_threads=12)

        self.assertTrue(output < 0.95 and output > 0.85, 
                        "Heading Accuracy calculation failed")

#-----------------------------------------------------------------------------#

class TestCNN(unittest.TestCase):

    def test_CNN(self):
        retina_signal = spin_dynamics.RetinaSignal(
                            40, func_form=gx, func_parameters=[np.pi/7.0])
        conv_model = spin_dynamics.ConvolutionalModel(40)
        with open('src/cnn_model.txt','r') as f:
            json_model = f.read()   
        new_model = spin_dynamics.read_in_model(json_model)

        for i in range(len(new_model.layers)):
            self.assertTrue(conv_model.model.input_shape == 
                                new_model.input_shape)
            self.assertTrue(conv_model.model.output_shape == 
                                new_model.output_shape)

def gx(th, phi, params):                                                                                 
        gx = 0.5 - 1.0*1.27E-3*np.exp(-0.5*((th - np.pi/2.0)/params[0])**2.0)\
                /(1.0 - np.exp(-0.5*(np.pi/2.0/params[0])**2.0))
        return gx

#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#

if __name__ == "__main__":
    unittest.main()

