# spin_dynamics
Python package for modelling a radical pair magnetoreceptor. The primary use of this package is in calculating the yield of radical pair reactions using a variety of different algorithms, depending on the choice of parameters. Once the anisotropic magnetic field effect has been calculated, there is also the functionality to construct a representation of the signal in the retina generated by a radical pair sensor. The pacakge also has the functionality to build and train a convolutional neural network on these representations to learn the mapping from heading direction to singlet yield signal. This can be extended to compute an Information Theoretic upper bound on the attainable accuracy of a radical pair sensor.

Details of the different algorithms used can be found in my PhD thesis: H. G. Hiscock (2018). _Long-lived spin coherence in radical pair compass magnetoreception_. (University of Oxford)
### Installation
Prerequisites:
- Python3
- numpy
- scipy
- numba
- tensorflow
- CUDA if using a GPU

To build the package, run:
```
python3 setup.py build
```
## Computing the Singlet Yield
The general structure of a singlet yield calculation using the `spin_dynamics` package is as follows:
- Declare the parameters for the calculation, including which algorithm to use
- Build the relevant Hamiltonians for the system
- Transform the Hilbert space into the eigenbasis for a given external field strength and direction
- (Parameter dependent) Construct additional matrices needed for the final calculation
- Compute the singlet yield

Here's a simple example code:
```python
import numpy as np
from spin_dynamics import Parameters, build_hamiltonians, compute_singlet_yield

#-----------------------------------------------------------------------------#

#External field parameters
B0 = 50
theta = 0.0
phi = 0.0

# Radical parameters
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

#-----------------------------------------------------------------------------#

parameters = Parameters(calculation_flag="static", kS=1.0E6, kT=1.0E6, 
                        J=0.0, D=0.0, D_epsilon=0.0, num_threads=1,
                        approx_flag="exact")
hA, hB = build_hamiltonians(parameters, nA, nB, mA, mB, A_tensorA, A_tensorB)
hA.transform(B0, theta, phi)
hB.transform(B0, theta, phi)

print(compute_singlet_yield(parameters, hA=hA, hB=hB))
```
This will output the singlet yield for a radical pair in a static magnetic field of strength 50 mT and direction relative to the principle axis of the radical pair described by (theta, phi) = (0, 0) with symmetric recombination, kS = kT = 1.0 x 10^6 s^(-1). The two radicals are uncoupled, and both include a single spin-1/2 nucleus with hyperfine tensor given by A_tensor(A/B). Because the system is fully separable, the Hamiltonians for each radical are handled separately, greatly speeding up the calculation, and combined only to calculate the singlet yield.

Where the previous example contained separate Hamiltonians `hA` and `hB`, here's an example using a different calculation type and a non-separable Hilbert space
```python
# E-E coupling parameters
J = 0.0
D = -0.4065
D_epsilon = np.pi/4.0


# RF field parameters
B1 = 50
theta_rf = 0.0
phi_rf = 0.0
w_rf = 1.317E7
phase = 0.0

#-----------------------------------------------------------------------------#

parameters = Parameters(calculation_flag='floquet', kS=1.0E3, kT=1.0E3,
                        J=J, D=D, D_epsilon=D_epsilon, num_threads=1,
                        epsilon=epsilon, nfrequency_flag='single_frequency')
h = build_hamiltonians(parameters, nA, nB, mA, mB, A_tensorA, A_tensorB)
h.transform(B0, theta, phi, B1, theta_rf, phi_rf)
h.floquet_matrices(parameters, w_rf, phase)

print(compute_singlet_yield(parameters, h=h))
```
In contrast to the previous example, with a non-separable Hilbert space the function `build_hamiltonians` outputs a single class object representing the combined Hilbert space. The `transform` function also takes additional arguments to define the time-dependent field. There is an additional function call `floquet_matrices` which constructs the matrices used in the Floquet algorithm to compute the singlet yield.

## Generating the signal across the retina
With the singlet yield calculated for a given radical pair over the full domain of polar angles, we can generate a representation of the resulting signal available across the retina. With an array of `N` singlet yield values `sy_values` and a `N x 2` array containing the corresponding polar angles `angles` (phi values in the first column, theta in the second):
```python
from spin_dynamics import RetinaSignal

#Specify number of grid points in each dimension for the retina signal
Ngrid = 40

retina_signal = RetinaSignal(Ngrid, angles, sy_values)
sy_retina = retina_signal.generate_signal(heading = 0.0)
```
This can be repeated, but instead of giving the singlet yield signal in the limit of infinite averaging over photocycles, you can specify a fixed number of trajectories over which to average
```python
sy_retina = retina_signal.generate_signal(ntrajectories=10000)
```

## Building and training a CNN
With this representation in hand we can train a convolutional neural network to learn the mapping between heading direction and the singlet yield signal. The network and hyperparameters are automatically set, the only parameter we need to pass to the neural net is the size of our representation of the signal (`Ngrid`), and the `summary()` function gives the TensorFlow summary of the network architecture
```python
from spin_dynamics import ConvolutionalModel

CNN = ConvolutionalModel(Ngrid)
CNN.summary()
```
To generate training, dev and test sets for this model, we simply call the relevant functions
```python
CNN.training_data(retina_signal, ntrajectories=None)
CNN.test_data(retina_signal, ntrajectories=None)
```
where as above, you can specify the number of photocycles to average over.
Finally, simply train the model on the training set and evaluate the accuracy on the test set
```python
CNN.train_model(epochs=10)
print(CNN.evaluate())
```

## Calculating the lower bound heading error
Moving a step further, we can determine a lower bound on Bayes error for a given set of parameters, by using Information Theory to calculate a lower bound on the error in determining the heading direction from this radical pair sensor for a given number of photocycles, `ntrajectories`. The result is calculated over `nheadings` samples to converge the integral in the expression:
```python
from spin_dynamics import HeadingAccuracy

#Number of heading directions over which to sample
nheadings = 1000
#Number of photocycles to average over
ntrajectories = 1.0E6

heading_accuracy = HeadingAccuracy(retina_signal, nheadings)
heading_error = heading_accuracy.lower_bound_error(retina_signal, ntrajectories)
```

## Regression tests
The file `tests.py` contains regression tests for each possible type of calculation using this package, which serves as a useful template for all the configurations of different parameters that lead to different computations. `gpu_tests.py` contains unit tests for calculations using a CUDA enabled GPU.
