# spin_dynamics
Python package for calculating the yield of radical pair reactions with either a static or a periodic time dependent external field.
### Installation
Prerequisites:
- Python3
- numpy
- scipy
- numba

To build the package, run:
```
python setup.py build
```
## Example calculation
The general structure of a calculation using the `spin_dynamics` package is as follows:
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
In contrast to the previous example, with a non-separable Hilbert space the function `build_hamiltonians` outputs a single class object representing the combined Hilbert space. The `transform` function also takes additional arguments to define the time-dependent field. There is an additional function call `floquet_matrices` which constructs the matrices used in the Floquet algoriothm to compute the singlet yield.
 
The file `tests.py` contains unit tests for each possible type of calculation using this package, which serves as a useful template for all the configurations of different parameters that lead to different computations.
