# spin_dynamics
Python package for calculating the yield of radical pair reactions with either a static or a periodic time dependent external field.
### Installation
Prerequisites:
- Python 3
- numpy
- scipy

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
                         [0.0, 0.0, 1.7569]]

#-----------------------------------------------------------------------------#

parameters = Parameters(calculation_flag="static", kS=1.0E6, kT=1.0E6, 
                        J=0.0, D=0.0, D_epsilon=0.0, num_threads=1,
                        approx_flag="exact")
hA, hB = build_hamiltonians(parameters, nA, nB, mA, mB, A_tensorA, A_tensorB)
hA.transform(B0, theta, phi)
hB.transform(B0, theta, phi)

print(compute_singlet_yield(parameters, hA=hA, hB=hB))
```
