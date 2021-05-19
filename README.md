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
from spin_dynamics import Parameters, build_hamiltonians, compute_singlet_yield
parameters = Parameters()
hA, hB = build_hamiltonians()
hA.transform()
hB.transform()
compute_singlet_yield()
```
