# spin_dynamics
Ppython package to run radical pair 
spin dynamics calulations
## Example calculation
```python
import spin_dynamics
parameters = spin_dynamics.Parameters()
hA, hB = spin_dynamics.build_hamiltonians()
hA.transform()
hB.transform()
spin_dynamics.compute_singlet_yield()
```
