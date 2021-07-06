# Running a singlet yield calculation
The first step is to declare parameters using the `Parameters` class
```python
parameters = spin_dynamics.Parameters(
                 calculation_flag="static", kS=1.0E6, kT=1.0E6,
                 J=0.0, D=0.0, D_epsilon=0.0, num_threads=1,
                 approx_flag="exact", epsilon=100,
                 nlow_bins=4000, nhigh_bins=1000,
                 nfrequency_flag="broadband",
                 nt=128, ntrajectories=1000000,
                 tau=5.0E-10, gpu_flag=False
                 )
```
```
calculation_flag : str, optional  
    Specify which type of calculation to run. Options are static,  
    floquet, gamma_compute, KMC, wavepacket (Default: static)  
kS : float, optional
    Singlet recombination reaction rate (Default: 1.0E6)
kT : float, optional
    Triplet recombination reaction rate (Default: 1.0E6)
J : float, optional
    Exchange coupling strength in mT (Default: 0.0)
D : float, optional
    Dipolar coupling strength in mT (Default: 0.0)
D_epsilon : float, optional
    Angle defining the dipolar coupling tensor (Default: 0.0)
num_threads : int, optional
    Number of threads to run calculation steps. NOTE: the linear
    algebra steps will automatically parallelise, to control the
    number of threads on these steps, set the environment variables
    e.g. OMP_NUM_THREADS (Default: 1)
approx_flag : str, optional
    For static calculations, determine whether to run the exact or
    approximate calculation. Options exact or approx. (Default: exact)
epsilon : float, optional
    Set convergence parameter for approximate static or floquet
    calculations (Default: 100)
nlow_bins : int, optional
    Number of 'low' frequency histogram bins (Default: 4000)
nhigh_bins : int, optional
    Number of 'high' frequency histogram bins (Default: 1000)
nfrequency_flag : str, optional
    Specify if floquet calculation is for a single mode or broadband
    (Default: broadband)
nt : int, optional
    Number of time steps into which to divide time period of gamma_compute
    calculation (Default: 128)
ntrajectories : int, optional
    Number of KMC trajectories to average over (Default: 1000000)
tau : float, optional
    Size of timestep in wavepacket calculation in seconds
    (Default: 5.0E-10)
gpu_flag : boolean, optional
    Allow use of CUDA enabled GPU. Note, this is currently only 
    implemented for Symmetric, Exact, Separable calculations (Default:
    False)
```
## How to define electron-nuclear hyperfine coupling
For your radical pair, there will be spin active nuclei coupled to the electrons on radicals A and B. You will need an integer of how many nuclei are in each radical, an array containing the spin multiplicities of the nuclei, and another of the hyperfine coupling tensors. Here are some examples
```python
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
```
## Is the Hamiltonian separable?
In the next step, we'll build the system Hamiltonians. If the radical pair is separable, the Hilbert spaces for the two radicals will be kept separate and you will keep separate class objects for the two radicals, whereas if not, there will be a single combined set of matrices.  

A radical pair is not separable if one or more of these applies:
- The radicals are coupled i.e. non-zero J or D
- Asymmetric recombination
- Gamma compute calculation
- KMC calculation

Knowing this, the next step is to build the Hamiltonians
```python
#Separable
hA, hB = spin_dynamics.build_hamiltonians(
                          parameters, nA, nB, mA, mB,
                          A_tensorA, A_tensorB
                          )
```
or
```python
#Not separable
h = spin_dynamics.build_hamiltonians(
                     parameters, nA, nB, mA, mB,
                     A_tensorA, A_tensorB
                     )
```
```
parameters : Parameters
    Object containing calculation parameters
nA : int
    Number of spin-active nuclei in radical A
nB : int
    Number of spin-active nuclei in radical B
mA : (M) array_like
    Spin multiplicities of nuclei in radical A
mB : (M) array_like
    Spin multiplicities of nuclei in radical B
A_tensorA : (M,3,3) array_like
    Hyperfine coupling tensors of nuclei in radical A in mT
A_tensorB : (M,3,3) array_like
    Hyperfine coupling tensors of nuclei in radical B in mT
```
## Transform to the eigenbasis
With the Hamiltonians defined, we now transform the system into either the combined or the separate radical eigenbasis
```python
#Separable
hA.transform(B0, theta, phi)
hB.transform(B0, theta, phi)

#Not separable
h.transform(B0, theta, phi)
```
```
B0 : float
    External field strength in muT
theta : float
    Polar angle of the external field
phi : float
    Azimuthal angle of the external field
```
One exception is for single frequency Floquet calculations where we need to transform the perturbation Hamiltonian too
```python
#Single frequency Floquet, separable
hA.transform(B0, theta, phi, B1, theta_rf, phi_rf)
hB.transform(B0, theta, phi, B1, theta_rf, phi_rf)

#Single frequency Floquet, not separable, or gamma compute
h.transform(B0, theta, phi, B1, theta_rf, phi_rf) 
```
```

B1 : float
    Strength of perturbation field in nT
theta_rf : float
    Polar angle of perturbation field
phi_rf : float
    Azimuthal angle of perturbation field
```
Another is for wavepacket calculations, where we need to include the phase of the time dependent field (Note: this doesn't transform to the eigenbasis, but simply builds the static and time-dependent Hamiltonians)
```python
hA.transform(B0, theta, phi, B1, theta_rf, phi_rf, phase)                                                
hB.transform(B0, theta, phi, B1, theta_rf, phi_rf, phase)
```
```
phase : float
  Phase of time dependent field
```
## Additional steps
Depending on the type of calculation you're doing, there may be an additional step required before we can compute the singlet yield
```python
#Asymmetric recombination, approximate
h.build_degenerate_blocks(parameters)
```
```
parameters : Parameters
    Object containing calculation parameters
```
```python
#Floquet uncoupled, single frequency
hA.floquet_matrices(parameters, w_rf, phase)
hB.floquet_matrices(parameters, w_rf, phase)

#Floquet coupled, single frequency
h.floquet_matrices(parameters, w_rf, phase)

#Gamma compute
h.build_propagator(parameters, w_rf)
```
```
B1 : float
    Field strength in nT
w_rf : float
    Frequency of the RF field in s^(-1)
phase : float
    Phase of the RF field
```
```python
#Floquet uncoupled, broadband
hA.floquet_matrices(parameters, B1, wrf_min, wrf_max,
                    phase=phase, theta_rf=theta_rf, phi_rf=phi_rf,
                    wrf_0=wrf_0)
hB.floquet_matrices(parameters, B1, wrf_min, wrf_max,
                    phase=phase, theta_rf=theta_rf, phi_rf=phi_rf,
                    wrf_0=wrf_0)
                    
#Floquet coupled, broadband
h.floquet_matrices(parameters, B1, wrf_min, wrf_max,
                   phase=phase, theta_rf=theta_rf, phi_rf=phi_rf,
                   wrf_0=wrf_0)
```
```
B1 : (M) array_like or float
    Field strength in nT. If a scalar is passed, equivalent to
    np.ones(M)*B1
wrf_min : float
    Minimum frequency of RF band in s^{-1}
wrf_max : float
    Maximum frequency of RF band in s^{-1}
phase : (M) array_like or float, optional
    Phase of the RF components. If a scalar is passed, equivalent
    to np.ones(M)*phase (Default: 0.0)
theta_rf : (M) array_like or float, optional
    Polar angle of the field vectors of the RF components. If a 
    scalar is passed, equivalent to np.ones(M)*phase (Default: 0.0)
phi_rf : (M) array_like or float, optional
    Azimuthal angle of the field vectors of the RF components. If a 
    scalar is passed, equivalent to np.ones(M)*phase (Default: 0.0)
wrf_0 : float, optional
    Frequency spacing in the RF band (Default: 1.0E3
```
## Calculating the singlet yield
Now we're ready to calculate the singlet yield which is different if the radical pair is separable or not
```python
#Separable
spin_dynamics.compute_singlet_yield(parameters, hA = hA, hB = hB)

#Not separable
spin_dynamics.compute_singlet_yield(parameters, h = h)
```
The only exception is for wavepacket calculations
```python
spin_dynamics.compute_singlet_yield(parameters, hA = hA, hB = hB, wrf=wrf, phase=phase)
```
and we've calculated the singlet yield for your radical pair!
