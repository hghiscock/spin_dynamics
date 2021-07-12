# Calculating the lower bound heading error
In order to calculate the lower bound of the heading error for a given radical pair, we must first construct the representation of the singlet yield in the receptors across the retina
```python
retina_signal = spin_dynamics.RetinaSignal(
			receptor_grid, angles, sy_values,
	                beta = 0.5*np.pi, gamma = 0.0, eta = 0.5*np.pi,
           	        inclination = np.pi*11/30
			)
```
```
receptor_grid : int
    Number of receptors in each dimension in a square grid (Note, the
    total number of receptors will end up being less than this number
    squared because the retina is modelled as a circle)
angles : (Npoints, 2) ndarray of floats
    Angles for which input singlet yield values are calculated. First
    dimension contained phis values, second contained theta values
sy_values : (Npoints,) ndarray of floats
    Singlet yield values evaluated for the field directions specified
    in the angles array
beta : float, optional
    beta angle for orientation of magnetoreceptor molecule in receptor
    cell (Default: 0.5*pi)
gamma : float, optional
    gamma angle for orientation of magnetoreceptor molecule in receptor
    cell (Default: 0.0)
eta : float, optional
    eta angle for orientation of magnetoreceptor molecule in receptor
    cell (Default: 0.5*pi)
inclination : float, optional
    The inclination angle of the geomagnetic field
```
The angles beta, gamma and eta define the orientation of the radical pair in the receptor cell.

## Generate the signal
Having built the retina representation, we can build the signal for a specified or random heading direction to visualise the result
```python
sy_retina = retina_signal.generate_signal(heading=None)
```
```
heading : float, optional
    The heading direction for which to generate the singlet yield signal
    in radians, if None, the heading will be randomly selected 
    (Default: None)
```

## Calculate the lower bound error
We can also sample the retina signal over the domain of heading directions and use that data to compute the lower bound heading error. First, construct a `HeadingAccuracy` class object
```python
heading_accuracy = HeadingAccuracy(retina_signal, nheadings)
```
```
retina_signal : RetinaSignal
    Class object containing the grid of receptors and methods to generate
    a singlet yield signal
nheadings : int
    Number of headings over which to sample to converge the information
    theory integral over the domain of heading directions
```
Once this is all constructed, we can calculate the lower bound error for any number of photocycles the signal is averaged over
```python
heading_error = heading_accuracy.lower_bound_error(
			self, retina_signal,
			ntrajectories, num_threads=1
			)
```
```
retina_signal : RetinaSignal
    Class object containing the grid of receptors and methods to generate
    a singlet yield signal
ntrajectories : int
    Number of photocycles for which to calculate the lower bound heading
    error
num_threads : int, optional
    Number of threads to use in building the covariance matrix (Default:
    1)
```
