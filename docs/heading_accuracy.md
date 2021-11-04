# Calculating the lower bound heading error
In order to calculate the lower bound of the heading error for a given radical pair, we must first construct the representation of the singlet yield in the receptors across the retina
```python
retina_signal = spin_dynamics.RetinaSignal(
			receptor_grid, func_form=None, func_parameters=None,
	                angles=None, sy_values=None, beta = 0.5*np.pi, 
        	        gamma = 0.0, eta = 0.5*np.pi, 
                	inclination = np.pi*11/30
			)
```
```
receptor_grid : int
    Number of receptors in each dimension in a square grid (Note, the
    total number of receptors will end up being less than this number
    squared because the retina is modelled as a circle)
angles : (Npoints, 2) ndarray of floats, optional
    Angles for which input singlet yield values are calculated. First
    dimension contains phis values, second contains theta values
    (Default: None)
sy_values : (Npoints,) ndarray of floats, optional
    Singlet yield values evaluated for the field directions specified
    in the angles array (Default: None)
func_form : function, optional
    The functional form for the singlet yield as a function of polar
    angles theta, phi and list of other parameters (Default: None)
func_parameters : () ndarray, optional
    Extra parameters to pass to the func_form function (Default: None)
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

Raises
------
Please specify functional form
```
The angles beta, gamma and eta define the orientation of the radical pair in the receptor cell.

## Generate the signal
Having built the retina representation, we can build the signal for a specified or random heading direction to visualise the result
```python
sy_retina = retina_signal.generate_signal(heading=None, normalize=False, ntrajectories=None)
```
```
heading : float, optional
    The heading direction for which to generate the singlet yield signal
    in radians, if None, the heading will be randomly selected 
    (Default: None)
normalize : boolean, optional
    Whether to normalize the singlet yield signal for visualising or
    as training data for CNN (Default: False)
ntrajectories : int, optional
    How many photocycles over which to average, if None gives the
    infinite limit (Default: None)
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
			ntrajectories, num_threads=1,
			gpu_flag=False
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
gpu_flag : bool, optional
    Allow use of CUDA enabled GPU (Default: False)
```
