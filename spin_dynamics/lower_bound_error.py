import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator
from scipy.linalg import eigh
from numba import jit, prange, set_num_threads
from numba import float64, int64

#-----------------------------------------------------------------------------#

class RetinaSignal:
    '''Set up grid of receptors in the retina and interpolator to generate
    singlet yield signal

    Parameters
    ----------
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
'''

    def __init__(self, receptor_grid, angles, sy_values,
                 beta = 0.5*np.pi, gamma = 0.0, eta = 0.5*np.pi,
                 inclination = np.pi*11/30):

        #Store angles
        self.beta = beta
        self.eta = eta
        self.gamma = gamma
        self.inclination = inclination

        #Set up grid in retina                                                                   
        r1grid, r2grid = np.mgrid[-1:1:receptor_grid*1j, -1:1:receptor_grid*1j]
        rgrid = np.sqrt(r1grid*r1grid + r2grid*r2grid)                                           
        rmask = rgrid < 1.0                                                                     
        rgrid = rgrid[rmask]                                                                    

        thgrid = np.arctan2(r1grid,r2grid)
        thgrid = thgrid[rmask]
        self.nreceptors = len(rgrid)                                                             

        self.theta = 2.0*np.arctan(rgrid)                                                       
        self.phi = np.arctan2(rgrid*np.sin(thgrid),rgrid*np.cos(thgrid))

        #Singlet yield interpolator
        self.c0int = CloughTocher2DInterpolator(angles,sy_values)

    def generate_signal(self, heading=None):
        '''Generate the average singlet yield signal across the retina, for the
        specified heading, or for a random heading

        Parameters
        ----------
        heading : float, optional
            The heading direction for which to generate the singlet yield signal
            in radians, if None, the heading will be randomly selected 
            (Default: None)

        Returns
        -------
        sy_dat : (Npoints,) ndarray of floats
            Singlet yield value at each receptor
'''

        if not heading:
            heading = np.random.uniform(-np.pi, np.pi)

        #Calculate the field direction in the retinal axis frame
        chi = np.arccos(-np.cos(heading)*np.cos(self.inclination))
        psi = np.arctan2(np.sin(self.inclination),
                         -np.cos(self.inclination)*np.sin(heading))
        Br = np.array([[np.sin(chi)*np.cos(psi)],[np.sin(chi)*np.sin(psi)],
                        [np.cos(chi)]])

        #Calculate c0 at grid points and average over eta
        self.xi, self.delta = Bm(Br,self.beta,self.gamma,self.phi,
                                 self.theta,self.eta)

        sy_dat = self.c0int(self.xi,self.delta)
        return np.squeeze(sy_dat)

#-----------------------------------------------------------------------------#

class HeadingAccuracy:
    '''Sample the singlet yield signal over heading directions and use to
    calculate the lower bound error in the heading accuracy

    Parameters
    ----------
    retina_signal : RetinaSignal
        Class object containing the grid of receptors and methods to generate
        a singlet yield signal
    nheadings : int
        Number of headings over which to sample to converge the information
        theory integral over the domain of heading directions
'''

    def __init__(self, retina_signal, nheadings):
        self.sy_data = np.zeros([retina_signal.nreceptors,nheadings])
        self.sy_av = np.zeros(retina_signal.nreceptors)
        self.dep_covar = np.zeros([retina_signal.nreceptors,nheadings])
        self.nheadings = nheadings

        for i in range(nheadings):
            sy_tmp = retina_signal.generate_signal()
            self.sy_data[:,i] = sy_tmp
            self.sy_av += sy_tmp
            self.dep_covar[:,i] = sy_tmp * (1.0-sy_tmp)

        self.sy_av = self.sy_av/nheadings

    def lower_bound_error(self, retina_signal, ntrajectories, num_threads=1):
        '''Calculate the lower bound heading error

        Parameters
        ----------
        retina_signal : RetinaSignal
            Class object containing the grid of receptors and methods to generate
            a singlet yield signal
        ntrajectories : int
            Number of photocycles for which to calculate the lower bound heading
            error
        num_threads : int, optional
            Number of threads to use in building the covariance matrix (Default:
            1)

        Returns
        -------
        lb_error : float
            Information Theory lower bound error in the heading accuracy in
            radians
'''
        set_num_threads(num_threads)

        covar = covariance(
                retina_signal.nreceptors,self.nheadings,
                ntrajectories,self.sy_data*ntrajectories,self.sy_av*ntrajectories)
        dep_covar_tmp = self.dep_covar * ntrajectories
        dep_covar = np.zeros(retina_signal.nreceptors, dtype=float)
        inheads = 1.0 / float(self.nheadings)
        for j in range(retina_signal.nreceptors):
            dep_covar[j] = np.prod(dep_covar_tmp[j,:]/dep_covar_tmp[j,0])**inheads \
                           *dep_covar_tmp[j,0]

        ecovar = eigh(covar, eigvals_only=True, overwrite_a=True,
                      overwrite_b=True, turbo=True)

        info = 0.5 * np.log(np.prod(ecovar/dep_covar))
        theta_entropy = np.log(2.0*np.pi)
        err_tmp = theta_entropy - info
        lb_error = np.sqrt(np.exp(2.0*err_tmp)/(2.0*np.pi*np.e))
                
        return lb_error

#-----------------------------------------------------------------------------#

#Define rotation matrices
def Ry(a):
    Ry = np.array([[np.cos(a),0.0,np.sin(a)],
                   [0.0,1.0,0.0],
                   [-np.sin(a),0.0,np.cos(a)]],
                  dtype=object)
    return Ry
def Rz(a):
    Rz = np.array([[np.cos(a),np.sin(a),0.0],
                   [-1.0*np.sin(a),np.cos(a),0.0],
                   [0.0,0.0,1.0]],
                   dtype=object)
    return Rz

#Transform magnetic field to molecular axis frame
def Bm(Br,beta,gamma,phi,theta,eta):
    R = np.dot(Rz(phi),np.dot(Ry(theta),np.dot(Rz(eta),
                np.dot(Ry(beta),Rz(gamma)))))
    Bm1 = np.dot(R.T,Br)
    xi = np.arccos(Bm1[2,0])
    delta = np.arccos(Bm1[0,0]/np.sin(xi))
    return xi, delta

#-----------------------------------------------------------------------------#

@jit(float64[:,:](int64,int64,int64,float64[:,:],float64[:]),
     nopython=True, cache=True, parallel=True)
def covariance(nreceptors, nheadings, ntrajectories,
               sy_dat, sy_av):

    covar = np.zeros((nreceptors,nreceptors), dtype=np.float64)
    for i in range(nreceptors):
        for j in range(i,nreceptors):
            covar_tmp = 0.0
            for k in prange(nheadings):
                covar_tmp += (sy_dat[i,k]-sy_av[i])*(sy_dat[j,k]-sy_av[j])
            covar[i,j] = covar_tmp
            covar[j,i] = covar_tmp

    for i in range(nreceptors):
        covar_tmp = 0.0
        for k in prange(nheadings):
            mu = sy_dat[i,k]
            sigma2 = mu*(ntrajectories-mu)/ntrajectories
            covar_tmp += sigma2 + mu*mu - 2.0*mu*sy_av[i] + sy_av[i]*sy_av[i]
        covar[i,i] = covar_tmp

    return covar/nheadings

#-----------------------------------------------------------------------------#

#Read in data for test calculation
def load_test_data():
    '''Load singlet yield data for unit test calculation
'''

    data = open('src/test_data.dat','r')
    l = data.readlines()
    data.close()

    ngrid = len(l)
    angles = np.zeros((ngrid,2), dtype=float)
    sy_data = np.zeros(ngrid, dtype=float)

    try:
        sy_tmp = float(l[-1].split()[2])
    except IndexError:
        ngrid -= 1

    for i in range(ngrid):
        ltmp = l[i].split()
        angles[i,0] = ltmp[0]
        angles[i,1] = ltmp[1]
        sy_data[i] = ltmp[2]

    return angles, sy_data

#-----------------------------------------------------------------------------#

