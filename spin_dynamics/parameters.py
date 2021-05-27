import numpy as np
from numba import set_num_threads

#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#

class Parameters:
    '''Specify the parameters for spin_dynamics calculation

    Parameters 
    ----------
    calculation_flag : str, optional
        Specify which type of calculation to run. Options are static,
        floquet, gamma_compute, KMC (Default: static)
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

    Returns
    -------
    p : Parameters
        Object containing the calculation parameters

    Raises
    ------
    Rate constants too large for approx calculation
        Trying to use recombination rate constants that are too large
        for the approximate method for asymmetric recombination
    Coupling cannot be included for approx calculation
        For symmetric or asymmetric "static" calculations, J or D cannot
        be non-zero for the approximate methods
    Symmetric recombination required for floquet calculation
        The Floquet algorithm requires symmetric recombination
    Symmetric recombination required for gamma compute calculation
        The gamma compute algorithm requires symmetric recombination
    Coupling cannot be included for gamma compute calculation
        J and D must be zero for a gamma compute calculation
    Unrecognised calculation flag
        If the calculation flag is not recognised

    Notes
    -----
    A radical pair is not separable if one or more of these applies:
        The radicals are coupled i.e. non-zero J or D
        Asymmetric recombination
        Gamma compute calculation
        KMC calculation
'''
    def __init__(self, calculation_flag="static", kS=1.0E6, kT=1.0E6,
                 J=0.0, D=0.0, D_epsilon=0.0, num_threads=1,
                 approx_flag="exact", epsilon=100,
                 nlow_bins=4000, nhigh_bins=1000,
                 nfrequency_flag="broadband",
                 nt=128, ntrajectories=1000000):
        '''Initialise parameters
'''

        self.__calculation_flag = calculation_flag
        self.__symmetric_flag = False
        self._kS = kS
        self._kT = kT
        if kS == kT:
            self.__symmetric_flag = True

        gamma_e = 1.76E8
        self._J = J * gamma_e
        self._D = D * gamma_e
        self.D_epsilon = D_epsilon
        self.__coupled_flag = False
        if (self.J != 0.0 or self.D != 0.0):
            self.__coupled_flag = True

        self.num_threads = num_threads
        set_num_threads(num_threads)

        if self.__calculation_flag == "static":
            self.__approx_flag = approx_flag
            self.epsilon = epsilon
            if self.__approx_flag == "approx":
                if not self.__symmetric_flag:
                    if (kS > 1.0E4 or kT > 1.0E4):
                        raise Exception(
                            "Rate constants too large for approx calculation")
                if self.__coupled_flag:
                    raise Exception(
                        "Coupling cannot be included for approx calculation")
                self.nlow_bins = nlow_bins
                self.nhigh_bins = nhigh_bins
                self.divider_bin = int(10*self.kS)
                self.low_delta = self.divider_bin/float(self.nlow_bins)
                self.low_bins = np.arange(0, self.divider_bin, 
                                          self.low_delta) + self.low_delta/2.0
                self.nbins = self.nlow_bins + self.nhigh_bins

        elif self.__calculation_flag == 'floquet':
            if not self.__symmetric_flag:
                raise Exception(
                    "Symmetric recombination required for floquet calculation")
            self.__nfrequency_flag = nfrequency_flag
            self.epsilon = epsilon
        elif self.__calculation_flag == 'gamma_compute':
            if not self.__symmetric_flag:
                raise Exception(
                    "Symmetric recombination required for gamma compute calculation")
            if self.__coupled_flag:
                raise Exception(
                    "Coupling cannot be included for gamma compute calculation")
            self.nt = nt
        elif self.__calculation_flag == 'KMC':
            self.ntrajectories = ntrajectories
        else:
            raise Exception("Unrecognised calculation flag")

    @property
    def kS(self):
        return self._kS

    @property
    def kT(self):
        return self._kT

    @kS.setter
    def kS(self, value):
        self._kS = value
        if self.__symmetric_flag:
            if self._kS != self._kT:
                self.__symmetric_flag = False
        
    @kT.setter
    def kT(self, value):
        self._kT = value
        if self.__symmetric_flag:
            if self._kS != self._kT:
                self.__symmetric_flag = False

    @property
    def J(self):
        return self._J

    @property
    def D(self):
        return self._D

    @J.setter
    def J(self, value):
        self._J = value
        if value != 0.0:
            self.__coupled_flag = True

    @D.setter
    def D(self, value):
        self._D = value
        if value != 0.0:
            self.__coupled_flag = True

    @property
    def calculation_flag(self):
        return self.__calculation_flag

    @property
    def symmetric_flag(self):
        return self.__symmetric_flag

    @property
    def approx_flag(self):
        return self.__approx_flag

    @property
    def coupled_flag(self):
        return self.__coupled_flag

    @property
    def nfrequency_flag(self):
        return self.__nfrequency_flag

#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#

def readinfile(filename):
    """Read in singlet yield data from file
    output theta array and singlet yield array"""
    input_file = open(filename,'r')
    data = input_file.readlines()

    try:
        float(data[-1].split()[0])
        ntheta = len(data)
    except IndexError:
        ntheta = len(data)-1

    theta = np.zeros(ntheta)
    phis = np.zeros(ntheta)

    for i in range(ntheta):
        theta[i] = float(data[i].split()[0])
        phis[i] = float(data[i].split()[1])

    input_file.close()

    return theta, phis

#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#

