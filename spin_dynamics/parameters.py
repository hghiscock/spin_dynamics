import numpy as np

#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#

class Parameters:
    '''Parameters for calculation
    contains:

    Type of calculation             calculation_flag
    Recombination rates             kS, kT
    Flag if symmetric               symmetric_flag
    Electron coupling               J, D, D_epsilon
    Flag if coupled                 coupled_flag
    Number of threads               num_threads

    Flag if approximation           approx_flag
    Tolerance parameter             epsilon
    Histogram parameters            nlow_bins, nhigh_bins, nbins
                                    divider_bin, low_delta, low_bins
    Flag for number of freqs        nfrequencies_flag
    Number of time steps            nt
    Number of trajectories          ntrajectories
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

