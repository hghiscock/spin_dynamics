import numpy as np
from scipy import sparse 
from scipy import linalg
from .singlet_yield import (
        energy_differences, degeneracy_check, get_indices, perturbation_matrix,
        sy_asymmetric, single_frequency_build_matrix, bin_frequencies, 
        single_frequency_build_matrix_combined, broadband_build_matrix,
        broadband_build_matrix_combined, sy_gamma_compute, complexgramschmidt,
        get_omega_rs
        )

#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#

class Hamiltonians:
    '''Read in parameters and build Hamiltonians for a given radical
    contains:

    Number of spins                n
    Value of 2*(2I+1)              m
    Number of energy gaps          ngaps

    Sx, Sy, Sz:                    hx, hy, hz
    HFI hamiltonian:               hhf
    Identity matrix:               iden

    All matrices are in sparse csr format   
'''
    def __init__(self, nspins, multiplicities, A_tensor):
        '''Build x, y and z electron spin operators and the hyperfine interaction hamiltonian
'''
        self.n = nspins
        self.m = 2*np.prod(multiplicities)
        self.ngaps = 0.5*(self.m*self.m - self.m)

        if (self.n == 0):
            mmax = 2
        else:
            mmax = max(multiplicities)
        sops = getsoperators(mmax)
        self.hx = sops[0,0]
        self.hy = sops[0,1]
        self.hz = sops[0,2]
        #Build cartesian spin operators
        for i in range(self.n):
            self.hx = sparse.kron(self.hx, sparse.eye(multiplicities[i]),
                                  format="csr")
            self.hy = sparse.kron(self.hy, sparse.eye(multiplicities[i]),
                                  format="csr")
            self.hz = sparse.kron(self.hz, sparse.eye(multiplicities[i]),
                                  format="csr")

        #Build hyperfine hamiltonians
        gamma_e = 1.76E8
        self.hhf = sparse.csr_matrix((self.m,self.m), dtype=complex)
        for i in range(self.n):
            if multiplicities[i] != 1:
                for j in range(3):
                    for l in range(3):
                        htmp = sops[0,j]
                        mtmp = int(np.prod(multiplicities[:i]))
                        htmp = sparse.kron(htmp, sparse.eye(mtmp), format="csr")
                        htmp = sparse.kron(htmp, sops[multiplicities[i]-2,l],
                                           format="csr")
                        mtmp = int(np.prod(multiplicities[i+1:]))
                        htmp = sparse.kron(htmp, sparse.eye(mtmp), format="csr")

                        atmp = A_tensor[i,j,l]*gamma_e
                        self.hhf = self.hhf + atmp*htmp

        self.iden = sparse.identity(self.m, format="csr")

#------------------------------------------------------------------------------#

class CombinedHamiltonians:
    '''Build Hamiltonians for the entire Hilbert space
    contains

    m                   Combined m
    hx, hy, hz, hhf     Components fo overall Hamiltonian
    ps, pt              Singlet and Triplet projection operators
    hjd                 Coupling Hamiltonian term
    rho0                t=0 density matrix
    K                   Recombination operator
    Svectors            Singlet states

    all matrices in sparse csr format
'''

    def __init__(self, parameters, hA, hB):
        '''Combine Hamiltonians from radicals A and B into
        full Hilbert space representation
'''
        self.m = hA.m * hB.m

        self.hx = sparse.kron(hA.hx, hB.iden) + sparse.kron(hA.iden, hB.hx)
        self.hy = sparse.kron(hA.hy, hB.iden) + sparse.kron(hA.iden, hB.hy)
        self.hz = sparse.kron(hA.hz, hB.iden) + sparse.kron(hA.iden, hB.hz)
        self.hhf = sparse.kron(hA.hhf, hB.iden) + sparse.kron(hA.iden, hB.hhf)        

        self.ps = 0.25*sparse.identity(self.m) - (sparse.kron(hA.hx, hB.hx)
                  + sparse.kron(hA.hy, hB.hy)  + sparse.kron(hA.hz, hB.hz))
        self.pt = sparse.identity(self.m) - self.ps

        self.hx.asformat("csr")
        self.hy.asformat("csr")
        self.hz.asformat("csr")
        self.hhf.asformat("csr")

        self.ps.asformat("csr")
        self.pt.asformat("csr")

        self.rho0 = self.ps * 4.0/float(self.m)
        self.K = parameters.kS*0.5*self.ps + parameters.kT*0.5*self.pt

        self.hjd = sparse.csr_matrix((self.m,self.m), dtype=complex)

        #Find singlet states as eigenvectors of singlet projection operator
        _, self.Svectors = linalg.eigh(self.ps.todense(), eigvals = 
                                       (int(self.m-(self.m/4)),int(self.m-1)))

    def build_jd(self, parameters, hA, hB):
        '''Build electron coupling Hamiltonian terms
'''
        if (hA.n + hB.n > 8):
            raise Exception("Too many spins! Try <9 nuclei")
        self.hjd += -2.0*parameters.J*(sparse.kron(hA.hx, hB.hx)
                                       + sparse.kron(hA.hy, hB.hy)
                                       + sparse.kron(hA.hz, hB.hz))

        D_tensor = np.zeros([3,3], dtype=float)
        D_tensor = np.array([[2.0*np.sin(parameters.D_epsilon)
                               *np.sin(parameters.D_epsilon) - 2.0/3.0,
                               0.0, np.sin(2.0*parameters.D_epsilon)],
                              [0.0,-2.0/3.0,0.0],
                              [np.sin(2.0*parameters.D_epsilon), 0.0, 
                               2.0*np.cos(parameters.D_epsilon)
                               *np.cos(parameters.D_epsilon) - 2.0/3.0]])
        D_tensor = parameters.D*D_tensor

        self.hjd += D_tensor[0,0] * sparse.kron(hA.hx, hB.hx)
        self.hjd += D_tensor[0,1] * sparse.kron(hA.hx, hB.hy)
        self.hjd += D_tensor[0,2] * sparse.kron(hA.hx, hB.hz)
        self.hjd += D_tensor[1,0] * sparse.kron(hA.hy, hB.hx)
        self.hjd += D_tensor[1,1] * sparse.kron(hA.hy, hB.hy)
        self.hjd += D_tensor[1,2] * sparse.kron(hA.hy, hB.hz)
        self.hjd += D_tensor[2,0] * sparse.kron(hA.hz, hB.hx)
        self.hjd += D_tensor[2,1] * sparse.kron(hA.hz, hB.hy)
        self.hjd += D_tensor[2,2] * sparse.kron(hA.hz, hB.hz)

        self.hjd.asformat("csr")

#------------------------------------------------------------------------------#

class SymmetricUncoupled(Hamiltonians):
    '''Define operations if a static, symmetric, uncoupled calculation
'''

    def transform(self, B0, theta, phi):
        '''Form Hamiltonian for field direction specified by theta and phi 
        and field strength B0 (muT)
        Compute the eigensystem decomposition and transform spin operators 
        into the radical eigenbasis
        outputs self.{e, ev, evi, sx, sy, sz, h}
'''
        gamma_e = 1.76E5
        omega_0 = B0*gamma_e
        #Build Hamiltonian for that field direction
        self.h = omega_0*((self.hx*np.cos(phi)
                           + self.hy*np.sin(phi))
                           * np.sin(theta)
                           + self.hz*np.cos(theta))\
                 + self.hhf

        #Compute eigensystem
        self.e, self.ev = linalg.eigh(self.h.todense(), turbo = True, 
                                      overwrite_a = True)
        self.evi = np.conj(self.ev.T)

        #Transform cartesian spin operators
        self.sx = np.dot(self.evi,self.hx.dot(self.ev))
        self.sy = np.dot(self.evi,self.hy.dot(self.ev))
        self.sz = np.dot(self.evi,self.hz.dot(self.ev))

#------------------------------------------------------------------------------#

class SymmetricApprox(SymmetricUncoupled):
    '''Define operations if a static, symmetric, uncoupled, approx calculation
'''

    def bin_frequencies(self, parameters):
        '''Construct histogram and bin frequencies
        outputs histogram bins self.bins and 
        binned spin correlation tensors self.{rab, r0}
'''
        #Construct histogram
        max_gap = self.e[-1] - self.e[0]+1.0
        delta = (max_gap - parameters.divider_bin)/float(parameters.nhigh_bins)
        self.bins = np.arange(parameters.divider_bin, max_gap, 
                              delta) + delta/2.0
        self.bins = np.concatenate((parameters.low_bins, self.bins))

        #Calculate energy differences in eigensystem
        w_nm, indices = energy_differences(self.e, self.m, self.ngaps)

        #Bin contributions to SC tensors
        self.rab, self.r0 = bin_frequencies(
                                    self.ngaps, self.m, parameters.nlow_bins,
                                    parameters.nhigh_bins, parameters.low_delta,
                                    delta, parameters.divider_bin,
                                    w_nm, indices, self.sx, self.sy, self.sz,
                                    parameters.num_threads
                                    )


#------------------------------------------------------------------------------#

class SymmetricCoupled(CombinedHamiltonians):
    '''Define operations if a static, symmetric, uncoupled calculation
'''
    def transform(self, B0, theta, phi):
        '''Form Hamiltonian for field direction specified by theta and phi 
        and field strength B0 (micro Tesla)
        Compute the eigensystem decomposition and Singlet projection operator
        into the combined eigenbasis
        outputs self.{e, ev, evi, sx, sy, sz, tps, tpt, h}
'''
        gamma_e = 1.76E5
        omega_0 = B0*gamma_e
        self.h = omega_0*((self.hx*np.cos(phi)
                           + self.hy*np.sin(phi))
                           * np.sin(theta)
                           + self.hz*np.cos(theta))\
                 + self.hhf + self.hjd
        self.e, self.ev = linalg.eigh(self.h.todense(), turbo = True,
                                      overwrite_a = True)
        self.evi = np.conj(self.ev.T)
        self.tps = np.dot(self.evi, self.ps.dot(self.ev))
        self.tpt = np.dot(self.evi, self.pt.dot(self.ev))

        self.sx = np.dot(self.evi, self.hx.dot(self.ev))
        self.sy = np.dot(self.evi, self.hy.dot(self.ev))
        self.sz = np.dot(self.evi, self.hz.dot(self.ev))

#------------------------------------------------------------------------------#

class AsymmetricExact(CombinedHamiltonians):
    '''Define operations if a static, asymmetric, exact calculation
'''
    def transform(self, B0, theta, phi):
        '''Form Hamiltonian for field direction specified by theta and phi 
        and field strength B0 (micro Tesla)
        Compute the eigensystem decomposition and transform
        into the combined eigenbasis
        outputs self.{e, trho0, tA, c}
'''
        gamma_e = 1.76E5
        omega_0 = B0*gamma_e
        h = omega_0*((self.hx*np.cos(phi)
                      + self.hy*np.sin(phi))
                      * np.sin(theta)
                      + self.hz*np.cos(theta))\
            + self.hhf + self.hjd

        self.c = h - 1.0j*self.K

        self.e, right_ev = linalg.eig(self.c.todense())
        right_evi = linalg.inv(right_ev)
        left_evi = np.conj(right_evi.T)
        left_ev = np.conj(right_ev.T)

        self.trho0 = np.dot(right_evi,self.rho0.dot(left_evi))
        self.tA = np.dot(left_ev,self.ps.dot(right_ev))

#------------------------------------------------------------------------------#

class AsymmetricApprox:
    '''Define operations if a static, asymmetric, approx calculation
'''
    def __init__(self, hA, hB):
        '''Store both radical Hamiltonians that are instances of the
        SymmetricUncoupled class
'''
        self.hA = hA
        self.hB = hB
        self.m = hA.m * hB.m

    def transform(self, B0, theta, phi):
        '''Form separate Hamiltonians for radicals A and B for field direction 
        specified by theta and phi and field strength B0 (micro Tesla)
        Compute the eigensystem decompositions
        Computes hA.transform(...) and hB.transform(...)
        outputs self.{e, esort}
'''
        self.hA.transform(B0, theta, phi)
        self.hB.transform(B0, theta, phi)

        self.e = self.hA.e[None,:] + self.hB.e[:,None]
        self.e = self.e.reshape(1,-1)[0]
        self.esort = np.argsort(self.e)
        self.e = self.e[self.esort]

    def build_degenerate_blocks(self, parameters):
        '''Comstruct nearly degenerate subspaces in the Hamiltonians relative
        to the recombination rates for use in perturbation
        Returns dimension of subspaces (mblock), number of subspaces (nblocks)
        and the corresponding indices in the separate Hamiltonians (indices)
'''
        self.mblock, self.nblocks = degeneracy_check(
                                    self.e, max(parameters.kS, parameters.kT),
                                    parameters.epsilon, self.m
                                    )
        self.indices = get_indices(
                       self.esort, self.hA.m, self.hB.m, self.m
                       )

        if any(mblock>3500 for mblock in self.mblock):
            raise Exception(
                "Degenerate subspaces too large, you need to use a smaller\
                        epsilon or smaller rate constants")

    def calculate_singlet_yield(self, parameters):
        '''Function to calculate the singlet yield using the degenerate
        subspaces and non-Hermitian perturbation theory
'''
        PhiS = 0.0
        imin = 1
        for i in range(self.nblocks):
            pm, rho0, ps = perturbation_matrix(
                            self.hA.sx, self.hB.sx, self.hA.sy, self.hB.sy,
                            self.hA.sz, self.hB.sz, parameters.kS,
                            parameters.kT, self.e, self.mblock[imin-1],
                            self.indices, imin
                            )
            c, right_cv = linalg.eig(pm)
            right_cvi = linalg.inv(right_cv)
            left_cv = np.conj(right_cv.T)
            left_cvi = np.conj(right_cvi.T)
            rho0 = np.dot(right_cvi, np.dot(rho0, left_cvi))
            ps = np.dot(left_cv, np.dot(ps, right_cv))
            PhiStmp = sy_asymmetric(rho0, ps, c, parameters.kS, 
                                    self.mblock[imin-1], 
                                    parameters.num_threads)
            PhiS += PhiStmp
            imin += self.mblock[imin-1]

        return PhiS


#------------------------------------------------------------------------------#

class KMC(SymmetricCoupled):
    '''Define operations for a KMC calculation
'''

#------------------------------------------------------------------------------#

class FloquetUncoupledBroadband(SymmetricUncoupled):
    '''Define operations for a floquet, uncoupled, broadband calculation
'''

    def floquet_matrices(self, parameters, B1, wrf_min, wrf_max, 
                          phase=0.0, theta_rf=0.0, phi_rf=0.0, wrf_0=1.0E3):
        '''Build effective Floquet matrices and diagonalise to find first
        order corrections
        Field strength B1, frequency spacing wrf_0, frequency min and
        max wrf_min, wrf_max, phases and relative direction
        Returns e_floquet, A{x,y,z}_floquet and rho0{x,y,z}_floquet
'''

        nw = int(np.ceil(wrf_max/wrf_0) - np.floor(wrf_min/wrf_0))
        try:
            B1[0]
        except TypeError:
            B1 = B1*np.ones(nw, dtype=float)
        try:
            phase[0]
        except TypeError:
            phase = phase*np.ones(nw, dtype=float)
        try:
            theta_rf[0]
        except TypeError:
            theta_rf = theta_rf*np.ones(nw, dtype=float)
        try:
            phi_rf[0]
        except TypeError:
            phi_rf = phi_rf*np.ones(nw, dtype=float)

        gamma_e = 1.76E2
        omega_1 = B1 * gamma_e/2.0
        h_floquet, Ax_floquet, Ay_floquet, \
        Az_floquet, rho0x_floquet, rho0y_floquet, \
        rho0z_floquet \
        = broadband_build_matrix(
                               self.m, self.e,
                               self.sx, self.sy, self.sz, parameters.kS,
                               omega_1, phase, theta_rf, phi_rf, 
                               wrf_0, wrf_min, wrf_max
                               )

        self.e_floquet, ev = linalg.eigh(
                                h_floquet, 
                                turbo=True, 
                                overwrite_a=True
                                )
        evi = np.conj(ev.T)

        self.Ax_floquet = np.dot(evi, np.dot(Ax_floquet, ev))	
        self.Ay_floquet = np.dot(evi, np.dot(Ay_floquet, ev))	
        self.Az_floquet = np.dot(evi, np.dot(Az_floquet, ev))	

        self.rho0x_floquet = np.dot(evi, np.dot(rho0x_floquet, ev))	
        self.rho0y_floquet = np.dot(evi, np.dot(rho0y_floquet, ev))	
        self.rho0z_floquet = np.dot(evi, np.dot(rho0z_floquet, ev))	

#------------------------------------------------------------------------------#

class FloquetUncoupledSingleFrequency(FloquetUncoupledBroadband):
    '''Define operations for a floquet, uncoupled, single frequency calculation
'''

    def transform(self, B0, theta, phi, B1, theta_rf, phi_rf):
        '''Transform perturbation Hamiltonian with perturbation strength B1 (nT)
        and relative direction defined by theta_rf and phi_rf
        Returns h1 and th1
'''
        super().transform(B0, theta, phi)
        gamma_e = 1.76E2
        omega_1 = B1 * gamma_e / 2.0
        self.h1 = omega_1*((self.hx*np.cos(phi+phi_rf)
                            + self.hy*np.sin(phi+phi_rf))
                            * np.sin(theta+theta_rf)
                            + self.hz*np.cos(theta+theta_rf))
        self.th1 = np.dot(self.evi,self.h1.dot(self.ev))

    def floquet_matrices(self, parameters, wrf, phase):
        '''Build effective Floquet matrices and diagonalise to find first
        order corrections
        Returns e_floquet, A{x,y,z}_floquet and rho0{x,y,z}_floquet
'''

        h_floquet, Ax_floquet, Ay_floquet, \
        Az_floquet, rho0x_floquet, rho0y_floquet, \
        rho0z_floquet \
        = single_frequency_build_matrix(
                                  self.m, self.e, self.sx, self.sy, 
                                  self.sz, self.th1, parameters.kS, 
                                  wrf, phase, parameters.epsilon
                                 )

        self.e_floquet, ev = linalg.eigh(
                                h_floquet, 
                                turbo=True, 
                                overwrite_a=True
                                )
        evi = np.conj(ev.T)

        self.Ax_floquet = np.dot(evi, np.dot(Ax_floquet, ev))	
        self.Ay_floquet = np.dot(evi, np.dot(Ay_floquet, ev))	
        self.Az_floquet = np.dot(evi, np.dot(Az_floquet, ev))	

        self.rho0x_floquet = np.dot(evi, np.dot(rho0x_floquet, ev))	
        self.rho0y_floquet = np.dot(evi, np.dot(rho0y_floquet, ev))	
        self.rho0z_floquet = np.dot(evi, np.dot(rho0z_floquet, ev))	

#------------------------------------------------------------------------------#

class FloquetCoupledBroadband(SymmetricCoupled):
    '''Define operations for a floquet, coupled, broadband calculation
'''

    def floquet_matrices(self, parameters, B1, wrf_min, wrf_max, 
                         phase=0.0, theta_rf=0.0, phi_rf=0.0, wrf_0=1.0E3):
        '''Build effective Floquet matrices and diagonalise to find first
        order corrections
        Field strength B1, frequency spacing wrf_0, frequency min and
        max wrf_min, wrf_max, phases and relative direction
        Returns e_floquet, A_floquet and rho0_floquet
'''

        nw = int(np.ceil(wrf_max/wrf_0) - np.floor(wrf_min/wrf_0))
        try:
            B1[0]
        except TypeError:
            B1 = B1*np.ones(nw, dtype=float)
        try:
            phase[0]
        except TypeError:
            phase = phase*np.ones(nw, dtype=float)
        try:
            theta_rf[0]
        except TypeError:
            theta_rf = theta_rf*np.ones(nw, dtype=float)
        try:
            phi_rf[0]
        except TypeError:
            phi_rf = phi_rf*np.ones(nw, dtype=float)

        gamma_e = 1.76E2
        omega_1 = B1 * gamma_e/2.0
        h_floquet, A_floquet, rho0_floquet \
        = broadband_build_matrix_combined(
                                  self.m, self.e,
                                  self.sx, self.sy, self.sz, self.tps, 
                                  parameters.kS, omega_1, phase,
                                  theta_rf, phi_rf, wrf_0, wrf_min,
                                  wrf_max
                                  )

        self.e_floquet, ev = linalg.eigh(
                                h_floquet, 
                                turbo=True, 
                                overwrite_a=True
                                )
        evi = np.conj(ev.T)

        self.A_floquet = np.dot(evi, np.dot(A_floquet, ev))	
        self.rho0_floquet = np.dot(evi, np.dot(rho0_floquet, ev))	

#------------------------------------------------------------------------------#

class FloquetCoupledSingleFrequency(FloquetCoupledBroadband):
    '''Define operations for a floquet, coupled, single frequency calculation
'''

    def transform(self, B0, theta, phi, B1, theta_rf, phi_rf):
        '''Transform perturbation Hamiltonian with perturbation strength B1 (nT)
        and relative direction defined by theta_rf and phi_rf
        Returns h1 and th1
'''
        super().transform(B0, theta, phi)
        gamma_e = 1.76E2
        omega_1 = B1 * gamma_e/2.0
        self.h1 = omega_1*((self.hx*np.cos(phi+phi_rf)
                            + self.hy*np.sin(phi+phi_rf))
                            * np.sin(theta+theta_rf)
                            + self.hz*np.cos(theta+theta_rf))
        self.th1 = np.dot(self.evi,self.h1.dot(self.ev))


    def floquet_matrices(self, parameters, wrf, phase):
        '''Build effective Floquet matrices and diagonalise to find first
        order corrections
        Returns e_floquet, A_floquet and rho0_floquet
'''

        h_floquet, A_floquet, rho0_floquet \
        = single_frequency_build_matrix_combined(
                                     self.m, self.e, self.tps, self.th1, 
                                     parameters.kS, wrf, 
                                     phase, parameters.epsilon
                                    )

        self.e_floquet, ev = linalg.eigh(
                                h_floquet, 
                                turbo=True, 
                                overwrite_a=True
                                )
        evi = np.conj(ev.T)

        self.A_floquet = np.dot(evi, np.dot(A_floquet, ev))	
        self.rho0_floquet = np.dot(evi, np.dot(rho0_floquet, ev))	

#------------------------------------------------------------------------------#

class GammaComputeSeparate(Hamiltonians):
    '''Define operations in the separate radical Hilbert spaces for a 
    gamma-compute calculation, that are combined later
'''

    def transform(self, B0, theta, phi, B1, theta_rf, phi_rf):
        '''Form Hamiltonian for field direction specified by theta and phi 
        and field strength B0 (muT) and the perturbation Hamiltonian
        with perturbation strength B1 (nT) and relative direction defined
        by theta_rf and phi_rf and frequency w_rf
        outputs h0 and h1
'''
        gamma_e = 1.76E5
        omega_0 = B0*gamma_e
        self.h0 = omega_0*((self.hx*np.cos(phi)
                            + self.hy*np.sin(phi))
                            * np.sin(theta)
                            + self.hz*np.cos(theta))\
                  + self.hhf
        omega_1 = B1*gamma_e*1.0E-3
        self.h1 = omega_1*((self.hx*np.cos(phi+phi_rf)
                            + self.hy*np.sin(phi+phi_rf))
                            * np.sin(theta+theta_rf)
                            + self.hz*np.cos(theta+theta_rf))


    def build_propagator(self, parameters, w_rf):
        '''Build propagator for each time step in a period of the
        periodic Hamiltonian
'''
        self.tau = 2.0*np.pi/(parameters.nt*w_rf)
        self.T = parameters.nt*self.tau

        self.propagator = np.zeros([parameters.nt, self.m, self.m],
                                   dtype=complex)
        for i in range(parameters.nt):
            htmp = self.h0 + self.h1 * np.sin(w_rf*(i+0.5)*self.tau)
            self.propagator[i] = linalg.expm(-1.0j*self.tau*htmp.todense())
            if (i > 0):
                self.propagator[i] = np.dot(self.propagator[i],self.propagator[i-1])

#------------------------------------------------------------------------------#

class GammaCompute(CombinedHamiltonians):
    '''Define combined system for gamma compute calculations and the
    method to calculate PhiS
'''

    def __init__(self, parameters, hA, hB):
        '''Initialise with separate radical Hamiltonians'''

        super().__init__(parameters, hA, hB)
        self.hA = hA
        self.hB = hB

    def transform(self, B0, theta, phi, B1, theta_rf, phi_rf):
        '''Build separate radical Hamiltonians'''

        self.hA.transform(B0, theta, phi, B1, theta_rf, phi_rf)
        self.hB.transform(B0, theta, phi, B1, theta_rf, phi_rf)
        self.flag = True

    def build_propagator(self, parameters, w_rf):
        '''Build separate radical propagators and combine them, and then
        diagonalise the final propagator
'''

        self.w_rf = w_rf

        self.hA.build_propagator(parameters, w_rf)
        self.hB.build_propagator(parameters, w_rf)
        self.tau = self.hA.tau

        self.propagator = np.zeros([parameters.nt, self.m, self.m],
                                    dtype=complex)
        self.propagatorH = np.zeros([parameters.nt, self.m, self.m],
                                     dtype=complex)
        for i in range(parameters.nt):
            self.propagator[i] = np.kron(self.hA.propagator[i],
                                         self.hB.propagator[i])
            self.propagatorH[i] = np.conj(self.propagator[i].T)

        self.Lambda, self.X = linalg.eig(self.propagator[-1])
        self.X = complexgramschmidt(parameters.num_threads, self.m, self.X)
        self.XT = np.conj(self.X.T)

    def calculate_singlet_yield(self, parameters):
        '''Define calculation steps'''

        tps = np.zeros([parameters.nt, self.m, self.m], dtype=complex)
        g_rs = np.zeros([parameters.nt, self.m, self.m], dtype=complex)
        omega_rs = get_omega_rs(parameters.num_threads, self.m,
                                self.w_rf, self.Lambda)

        for i in range(parameters.nt):
            tps[i] = np.dot(self.XT, 
                            np.dot(self.propagatorH[i],
                                   np.dot(self.ps.todense(),
                                          np.dot(self.propagator[i],self.X))))
            g_rs[i] = tps[i]*np.exp(-1.0j*float(i)*self.tau*omega_rs)

        G_rs = np.fft.fft(g_rs, axis=0)
        J = np.conj(G_rs)*G_rs

        PhiS = sy_gamma_compute(
                    parameters.num_threads, self.m, parameters.nt,
                    np.real(J), np.real(omega_rs), self.w_rf, 
                    parameters.kS
                    )

        return np.sum(PhiS) * (4.0/(self.m*parameters.nt**2.0))

#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#

def soperators(s):
    '''Build cartesian S operators for a general value of S
    output: sx, sy, sz as sparse csr matrices
'''
    M = int(2*s+1)
    splus = sparse.lil_matrix((M,M), dtype=complex)
    sminus = sparse.lil_matrix((M,M), dtype=complex)
    for i in range(M-1):
        ms = float(s-(i+1))
        element = np.sqrt(s*(s+1) - ms*(ms+1))
        splus[i,i+1] = element
        sminus[i+1,i] = element
            
    splus.asformat("csr")
    sminus.asformat("csr")
    
    sx = 0.5*(splus+sminus)
    sy = (splus-sminus)/(2.0j)

    sz = sparse.lil_matrix((M,M), dtype=complex)
    for i in range(M):
        ms = float(s - i)
        sz[i,i] = ms

    sz.asformat("csr")

    return sx, sy, sz

def getsoperators(mmax):
    '''Get spin operators for for all values of 2S+1 up to mmax
    output a is an array a[i,j], i being value of 2S+1 and j being x, y, z
'''
    a = np.empty([mmax-1,3], dtype=object)
    for i in range(1,mmax):
        a[i-1,0], a[i-1,1], a[i-1,2] = soperators(i/2.0)

    return a

#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#

