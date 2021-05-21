import numpy as np
from .hamiltonians import (
        Hamiltonians, CombinedHamiltonians, SymmetricUncoupled,
        SymmetricApprox, SymmetricCoupled, AsymmetricExact,
        AsymmetricApprox, KMC, FloquetUncoupledBroadband,
        FloquetUncoupledSingleFrequency, FloquetCoupledBroadband,
        FloquetCoupledSingleFrequency, GammaComputeSeparate,
        GammaCompute
        )
from .singlet_yield import (
        bin_frequencies, sy_symmetric_combined, sy_symmetric_separable,
        sy_symmetric_approx, sy_symmetric_spincorr, sy_asymmetric,
        sy_floquet, sy_floquet_combined, trajectories
        )

#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#

def build_hamiltonians(parameters, nA, nB, multiplicitiesA, multiplicitiesB,
                       A_tensorA, A_tensorB):
    '''Build Hamiltonians for calculation defined by an instance of
    the Parameters class and the details of the two radicals

    Parameters
    ----------
    parameters : Parameters
        Object containing calculation parameters
    nA : int
        Number of spin-active nuclei in radical A
    nB : int
        Number of spin-active nuclei in radical B
    multiplicitiesA : (M) array_like
        Spin multiplicities of nuclei in radical A
    multiplicitiesB : (M) array_like
        Spin multiplicities of nuclei in radical B
    A_tensorA : (M,3,3) array_like
        Hyperfine coupling tensors of nuclei in radical A in mT
    A_tensorB : (M,3,3) array_like

    Returns
    -------
    hA : Hamiltonian class object
        Hamiltonians for radical A (If radical pair is separable)
    hB : Hamiltonian class object
        Hamiltonians for radical B (If radical pair is separable)
    h : Hamiltonian class object
        Hamiltonians for combined radical pair (If radical pair is not
        separable)       

    Notes
    -----
    A radical pair is not separable if one or more of these applies:
        The radicals are coupled i.e. non-zero J or D
        Asymmetric recombination
        Gamma compute calculation
        KMC calculation
'''
    if parameters.calculation_flag == "static":
        if parameters.symmetric_flag:
            if parameters.coupled_flag:
                hA = Hamiltonians(nA, multiplicitiesA, A_tensorA)
                hB = Hamiltonians(nB, multiplicitiesB, A_tensorB)
                h = SymmetricCoupled(parameters, hA, hB)
                h.build_jd(parameters, hA, hB)
                return h
            else:
                if parameters.approx_flag == "exact":
                    hA = SymmetricUncoupled(nA, multiplicitiesA, A_tensorA)
                    hB = SymmetricUncoupled(nB, multiplicitiesB, A_tensorB)
                elif parameters.approx_flag == "approx":
                    hA = SymmetricApprox(nA, multiplicitiesA, A_tensorA)
                    hB = SymmetricApprox(nB, multiplicitiesB, A_tensorB)
                return hA, hB
        else:
            if parameters.approx_flag == "exact":
                hA = Hamiltonians(nA, multiplicitiesA, A_tensorA)
                hB = Hamiltonians(nB, multiplicitiesB, A_tensorB)
                h = AsymmetricExact(parameters, hA, hB)
                h.build_jd(parameters, hA, hB)
                return h
            elif parameters.approx_flag == "approx":
                hA = SymmetricUncoupled(nA, multiplicitiesA, A_tensorA)
                hB = SymmetricUncoupled(nB, multiplicitiesB, A_tensorB)
                h = AsymmetricApprox(hA, hB)
                return h
    elif parameters.calculation_flag == "floquet":
        if parameters.coupled_flag:
            hA = Hamiltonians(nA, multiplicitiesA, A_tensorA)
            hB = Hamiltonians(nB, multiplicitiesB, A_tensorB)
            if parameters.nfrequency_flag == "single_frequency":
                h = FloquetCoupledSingleFrequency(parameters, hA, hB)
            elif parameters.nfrequency_flag == "broadband":
                h = FloquetCoupledBroadband(parameters, hA, hB)
            h.build_jd(parameters, hA, hB)
            return h
        else:
            if parameters.nfrequency_flag == "single_frequency":
                hA = FloquetUncoupledSingleFrequency(nA, multiplicitiesA, A_tensorA)
                hB = FloquetUncoupledSingleFrequency(nB, multiplicitiesB, A_tensorB)
            elif parameters.nfrequency_flag == "broadband":
                hA = FloquetUncoupledBroadband(nA, multiplicitiesA, A_tensorA)
                hB = FloquetUncoupledBroadband(nB, multiplicitiesB, A_tensorB)
            return hA, hB
    elif parameters.calculation_flag == "gamma_compute":
        hA = GammaComputeSeparate(nA, multiplicitiesA, A_tensorA)
        hB = GammaComputeSeparate(nB, multiplicitiesB, A_tensorB)
        h = GammaCompute(parameters, hA, hB)
        return h
    elif parameters.calculation_flag == "KMC":
        hA = Hamiltonians(nA, multiplicitiesA, A_tensorA)
        hB = Hamiltonians(nB, multiplicitiesB, A_tensorB)
        h = KMC(parameters, hA, hB)
        return h

#------------------------------------------------------------------------------#

def compute_singlet_yield(parameters, hA = None, hB = None, h = None):
    '''Calculate the singlet yield for a radical pair reaction

    Parameters
    ----------
    parameters : Parameters
        Object containing calculation parameters
    hA : Hamiltonian, optional
        Hamiltonians of radical A (if radical pair is separable)
    hB : Hamiltonian, optional
        Hamiltonians of radical B (if radical pair is separable)
    h : Hamiltonian, optional
        Hamiltonians of combined radical pair (if radical pair is not
        separable)

    Returns
    -------
    PhiS : float
        Singlet yield

    Raises
    ------
    You need to transform first!
        If the Hilbert space(s) hasn't been transformed into the eigenbasis
    You need to build degenerate blocks!
        If degenerate subspaces haven't been constructed for the approximate
        asymmetric calculation
    You need to build floquet matrices first!
        If the Floquet matrices haven't been constructed to use the Floquet
        algorithm
    You need to construct the propagator first!
        If the propagator for a time period in a gamma compute calculation
        hasn't been constructed
'''

    if parameters.calculation_flag == "static":
        if parameters.symmetric_flag:
            if parameters.coupled_flag:
                if hasattr(h, 'e'):
                    PhiS = sy_symmetric_combined(
                                h.m, h.e, h.tps, parameters.kS, parameters.num_threads
                                )
                    PhiS = PhiS*4.0/float(h.m)
                else:
                    raise Exception("You need to transform first!")
            else:
                if hasattr(hA, 'e') and hasattr(hB, 'e'):
                    if parameters.approx_flag == "exact":
                        PhiS = sy_symmetric_separable(
                                hA.m, hB.m, parameters.kS, hA.e, hB.e,
                                hA.sx, hB.sx, hA.sy, hB.sy, hA.sz, hB.sz,
                                parameters.num_threads
                                )
                    elif parameters.approx_flag == "approx":
                        if parameters.kS > 1.0E4:
                            hA.bin_frequencies(parameters)
                            hB.bin_frequencies(parameters)
                            PhiS = sy_symmetric_spincorr(
                                    hA.rab, hB.rab, hA.r0, hB.r0, 
                                    hA.bins, hB.bins, parameters.epsilon,
                                    parameters.kS, parameters.nbins,
                                    parameters.num_threads
                                    )
                        else:
                            PhiS = sy_symmetric_approx(
                                    hA.m, hB.m, parameters.kS, hA.e, hB.e,
                                    hA.sx, hB.sx, hA.sy, hB.sy, hA.sz, hB.sz,
                                    parameters.epsilon, parameters.num_threads
                                    )
                    PhiS = PhiS*4.0/float(hA.m*hB.m) + 0.25
                else:
                    raise Exception("You need to transform first!")
        else:
            if parameters.approx_flag == "exact":
                if hasattr(h, 'e'):
                    PhiS = sy_asymmetric(
                            h.trho0, h.tA, h.e, parameters.kS, h.m, 
                            parameters.num_threads
                            )
                else:
                    raise Exception("You need to transform first!")
            elif parameters.approx_flag == "approx":
                if hasattr(h, 'e'):
                    if hasattr(h, 'mblock'):
                        PhiS = h.calculate_singlet_yield(parameters)
                        PhiS = PhiS*4.0/float(h.m)
                    else:
                        raise Exception("You need to build degenerate blocks!")
                else:
                    raise Exception("You need to transform first!")
    elif parameters.calculation_flag == "floquet":
        if parameters.coupled_flag:
            if hasattr(h, 'e'):
                if hasattr(h, 'e_floquet'):
                    PhiS = sy_floquet_combined(
                            h.m, parameters.kS, h.e_floquet, h.A_floquet,
                            h.rho0_floquet, parameters.num_threads
                            )
                    PhiS = PhiS*4.0/float(h.m)
                else:
                    raise Exception(
                        "You need to build floquet matrices first!")
            else:
                raise Exception("You need to transform first!")
        else:
            if hasattr(hA, 'e') and hasattr(hB, 'e'):
                if hasattr(hA, 'e_floquet') and hasattr(hA, 'e_floquet'):
                    PhiS = sy_floquet(
                            hA.m, hB.m, parameters.kS, hA.e_floquet,
                            hB.e_floquet, hA.Ax_floquet, hB.Ax_floquet,
                            hA.Ay_floquet, hB.Ay_floquet, hA.Az_floquet,
                            hB.Az_floquet, hA.rho0x_floquet, hB.rho0x_floquet,
                            hA.rho0y_floquet, hB.rho0y_floquet,
                            hA.rho0z_floquet, hB.rho0z_floquet,
                            parameters.num_threads
                            )
                    PhiS = PhiS*4.0/float(hA.m*hB.m) + 0.25
                else:
                    raise Exception(
                        "You need to build floquet matrices first!")
            else:
                raise Exception("You need to transform first!")
    elif parameters.calculation_flag == "gamma_compute":
        if hasattr(h, 'flag'):
            if hasattr(h, 'w_rf'):
                PhiS = h.calculate_singlet_yield(parameters)
            else:
                raise Exception("You need to construct the propagator first!")
        else:
            raise Exception("You need to transform first!")
    elif parameters.calculation_flag == "KMC": 
        if hasattr(h, 'e'):
           PhiS = trajectories.sy_kmc(
                   h.m, parameters.ntrajectories, h.e, h.evi, h.tps, h.tpt, 
                   parameters.kS, parameters.kT, h.Svectors, 
                   parameters.num_threads
                   )
           PhiS = PhiS/float(parameters.ntrajectories)
        else:
            raise Exception("You need to transform first!")
    return np.real(PhiS)


