from numba import jit, prange
from numba import complex128, float64, int64
from numba.types import Tuple
import numpy as np

#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#

@jit(Tuple((float64[:],int64[:,:]))(float64[:],int64,int64),
     nopython=True, cache=True, parallel=True)
def energy_differences(e, m, n):
    wnm = np.zeros(n, dtype=np.float64)
    lp = np.zeros((n,2), dtype=np.int64)
    for i in prange(m):
        for j in prange(i+1,m):
            c = int(i*m - (i/2.0)*(i+1) + j-(i+1))
            wnm[c] = e[j]-e[i]
            lp[c,0] = j
            lp[c,1] = i
    return wnm, lp

#------------------------------------------------------------------------------#

@jit(Tuple((complex128[:,:],complex128[:]))(
                int64,int64,int64,int64,float64,float64,float64,float64[:],
                int64[:,:],complex128[:,:],complex128[:,:],complex128[:,:]),
     nopython=True, cache=True, parallel=True)
def bin_frequencies(n, m, nlw, nuw, dl, du, wmax, wnw, lp, sx, sy, sz):
    rab = np.zeros((nlw+nuw,9), dtype=np.complex128)
    r0 = np.zeros(9, dtype=np.complex128)
    for i in prange(n):
        rabtmp = np.zeros((nlw+nuw,9), dtype=np.complex128)
        p1 = lp[i,0]
        p2 = lp[i,1]
        sxtmp = sx[p1,p2]
        sytmp = sy[p1,p2]
        sztmp = sz[p1,p2]
        ind = int(wnw[i]/dl)
        if ind <= nlw-1:
            pass
        else:
            ind = int((wnw[i]-wmax)/du)+nlw
        rabtmp[ind,0] += sxtmp*np.conj(sxtmp)
        rabtmp[ind,1] += sytmp*np.conj(sxtmp)
        rabtmp[ind,2] += sztmp*np.conj(sxtmp)
        rabtmp[ind,3] += sxtmp*np.conj(sytmp)
        rabtmp[ind,4] += sytmp*np.conj(sytmp)
        rabtmp[ind,5] += sztmp*np.conj(sytmp)
        rabtmp[ind,6] += sxtmp*np.conj(sztmp)
        rabtmp[ind,7] += sytmp*np.conj(sztmp)
        rabtmp[ind,8] += sztmp*np.conj(sztmp)

        rab += rabtmp

    for i in range(m):
        r0[0] += sx[i,i]*sx[i,i]
        r0[1] += sy[i,i]*sx[i,i]
        r0[2] += sz[i,i]*sx[i,i]
        r0[3] += sx[i,i]*sy[i,i]
        r0[4] += sy[i,i]*sy[i,i]
        r0[5] += sz[i,i]*sy[i,i]
        r0[6] += sx[i,i]*sz[i,i]
        r0[7] += sy[i,i]*sz[i,i]
        r0[8] += sz[i,i]*sz[i,i]

    return rab, r0


#------------------------------------------------------------------------------#

@jit(complex128(int64,float64[:],complex128[:,:],float64),
     nopython=True, parallel=True, cache=True)
def sy_symmetric_combined(m, e, tps, k):
    c0 = 0.0 + 0.0j
    for i in prange(m):
        for j in prange(m):
            de = k + 1.0j*(e[i]-e[j])
            c0 += k/de*tps[i,j]*tps[j,i]
    return c0

#------------------------------------------------------------------------------#

@jit(complex128(int64,int64,float64,float64[:],float64[:],complex128[:,:],
                complex128[:,:],complex128[:,:],complex128[:,:],complex128[:,:],
                complex128[:,:]),
     nopython=True, parallel=True, cache=True)
def sy_symmetric_separable(ma, mb, k, ea, eb, sxa, sxb, sya, syb, sza, szb):
    c0 = 0.0 + 0.0j
    for i in prange(ma):
        for j in prange(ma):
            for m in prange(mb):
                for l in prange(mb):
                    de = k+1.0j*(ea[i]-ea[j]+eb[m]-eb[l])
                    ps=sxa[i,j]*sxb[m,l]+sya[i,j]*syb[m,l]+sza[i,j]*szb[m,l]
                    c0 += k/de*ps*np.conj(ps)
    return c0

#------------------------------------------------------------------------------#

@jit(Tuple((float64,complex128[:,:],complex128[:,:]))(
                float64[:],float64[:],float64,float64,complex128[:,:],
                complex128[:,:],float64,complex128[:,:],complex128[:,:]),
     nopython=True, cache=True)
def evolve(a, b, wrf, phase, q, p, t, h, h1):
    q1 = q + a[0]*p
    p1 = p
    t1 = t + a[0]
    for i in range(1,4):
        arg = wrf*t1 + phase
        htmp = h + h1*np.sin(arg)
        dhdt = wrf*np.cos(arg)*h1

        w = np.dot(htmp,htmp) + 1.0j*dhdt
        p1 -= b[i]*np.dot(w,q1)
        q1 += a[i]*p1

        t1 += a[i]
    return t1, q1, p1

#------------------------------------------------------------------------------#

@jit(float64[:,:](complex128[:,:],int64), nopython=True, parallel=True,
     cache=True)
def calc_rab(q, m):
    rab = np.zeros((3,3), dtype=np.float64)

    rxx = np.complex128(0.0)
    ryy = np.complex128(0.0)
    rzz = np.complex128(0.0)
    rxz = np.complex128(0.0)
    rzx = np.complex128(0.0)
    for i in prange(m):
        iprime = i+m
        for j in prange(m):
            jprime = j+m
            rxx += (np.conj(q[i,jprime])*q[iprime,j] +
                    np.conj(q[i,j])*q[iprime,jprime])

            ryy += (np.conj(q[iprime,jprime])*q[i,j] - 
                    np.conj(q[iprime,j])*q[i,jprime])

            rzz += (np.conj(q[i,j])*q[i,j] + 
                    np.conj(q[iprime,jprime])*q[iprime,jprime] - 
                    np.conj(q[i,jprime])*q[i,jprime] - 
                    np.conj(q[iprime,j])*q[iprime,j])

            rxz += (np.conj(q[i,jprime])*q[i,j] - 
                    np.conj(q[iprime,jprime])*q[iprime,j])

            rzx += (np.conj(q[i,j])*q[iprime,j] -
                    np.conj(q[i,jprime])*q[iprime,jprime])

    rab[0,0] = np.real(rxx)
    rab[1,1] = np.real(ryy)
    rab[2,2] = np.real(rzz)/2.0
    rab[0,1] = np.imag(rxx)
    rab[1,0] = np.imag(ryy)
    rab[0,2] = np.real(rxz)
    rab[2,0] = np.real(rzx)
    rab[1,2] = np.imag(rxz)
    rab[2,1] = np.imag(rzx)

    return rab

#------------------------------------------------------------------------------#

@jit(float64[:,:,:](int64,int64,float64[:],float64[:],float64,float64,
                    complex128[:,:],complex128[:,:],complex128[:,:],
                    complex128[:,:]),
     nopython=True, cache=True)
def spincorr_tensor(m, nt, a, b, wrf, phase, q0, p0, h, h1):
    t = 0
    q = q0
    p = p0
    rab = np.zeros((nt,3,3), dtype=np.float64)
    for i in range(nt):
        t, q, p = evolve(a, b, wrf, phase, q, p, t, h, h1)
        rab[i] = calc_rab(q, m/2)
    rab = rab / m
    return rab

#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#
