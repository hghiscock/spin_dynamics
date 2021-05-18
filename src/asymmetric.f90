!-----------------------------------------------------------------------------!

      subroutine degeneracy_check(e, k, tol, m, ni, ng)

      !find near degeneracies in radical states, energy (e)
      !relative to recombination rate (k) and some tolerance

      real(8), intent(in) :: e(:), k, tol
      integer, intent(in) :: m

      integer, intent(out) :: ni(m)
      integer, intent(out) :: ng

      real(8) :: de
      integer :: i, j, itmp
      logical :: free(m)

      ng = m
      ni = 1
      free = .true.

      do i = 1, m
          if (free(i)) then
            itmp = i
            do j = i+1, m
              de = e(j) - e(itmp)
              if (de < tol*k) then
                ni(i) = ni(i) + 1
                free(j) = .false.
                ng = ng - 1
                itmp = j
              else
                exit
              end if
            end do
          end if
      end do

      end subroutine degeneracy_check

!-----------------------------------------------------------------------------!

      subroutine get_indices(ep, ma, mb, m, inds)

      !get the indicies of energy array when combining 
      !energy arrays from the two radicals 

      integer, intent(in) :: ep(:), ma, mb, m
      integer, intent(out) :: inds(m, 2)

      integer :: i, itmp

      do i = 1, m
        itmp = ep(i)
        inds(i,2) = itmp/ma
        inds(i,1) = itmp - inds(i,2)*ma
      end do

      inds = inds + 1        

      end subroutine get_indices

!-----------------------------------------------------------------------------!

      subroutine perturbation_matrix(sxa, sxb, sya, syb, sza, szb, ks, kt, e, &
                                     ni, inds, imin, &
                                     pm, rho0, ps)

      !build matrix of nearly degenerate states for perturbative matrix
      !diagonalisation

      complex(8), intent(in) :: sxa(:,:), sxb(:,:)
      complex(8), intent(in) :: sya(:,:), syb(:,:)
      complex(8), intent(in) :: sza(:,:), szb(:,:)
      real(8), intent(in) :: ks, kt, e(:)
      integer, intent(in) :: inds(:,:), ni, imin

      complex(8), intent(out) :: pm(ni,ni), rho0(ni,ni), ps(ni,ni)

      complex(8), parameter :: xj = (0.00d0, 1.00d0)
      complex(8) :: pstmp, pttmp, ktmp
      integer :: i, j, ia, ib, ja, jb, itmp, jtmp

      call omp_set_num_threads(10)

      do i = imin, imin+ni-1
        ia = inds(i,1)
        ib = inds(i,2)
        itmp = i-imin+1
        pstmp = 0.25d0 - (sxa(ia,ia)*sxb(ib,ib) + sya(ia,ia)*syb(ib,ib) &
                + sza(ia,ia)*szb(ib,ib))
        pttmp = 0.75d0 + (sxa(ia,ia)*sxb(ib,ib) + sya(ia,ia)*syb(ib,ib) &
                + sza(ia,ia)*szb(ib,ib))
        ktmp = ks*pstmp*0.50d0 + kt*pttmp*0.50d0
        pm(itmp,itmp) = e(i) - xj*ktmp
        rho0(itmp,itmp) = pstmp
        ps(itmp,itmp) = pstmp
      end do

!$omp parallel do default(shared) private(i,j,ia,ib,ja,jb,pstmp,pttmp,ktmp,jtmp,itmp)
      do i = imin, imin+ni-1
        do j = i+1, imin+ni-1
          ia = inds(i,1)
          ib = inds(i,2)
          ja = inds(j,1)
          jb = inds(j,2)
          pstmp = - (sxa(ia,ja)*sxb(ib,jb) + sya(ia,ja)*syb(ib,jb) &
                  + sza(ia,ja)*szb(ib,jb))
          pttmp = (sxa(ia,ja)*sxb(ib,jb) + sya(ia,ja)*syb(ib,jb) &
                  + sza(ia,ja)*szb(ib,jb))
          ktmp = ks*pstmp*0.50d0 + kt*pttmp*0.50d0
          itmp = i - imin + 1
          jtmp = j - imin + 1
          pm(itmp,jtmp) = -xj*ktmp
          pm(jtmp,itmp) = -xj*conjg(ktmp)
          rho0(itmp,jtmp) = pstmp
          rho0(jtmp,itmp) = conjg(pstmp)
          ps(itmp,jtmp) = pstmp
          ps(jtmp,itmp) = conjg(pstmp)
        end do
      end do        
!$omp end parallel do

      end subroutine perturbation_matrix

!-----------------------------------------------------------------------------!

      subroutine sy_asymmetric(rho0, ps, c, ks, ni, num_cores, c0)

      !Calculate the contribution to the singlet yield of a nearly
      !degenerate block

      complex(8), intent(in) :: rho0(:,:), ps(:,:), c(:)
      real(8), intent(in) :: ks
      integer, intent(in) :: ni
      integer, intent(in) :: num_cores

      complex(8), intent(out) :: c0

      complex(8), parameter :: xj = (0.00d0, 1.00d0)
      integer :: i, j

      call omp_set_num_threads(num_cores)
      c0 = (0.00d0, 0.00d0)

!$omp parallel do default(shared) private(i,j) reduction(+:c0)
      do i = 1, ni
          do j = 1, ni
            c0 = c0 + ks*rho0(i,j)*ps(j,i)/(xj*(c(i) - conjg(c(j))))
          end do
      end do
!$omp end parallel do

      end subroutine sy_asymmetric

!-----------------------------------------------------------------------------!
!-----------------------------------------------------------------------------!
