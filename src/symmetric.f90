!-----------------------------------------------------------------------------!

      subroutine energy_differences(e, m, n, w_nm, lp)
      
      !take energies e and find energy differences (w_nm)
      !with indices in lp

      complex(8), intent(in) :: e(:)
      integer, intent(in) :: m, n
        
      real(8), intent(out) :: w_nm(n)
      integer, intent(out) :: lp(n,2)

      integer :: i, j, c

      c = 0

      do i = 1, m
          do j = i+1, m
              c = c+1
              w_nm(c) = real(e(j) - e(i))
              lp(c,1) = j
              lp(c,2) = i
          end do
      end do

      end subroutine energy_differences

!-----------------------------------------------------------------------------!

      subroutine bin_frequencies(n, m, nlw, nuw, deltal, deltau, wmax, &
                w_nm, lp, sx, sy, sz, num_cores, rab, r0)

      !bin contributions to spin correlation tensors (rab, r0)
      !depending on frequency (w_nm) with binning parameters
      !nlwi, deltalm deltau, wmax

      complex(8), intent(in) :: sx(:,:), sy(:,:), sz(:,:)
      integer, intent(in) :: nlw, lp(:,:), n, m, num_cores
      real(8), intent(in) :: deltal, deltau, wmax, w_nm(:)

      complex(8), intent(out) :: rab(nlw+nuw,9)
      complex(8), intent(out) :: r0(9)

      complex(8) :: sxtmp, sytmp, sztmp
      integer :: i, j, ind, p1, p2

      call omp_set_num_threads(num_cores)

      rab = (0.00d0, 0.00d0)
      r0 = (0.00d0, 0.00d0)
!$omp parallel do default(shared) private(i, p1, p2, sxtmp, sytmp, sztmp, ind) reduction(+:rab)
      do i = 1, n
          p1 = lp(i,1)
          p2 = lp(i,2)
          sxtmp = sx(p1,p2)
          sytmp = sy(p1,p2)
          sztmp = sz(p1,p2)
          !evaluate bin index this freq corresponds to
          ind = floor(w_nm(i)/deltal)+1
          if (abs(ind) <= nlw) then
            ind = ind
          else
            ind = floor((w_nm(i)-wmax)/deltau)+1+nlw
          end if
          !add contribution to spin correlation tensor
          rab(ind,1) = rab(ind,1) + sxtmp*conjg(sxtmp)
          rab(ind,2) = rab(ind,2) + sytmp*conjg(sxtmp) 
          rab(ind,3) = rab(ind,3) + sztmp*conjg(sxtmp) 
          rab(ind,4) = rab(ind,4) + sxtmp*conjg(sytmp) 
          rab(ind,5) = rab(ind,5) + sytmp*conjg(sytmp) 
          rab(ind,6) = rab(ind,6) + sztmp*conjg(sytmp) 
          rab(ind,7) = rab(ind,7) + sxtmp*conjg(sztmp) 
          rab(ind,8) = rab(ind,8) + sytmp*conjg(sztmp) 
          rab(ind,9) = rab(ind,9) + sztmp*conjg(sztmp) 
      end do
!$omp end parallel do

      !evaluate diagonal elements (zero frequency) of rab
      do i = 1, m
          r0(1) = r0(1) + sx(i,i)*sx(i,i)
          r0(2) = r0(2) + sy(i,i)*sx(i,i)
          r0(3) = r0(3) + sz(i,i)*sx(i,i)
          r0(4) = r0(4) + sx(i,i)*sy(i,i)
          r0(5) = r0(5) + sy(i,i)*sy(i,i)
          r0(6) = r0(6) + sz(i,i)*sy(i,i)
          r0(7) = r0(7) + sx(i,i)*sz(i,i)
          r0(8) = r0(8) + sy(i,i)*sz(i,i)
          r0(9) = r0(9) + sz(i,i)*sz(i,i)
      end do

      end subroutine bin_frequencies

!-----------------------------------------------------------------------------!

      subroutine sy_symmetric_combined(m, e, ps, k, num_cores, &
      c0)

      !calculate exact singlet yield using closed form expression
      !for non-separable RP using state energies (e), singlet
      !project operator (ps) and recombination rate (k)

      integer, intent(in) :: m, num_cores

      complex(8), intent(in) :: e(:)
      complex(8), intent(in) :: ps(:,:)
      real(8), intent(in) :: k

      complex(8), intent(out) :: c0

      complex(8), parameter :: XJ = (0.00d0, 1.00d0)
      complex(8) :: de
      integer :: i, j

      c0 = (0.00d0, 0.00d0)
      call omp_set_num_threads(num_cores)
!$omp parallel do reduction(+:c0) default(shared) private(i, j, de) 
      do i = 1, m
          do j = 1, m
              de = k + XJ*(e(i) - e(j))
              c0 = c0 + k/de*ps(i,j)*ps(j,i)
          end do
      end do
!$omp end parallel do

      end subroutine sy_symmetric_combined

!-----------------------------------------------------------------------------!

      subroutine sy_symmetric_separable(ma, mb, k, e_a, e_b, sxa, sxb, sya, syb, sza ,szb, &
      num_cores, c0)

      !calculate exact singlet yield using closed form expression
      !for separable radicals. Inputs Cartesian spin operators (sxa
      !etc.), radical energies (ea, eb) and recombination rate (k)


      integer, intent(in) :: ma, mb, num_cores
      complex(8), intent(in) :: e_a(:), e_b(:)
      complex(8), intent(in) :: sxa(:,:), sxb(:,:)
      complex(8), intent(in) :: sya(:,:), syb(:,:)
      complex(8), intent(in) :: sza(:,:), szb(:,:)
      real(8), intent(in) :: k

      complex(8), intent(out) :: c0

      complex(8), parameter :: XJ = (0.00d0, 1.00d0)
      complex(8) :: de, ps
      integer :: i, j, l, m

      c0 = (0.00d0, 0.00d0)
      call omp_set_num_threads(num_cores)
!$omp parallel do reduction(+:c0) default(shared) private(i, j, m, l, de, ps) 
      do  i = 1, ma
          do j = 1, ma
               do m = 1, mb
                   do l = 1, mb
                       de = k + XJ*(e_a(i)-e_a(j)+e_b(m)-e_b(l))
                       ps = sxa(i,j)*sxb(m,l) + &
                            sya(i,j)*syb(m,l) + sza(i,j)*szb(m,l)
                       c0 = c0 + k/de*ps*conjg(ps)
                   end do
               end do
          end do
      end do
!$omp end parallel do

      end subroutine sy_symmetric_separable

!-----------------------------------------------------------------------------!

      subroutine sy_symmetric_approx(ma, mb, k, ea, eb, sxa, sxb, sya, syb, sza ,szb, tol, &
      num_cores, c0)

      !calculate approximate singlet yield by only considering
      !energy pairs with energy gap smaller than some threshold
      !Inputs Cartesian spin operators (sxa etc.), radical energies 
      !(ea, eb) and recombination rate (k)

      integer, intent(in) :: ma, mb, num_cores

      complex(8), intent(in) :: sxa(:,:), sxb(:,:)
      complex(8), intent(in) :: sya(:,:), syb(:,:)
      complex(8), intent(in) :: sza(:,:), szb(:,:)
      complex(8), intent(in) :: ea(:), eb(:)
      real(8), intent(in) :: k, tol
      complex(8), intent(out) :: c0

      complex(8), parameter :: XJ = (0.00d0, 1.00d0)
      complex(8) :: ps, de
      integer :: i, m, ii, counter, ca, cb, j, tria, trib
      integer, allocatable :: pa(:,:), pb(:,:)

      call omp_set_num_threads(num_cores)

      !initiate arrays, pa/pb being arrays of close energy level indices
      tria = (ma*ma-ma)/2-ma
      trib = (mb*mb-mb)/2-mb
      allocate( pa(tria,2), pb(trib,2) )
      pa = 0
      pb = 0
      ca = 0
      cb = 0
      c0 = (0.00d0, 0.00d0)
      !calculate diagonal term contributions
!$omp parallel do default(shared) private(i, j, ps) reduction(+:c0)
      do i = 1, ma
               do m = 1, mb
                       ps = sxa(i,i)*sxb(m,m) + &
                            sya(i,i)*syb(m,m) + sza(i,i)*szb(m,m)
                       c0 = c0 + ps*conjg(ps)
               end do
      end do
!$omp end parallel do

      !add contributions from close energy levels in radical a
      do ii = 1, ma-1
              counter = ca
              do i = 1, ma-ii
                      de = ea(i+ii)-ea(i)
                      if (abs(de) < tol*k) then
                        ca = ca + 1
                        pa(ca,1) = i
                        pa(ca,2) = i+ii
                        de = k + XJ*de
                        de = k/de
                        do j = 1, mb
                        ps = sxa(i,i+ii)*sxb(j,j) + sya(i,i+ii)*syb(j,j) + &
                           sza(i,i+ii)*szb(j,j)
                        c0 = c0 + 2.00d0*de*ps*conjg(ps)
                        end do
                      end if
              end do
              if (counter == ca) exit
      end do

      !add contributions from close energy levels in radical b
      do ii = 1, mb-1
              counter = cb
              do i = 1, mb-ii
                    de = eb(i+ii)-eb(i)
                      if (abs(de) < tol*k) then
                        cb = cb + 1
                        pb(cb,1) = i
                        pb(cb,2) = i+ii
                        de = k + XJ*de
                        de = k/de
                        do j = 1, ma
                        ps = sxb(i,i+ii)*sxa(j,j) + syb(i,i+ii)*sya(j,j) + &
                           szb(i,i+ii)*sza(j,j)
                        c0 = c0 + 2.00d0*de*ps*conjg(ps)
                        end do
                      end if
              end do
              if (counter == cb) exit
      end do

      !add contributions from combination close energy levels
      if (ca /= 0 .and. cb /= 0) then
      do i = 1, ca
              do j = 1, cb
                de = k + XJ*(ea(pa(i,2))+eb(pb(j,1))-&
                        ea(pa(i,1))-eb(pb(j,2)))
                ps = sxa(pa(i,1),pa(i,2))*sxb(pb(j,2),pb(j,1)) + &
                     sya(pa(i,1),pa(i,2))*syb(pb(j,2),pb(j,1)) + &
                     sza(pa(i,1),pa(i,2))*szb(pb(j,2),pb(j,1))
                c0 = c0 + 2.00d0*k/de*ps*conjg(ps)
                de = k + XJ*(ea(pa(i,2))+eb(pb(j,2))-&
                        ea(pa(i,1))-eb(pb(j,1)))
                ps = sxa(pa(i,1),pa(i,2))*sxb(pb(j,1),pb(j,2)) + &
                     sya(pa(i,1),pa(i,2))*syb(pb(j,1),pb(j,2)) + &
                     sza(pa(i,1),pa(i,2))*szb(pb(j,1),pb(j,2))
                c0 = c0 + 2.00d0*k/de*ps*conjg(ps)
              end do
      end do
      end if

      deallocate( pa, pb )

      end subroutine sy_symmetric_approx

!-----------------------------------------------------------------------------!

      subroutine sy_symmetric_spincorr(raba, rabb, r0a, r0b, w_bina, w_binb, &
                tol, k, nw, num_cores, c0)

      !calculate approximate singlet yield using binning method
      !by discretising in frequency space of spin correlation tensors

      complex(8), intent(in) :: raba(:,:), rabb(:,:)
      complex(8), intent(in) :: r0a(:), r0b(:)
      real(8), intent(in) :: w_bina(:), w_binb(:), k, tol
      integer, intent(in) :: nw, num_cores

      complex(8), intent(out) :: c0

      complex(8) :: xj = (0.00d0, 1.00d0)
      real(8) :: sc, sca, scb
      integer :: i, j, ntmp, ntmp1

      c0 = (0.00d0, 0.00d0)

      call omp_set_num_threads(num_cores)

      !calculate zero frequency contribution
      do i = 1, 9
          c0 = c0 + r0a(i)*r0b(i)
      end do

      !calculate contribution from diagonal in b and off-diag in a
      do i = 1, nw
          if (w_bina(i) < tol*k) then
          sca = k*k/(k*k + w_bina(i)*w_bina(i))
          do l = 1, 9
              c0 = c0 + 2.00d0*sca*raba(i,l)*r0b(l)
          end do
          else
            exit
          end if
      end do

      !calculate contribution from diagonal in a and off-diag in b
      do i = 1, nw
          if (w_binb(i) < tol*k) then
          scb = k*k/(k*k + w_binb(i)*w_binb(i))
          do l = 1, 9
              c0 = c0 + 2.00d0*scb*rabb(i,l)*r0a(l)
          end do
          else
            exit
          end if
      end do
       
      !calculate contribution from off-diag terms in both a and b
!$omp parallel do default(shared) private(i, j, sca, scb) reduction(+:c0)
      do i = 1, nw
          do j = 1, nw
              sca = k*k/(k*k + (w_bina(i)+w_binb(j))*(w_bina(i)+w_binb(j)))
              scb = k*k/(k*k + (w_bina(i)-w_binb(j))*(w_bina(i)-w_binb(j)))
              do l = 1, 9
                  c0 = c0 + 2.00d0*sca*raba(i,l)*rabb(j,l)
                  c0 = c0 + 2.00d0*scb*raba(i,l)*conjg(rabb(j,l))
              end do
          end do
      end do
!$omp end parallel do

      end subroutine sy_symmetric_spincorr
        
!-----------------------------------------------------------------------------!
