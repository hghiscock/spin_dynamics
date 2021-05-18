!-----------------------------------------------------------------------------!
!-----------------------------------------------------------------------------!

      subroutine sy_floquet(ma, mb, k, e_a, e_b, sxa, sxb, sya, syb, sza ,szb, &
                        sxa1, sxb1, sya1, syb1, sza1, szb1, num_cores, c0)

      !calculate the singlet yield from effective matrices

      integer, intent(in) :: ma, mb, num_cores
      complex(8), intent(in) :: e_a(:), e_b(:)
      complex(8), intent(in) :: sxa(:,:), sxb(:,:)
      complex(8), intent(in) :: sya(:,:), syb(:,:)
      complex(8), intent(in) :: sza(:,:), szb(:,:)
      complex(8), intent(in) :: sxa1(:,:), sxb1(:,:)
      complex(8), intent(in) :: sya1(:,:), syb1(:,:)
      complex(8), intent(in) :: sza1(:,:), szb1(:,:)
      real(8), intent(in) :: k

      complex(8), intent(out) :: c0

      complex(8), parameter :: xj = (0.00d0, 1.00d0)
      complex(8) :: de, ps, ps1
      integer :: i, j, l, m

      c0 = (0.00d0, 0.00d0)

      call omp_set_num_threads(num_cores)
!$omp parallel do default(shared) private(i,j,m,l,de,ps,ps1) reduction(+:c0)
      do i = 1, ma
          do j = 1, ma
               do m = 1, mb
                   do l = 1, mb
                       de = k - xj*(e_a(j)-e_a(i)+e_b(l)-e_b(m))
                       ps = sxa(i,j)*sxb(m,l) + &
                            sya(i,j)*syb(m,l) + sza(i,j)*szb(m,l)
                       ps1 = sxa1(j,i)*sxb1(l,m) + sya1(j,i)*syb1(l,m) +&
                             sza1(j,i)*szb1(l,m)
                       c0 = c0 + (k/de)*ps*ps1
                   end do
               end do
          end do
      end do
!$omp end parallel do

      end subroutine sy_floquet

!-----------------------------------------------------------------------------!

      subroutine sy_floquet_combined(m, k, e, ps, ps1, num_cores, &
                        c0)

      !calculate singlet yield from effective matrices in combined
      !Hilbert space

      integer, intent(in) :: m, num_cores

      complex(8), intent(in) :: e(:)
      complex(8), intent(in) :: ps(:,:), ps1(:,:)
      real(8), intent(in) :: k

      complex(8), intent(out) :: c0

      complex(8), parameter :: xj = (0.00d0, 1.00d0)
      complex(8) :: de
      integer :: i, j

      c0 = (0.00d0, 0.00d0)

      call omp_set_num_threads(num_cores)
!$omp parallel do default(shared) private(i,j,de) reduction(+:c0)
      do i = 1, m
          do j = 1, m
              de = k - xj*(e(j)-e(i))
              c0 = c0 + (k/de)*ps(i,j)*ps1(j,i)
          end do
      end do
!$omp end parallel do

      end subroutine sy_floquet_combined

!-----------------------------------------------------------------------------!

      subroutine single_frequency_build_matrix(m, e, sx, sy, sz, h1, k, &
                        w_rf, phase, tol, &
                        h2, sx1, sy1, sz1, sx2, sy2, sz2)

      !build degenerate block of floquet hamiltonian and corresponding 
      !cartesian spin operators

      integer, intent(in) :: m
      real(8), intent(in) :: k, phase, w_rf, tol
      complex(8), intent(in) :: e(:)
      complex(8), intent(in) :: sx(:,:)
      complex(8), intent(in) :: sy(:,:)
      complex(8), intent(in) :: sz(:,:)
      complex(8), intent(in) :: h1(:,:)

      complex(8), intent(out) :: h2(m,m)
      complex(8), intent(out) :: sx1(m,m), sx2(m,m)
      complex(8), intent(out) :: sy1(m,m), sy2(m,m)
      complex(8), intent(out) :: sz1(m,m), sz2(m,m)

      integer :: i, j, c
      complex(8) :: de, de1
      complex(8), parameter :: xj = (0.00d0, 1.00d0)

      !initialise output matrices
      sx1 = (0.00d0, 0.00d0)
      sy1 = (0.00d0, 0.00d0)
      sz1 = (0.00d0, 0.00d0)
      sx2 = (0.00d0, 0.00d0)
      sy2 = (0.00d0, 0.00d0)
      sz2 = (0.00d0, 0.00d0)
      h2 = (0.00d0, 0.00d0)

      !change diagonal elements
      do i = 1, m
          h2(i,i) = e(i)
          sx1(i,i) = sx(i,i)
          sy1(i,i) = sy(i,i)
          sz1(i,i) = sz(i,i)
          sx2(i,i) = sx(i,i)
          sy2(i,i) = sy(i,i)
          sz2(i,i) = sz(i,i)
      end do

      c = 0
      !check for resonances
      do i = 1, m
          do j = i+1, m
              de = e(j) - e(i) - w_rf
              if (abs(de) < 4.0*tol*k) then
                  c = c+1
                  h2(i,j) = h1(i,j)*cdexp(xj*phase)
                  h2(j,i) = h1(j,i)*cdexp(-xj*phase)
                  h2(j,j) = h2(i,i) - e(i) + e(j) - w_rf
                  sx2(i,j) = sx(i,j)
                  sx2(j,i) = sx(j,i)
                  sy2(i,j) = sy(i,j)
                  sy2(j,i) = sy(j,i)
                  sz2(i,j) = sz(i,j)
                  sz2(j,i) = sz(j,i)
              end if
          end do
      end do

      !check for nearly degenerate levels in h0
      do i = 1, m
          do j = i+1, m
              de = e(j) - e(i)
              de1 = h2(j,j) - h2(i,i)
              if (abs(de) < 100.0*k) then
                  c = c+1
                  sx1(i,j) = sx(i,j)
                  sx1(j,i) = sx(j,i)
                  sy1(i,j) = sy(i,j)
                  sy1(j,i) = sy(j,i)
                  sz1(i,j) = sz(i,j)
                  sz1(j,i) = sz(j,i)
                  
                  sx2(i,j) = sx(i,j)
                  sx2(j,i) = sx(j,i)
                  sy2(i,j) = sy(i,j)
                  sy2(j,i) = sy(j,i)
                  sz2(i,j) = sz(i,j)
                  sz2(j,i) = sz(j,i)
                  if (de == de1) then
                    h2(j,j) = h2(i,i) + de
                  else if (real(de1) < 0.00d0) then
                    h2(i,i) = h2(j,j) - de
                  else
                    h2(j,j) = h2(i,i) + de
                  end if
              end if
          end do
      end do

      end subroutine single_frequency_build_matrix

!-----------------------------------------------------------------------------!

      subroutine single_frequency_build_matrix_combined(m, e, ps, h1, k, &
                        w_rf, phase, tol, h2, ps1, ps2)

      !build effective floquet hamiltonian and corresponding 
      !cartesian spin operators

      integer, intent(in) :: m
      real(8), intent(in) :: k, phase, w_rf, tol
      complex(8), intent(in) :: e(:)
      complex(8), intent(in) :: ps(:,:)
      complex(8), intent(in) :: h1(:,:)

      complex(8), intent(out) :: h2(m,m)
      complex(8), intent(out) :: ps1(m,m), ps2(m,m)

      integer :: i, j, c
      complex(8) :: de, de1
      complex(8), parameter :: xj = (0.00d0, 1.00d0)

      !initialise output matrices
      ps1 = (0.00d0, 0.00d0)
      ps2 = (0.00d0, 0.00d0)
      h2 = (0.00d0, 0.00d0)

      !change diagonal elements
      do i = 1, m
          h2(i,i) = e(i)
          ps1(i,i) = ps(i,i)
          ps2(i,i) = ps(i,i)
      end do

      c = 0
      !check for resonances
      do i = 1, m
          do j = i+1, m
              de = e(j) - e(i) - w_rf
              if (abs(de) < 4.0*tol*k) then
                  c = c+1
                  h2(i,j) = h1(i,j)*cdexp(xj*phase)
                  h2(j,i) = h1(j,i)*cdexp(-xj*phase)
                  h2(j,j) = h2(i,i) - e(i) + e(j) - w_rf
                  ps2(i,j) = ps(i,j)
                  ps2(j,i) = ps(j,i)
              end if
          end do
      end do

      !check for nearly degenerate levels in h0
      do i = 1, m
          do j = i+1, m
              de = e(j) - e(i)
              de1 = h2(j,j) - h2(i,i)
              if (abs(de) < 100.0*k) then
                  c = c+1
                  ps1(i,j) = ps(i,j)
                  ps1(j,i) = ps(j,i)
                  
                  ps2(i,j) = ps(i,j)
                  ps2(j,i) = ps(j,i)
                  if (de == de1) then
                    h2(j,j) = h2(i,i) + de
                  else if (real(de1) < 0.00d0) then
                    h2(i,i) = h2(j,j) - de
                  else
                    h2(j,j) = h2(i,i) + de
                  end if
              end if
          end do
      end do

      end subroutine single_frequency_build_matrix_combined

!-----------------------------------------------------------------------------!

      subroutine broadband_build_matrix(m, e, sx, sy, sz, k, w1, phases, thetas, phis, &
                        wrf_0, wrf_min, wrf_max, &
                        h2, sx1, sy1, sz1, sx2, sy2, sz2)

      !build effective floquet Hamiltonian with an array of values of
      !the Fourier index

      integer, intent(in) :: m
      real(8), intent(in) :: k, wrf_0, wrf_min, wrf_max
      real(8), intent(in) :: phases(:), thetas(:), phis(:), w1(:)
      complex(8), intent(in) :: e(:)
      complex(8), intent(in) :: sx(:,:), sy(:,:), sz(:,:)

      complex(8), intent(out) :: h2(m,m)
      complex(8), intent(out) :: sx1(m,m), sy1(m,m), sz1(m,m)
      complex(8), intent(out) :: sx2(m,m), sy2(m,m), sz2(m,m)

      integer :: i, j, lmin, lmax, ltmp, larray(m,m), larraymax(m), ll, z, l1
      real(8) :: de, de1, de_array(m), phi1, theta1
      complex(8) :: th1
      complex(8), parameter :: xj = (0.00d0, 1.00d0)
      logical :: coupled(m,m)

      lmin = floor(wrf_min/wrf_0)
      lmax = ceiling(wrf_max/wrf_0)

      !initialise outpout matrices
      h2 = (0.00d0, 0.00d0)
      sx1 = (0.00d0, 0.00d0)
      sy1 = (0.00d0, 0.00d0)
      sz1 = (0.00d0, 0.00d0)
      sx2 = (0.00d0, 0.00d0)
      sy2 = (0.00d0, 0.00d0)
      sz2 = (0.00d0, 0.00d0)

      de_array = 0.00d0
      larray = 0
      coupled = .false.
      larraymax = 1

      !Populate diagonal elements
      do i = 1, m
        sx1(i,i) = sx(i,i)
        sy1(i,i) = sy(i,i)
        sz1(i,i) = sz(i,i)
        sx2(i,i) = sx(i,i)
        sy2(i,i) = sy(i,i)
        sz2(i,i) = sz(i,i)
      end do

      !Add in coupling terms and shift energies for resonances
      do i = 1, m
        do j = i+1, m
          de = real(e(j) - e(i))
          ltmp = nint(de/wrf_0)
!          if (ltmp >= lmin .and. ltmp <= lmax .and. (de - ltmp*wrf_0) < 10.0d0*k) then
          if (ltmp >= lmin .and. ltmp <= lmax) then
!            if (ltmp>lmax) print *, ltmp-lmin, size(thetas)
            coupled(i,j) = .true.
            ll = larraymax(j)
            de1 = de - dble(ltmp)*wrf_0

            if (i /= 1 .and. coupled(ll,j) .and. coupled(ll,i)) then
              if (ltmp == larray(ll,j) - larray(ll,i)) then        
                h2(j,j) = h2(i,i) + de1
                l1 = ltmp-lmin+1
                phi1 = phis(l1)
                theta1 = thetas(l1)
                th1 = w1(l1)*((sx(i,j)*cos(phi1)+sy(i,j)*sin(phi1))*sin(theta1) &
                      + sz(i,j)*cos(theta1))
                h2(i,j) = th1*exp(xj*phases(l1))
                h2(j,i) = conjg(th1)*exp(-xj*phases(l1))
              else if (abs(de1) < abs(de_array(j))) then
                h2(j,j) = h2(i,i) + de1
                l1 = ltmp-lmin+1
                phi1 = phis(l1)
                theta1 = thetas(l1)
                th1 = w1(l1)*((sx(i,j)*cos(phi1)+sy(i,j)*sin(phi1))*sin(theta1) &
                      + sz(i,j)*cos(theta1))
                h2(i,j) = th1*exp(xj*phases(l1))
                h2(j,i) = conjg(th1)*exp(-xj*phases(l1))
                do z = 1, i-1
                  larray(z,j) = larray(z,i) + ltmp
                  if (larray(z,j) >= lmin .and. larray(z,j) <= lmax) then
                    coupled(z,j) = .true.
                    l1 = larray(z,j)-lmin+1
                    phi1 = phis(l1)
                    theta1 = thetas(l1)
                    th1 = w1(l1)*((sx(z,j)*cos(phi1)+sy(z,j)*sin(phi1))*sin(theta1) &
                          + sz(z,j)*cos(theta1))
                    h2(z,j) = th1*exp(xj*phases(l1))
                    h2(j,z) = conjg(th1)*exp(-xj*phases(l1))
                  else
                    coupled(z,j) = .false.
                    h2(z,j) = (0.00d0, 0.00d0)
                    h2(j,z) = (0.00d0, 0.00d0)
                  end if
                end do
              else
                ltmp = larray(ll,j) - larray(ll,i)
                if (ltmp >= lmin .and. ltmp <= lmax) then
                  de1 = de - dble(ltmp)*wrf_0
                  h2(j,j) = h2(i,i) + de1
                  l1 = ltmp-lmin+1
                  phi1 = phis(l1)
                  theta1 = thetas(l1)
                  th1 = w1(l1)*((sx(i,j)*cos(phi1)+sy(i,j)*sin(phi1))*sin(theta1) &
                        + sz(i,j)*cos(theta1))
                  h2(i,j) = th1*exp(xj*phases(l1))
                  h2(j,i) = conjg(th1)*exp(-xj*phases(l1))
                end if
              end if 
            else
              h2(j,j) = h2(i,i) + de1
              l1 = ltmp-lmin+1
              phi1 = phis(l1)
              theta1 = thetas(l1)
              th1 = w1(l1)*((sx(i,j)*cos(phi1)+sy(i,j)*sin(phi1))*sin(theta1) &
                    + sz(i,j)*cos(theta1))
              h2(i,j) = th1*exp(xj*phases(l1))
              h2(j,i) = conjg(th1)*exp(-xj*phases(l1))
            end if

            sx2(i,j) = sx(i,j)
            sx2(j,i) = sx(j,i)
            sy2(i,j) = sy(i,j)
            sy2(j,i) = sy(j,i)
            sz2(i,j) = sz(i,j)
            sz2(j,i) = sz(j,i)

            if ((i == 1) .or. (abs(de1) < abs(de_array(j)))) then
                de_array(j) = de1
                larraymax(j) = i
            end if

          else
            if (i == 1) then
                de_array(j) = de1
            end if

          end if
          larray(i,j) = ltmp
        end do
      end do

      !Check resonances have preserved near-degeneracies in H0
      do i = 1, m
        do j = i+1, m
          de = real(e(j) - e(i))
          de1 = real(h2(j,j) - h2(i,i))
          if (de < 10.0*k) then
            h2(i,j) = (0.00d0, 0.00d0)
            h2(j,i) = (0.00d0, 0.00d0)

            sx1(i,j) = sx(i,j)
            sx1(j,i) = sx(j,i)
            sy1(i,j) = sy(i,j)
            sy1(j,i) = sy(j,i)
            sz1(i,j) = sz(i,j)
            sz1(j,i) = sz(j,i)

            sx2(i,j) = sx(i,j)
            sx2(j,i) = sx(j,i)
            sy2(i,j) = sy(i,j)
            sy2(j,i) = sy(j,i)
            sz2(i,j) = sz(i,j)
            sz2(j,i) = sz(j,i)

            if (de /= de1) then
              if (de_array(j) > de_array(i)) then 
                h2(j,j) = h2(i,i) + de
                do z = 1, i-1
                  l1 = larray(z,i)
                  if (l1 >= lmin .and. l1 <= lmax) then
                  l1 = l1-lmin+1
                  phi1 = phis(l1)
                  theta1 = thetas(l1)
                  th1 = w1(l1)*((sx(z,j)*cos(phi1)+sy(z,j)*sin(phi1))*sin(theta1) &
                        + sz(z,j)*cos(theta1))
                  h2(z,j) = th1*exp(xj*phases(l1))
                  h2(j,z) = conjg(th1)*exp(-xj*phases(l1))
                  end if
                end do
                do z = j+1, m
                  l1 = larray(i,z)
                  if (l1 >= lmin .and. l1 <= lmax) then
                  l1 = l1-lmin+1
                  phi1 = phis(l1)
                  theta1 = thetas(l1)
                  th1 = w1(l1)*((sx(z,j)*cos(phi1)+sy(z,j)*sin(phi1))*sin(theta1) &
                        + sz(z,j)*cos(theta1))
                  h2(z,j) = th1*exp(xj*phases(l1))
                  h2(j,z) = conjg(th1)*exp(-xj*phases(l1))
                  end if
                end do
              else
                h2(i,i) = h2(j,j) - de
                do z = 1, i-1
                  l1 = larray(z,j)
                  if (l1 >= lmin .and. l1 <= lmax) then
                  l1 = l1-lmin+1
                  phi1 = phis(l1)
                  theta1 = thetas(l1)
                  th1 = w1(l1)*((sx(z,j)*cos(phi1)+sy(z,j)*sin(phi1))*sin(theta1) &
                        + sz(z,j)*cos(theta1))
                  h2(z,i) = th1*exp(xj*phases(l1))
                  h2(i,z) = conjg(th1)*exp(-xj*phases(l1))
                  end if 
                end do
                do z = j+1, m
                  l1 = larray(j,z)
                  if (l1 >= lmin .and. l1 <= lmax) then
                  l1 = l1-lmin+1
                  phi1 = phis(l1)
                  theta1 = thetas(l1)
                  th1 = w1(l1)*((sx(z,j)*cos(phi1)+sy(z,j)*sin(phi1))*sin(theta1) &
                        + sz(z,j)*cos(theta1))
                  h2(z,i) = th1*exp(xj*phases(l1))
                  h2(i,z) = conjg(th1)*exp(-xj*phases(l1))
                  end if
                end do
              end if
            end if
          end if
        end do
      end do

      end subroutine broadband_build_matrix

!-----------------------------------------------------------------------------!

      subroutine broadband_build_matrix_combined(m, e, sx, sy, sz, ps, k, w1, &
                        phases, thetas, phis, wrf_0, wrf_min, wrf_max, &
                        h2, ps1, ps2)

      !build effective floquet Hamiltonian with an array of values of
      !the Fourier index

      integer, intent(in) :: m
      real(8), intent(in) :: k, wrf_0, wrf_min, wrf_max
      real(8), intent(in) :: phases(:), thetas(:), phis(:), w1(:)
      complex(8), intent(in) :: e(:)
      complex(8), intent(in) :: sx(:,:), sy(:,:), sz(:,:)
      complex(8), intent(in) :: ps(:,:)

      complex(8), intent(out) :: h2(m,m)
      complex(8), intent(out) :: ps1(m,m)
      complex(8), intent(out) :: ps2(m,m)

      integer :: i, j, lmin, lmax, ltmp, larray(m,m), larraymax(m), ll, z, l1
      real(8) :: de, de1, de_array(m), phi1, theta1
      complex(8) :: th1
      complex(8), parameter :: xj = (0.00d0, 1.00d0)
      logical :: coupled(m,m)

      lmin = floor(wrf_min/wrf_0)
      lmax = ceiling(wrf_max/wrf_0)

      !initialise outpout matrices
      h2 = (0.00d0, 0.00d0)
      ps1 = (0.00d0, 0.00d0)
      ps2 = (0.00d0, 0.00d0)

      de_array = 0.00d0
      larray = 0
      coupled = .false.
      larraymax = 1

      !Populate diagonal elements
      do i = 1, m
        ps1(i,i) = ps(i,i)
        ps2(i,i) = ps(i,i)
      end do

      !Add in coupling terms and shift energies for resonances
      do i = 1, m
        do j = i+1, m
          de = real(e(j) - e(i))
          ltmp = nint(de/wrf_0)
!          if (ltmp >= lmin .and. ltmp <= lmax .and. (de - ltmp*wrf_0) < 10.0d0*k) then
          if (ltmp >= lmin .and. ltmp <= lmax) then
!            if (ltmp>lmax) print *, ltmp-lmin, size(thetas)
            coupled(i,j) = .true.
            ll = larraymax(j)
            de1 = de - dble(ltmp)*wrf_0

            if (i /= 1 .and. coupled(ll,j) .and. coupled(ll,i)) then
              if (ltmp == larray(ll,j) - larray(ll,i)) then        
                h2(j,j) = h2(i,i) + de1
                l1 = ltmp-lmin+1
                phi1 = phis(l1)
                theta1 = thetas(l1)
                th1 = w1(l1)*((sx(i,j)*cos(phi1)+sy(i,j)*sin(phi1))*sin(theta1) &
                      + sz(i,j)*cos(theta1))
                h2(i,j) = th1*exp(xj*phases(l1))
                h2(j,i) = conjg(th1)*exp(-xj*phases(l1))
              else if (abs(de1) < abs(de_array(j))) then
                h2(j,j) = h2(i,i) + de1
                l1 = ltmp-lmin+1
                phi1 = phis(l1)
                theta1 = thetas(l1)
                th1 = w1(l1)*((sx(i,j)*cos(phi1)+sy(i,j)*sin(phi1))*sin(theta1) &
                      + sz(i,j)*cos(theta1))
                h2(i,j) = th1*exp(xj*phases(l1))
                h2(j,i) = conjg(th1)*exp(-xj*phases(l1))
                do z = 1, i-1
                  larray(z,j) = larray(z,i) + ltmp
                  if (larray(z,j) >= lmin .and. larray(z,j) <= lmax) then
                    coupled(z,j) = .true.
                    l1 = larray(z,j)-lmin+1
                    phi1 = phis(l1)
                    theta1 = thetas(l1)
                    th1 = w1(l1)*((sx(z,j)*cos(phi1)+sy(z,j)*sin(phi1))*sin(theta1) &
                          + sz(z,j)*cos(theta1))
                    h2(z,j) = th1*exp(xj*phases(l1))
                    h2(j,z) = conjg(th1)*exp(-xj*phases(l1))
                  else
                    coupled(z,j) = .false.
                    h2(z,j) = (0.00d0, 0.00d0)
                    h2(j,z) = (0.00d0, 0.00d0)
                  end if
                end do
              else
                ltmp = larray(ll,j) - larray(ll,i)
                if (ltmp >= lmin .and. ltmp <= lmax) then
                  de1 = de - dble(ltmp)*wrf_0
                  h2(j,j) = h2(i,i) + de1
                  l1 = ltmp-lmin+1
                  phi1 = phis(l1)
                  theta1 = thetas(l1)
                  th1 = w1(l1)*((sx(i,j)*cos(phi1)+sy(i,j)*sin(phi1))*sin(theta1) &
                        + sz(i,j)*cos(theta1))
                  h2(i,j) = th1*exp(xj*phases(l1))
                  h2(j,i) = conjg(th1)*exp(-xj*phases(l1))
                end if
              end if 
            else
              h2(j,j) = h2(i,i) + de1
              l1 = ltmp-lmin+1
              phi1 = phis(l1)
              theta1 = thetas(l1)
              th1 = w1(l1)*((sx(i,j)*cos(phi1)+sy(i,j)*sin(phi1))*sin(theta1) &
                    + sz(i,j)*cos(theta1))
              h2(i,j) = th1*exp(xj*phases(l1))
              h2(j,i) = conjg(th1)*exp(-xj*phases(l1))
            end if

            ps2(i,j) = ps(i,j)
            ps2(j,i) = ps(j,i)

            if ((i == 1) .or. (abs(de1) < abs(de_array(j)))) then
                de_array(j) = de1
                larraymax(j) = i
            end if

          else
            if (i == 1) then
                de_array(j) = de1
            end if

          end if
          larray(i,j) = ltmp
        end do
      end do

      !Check resonances have preserved near-degeneracies in H0
      do i = 1, m
        do j = i+1, m
          de = real(e(j) - e(i))
          de1 = real(h2(j,j) - h2(i,i))
          if (de < 10.0*k) then
            h2(i,j) = (0.00d0, 0.00d0)
            h2(j,i) = (0.00d0, 0.00d0)

            ps1(i,j) = ps(i,j)
            ps1(j,i) = ps(j,i)

            ps2(i,j) = ps(i,j)
            ps2(j,i) = ps(j,i)

            if (de /= de1) then
              if (de_array(j) > de_array(i)) then 
                h2(j,j) = h2(i,i) + de
                do z = 1, i-1
                  l1 = larray(z,i)
                  if (l1 >= lmin .and. l1 <= lmax) then
                  l1 = l1-lmin+1
                  phi1 = phis(l1)
                  theta1 = thetas(l1)
                  th1 = w1(l1)*((sx(z,j)*cos(phi1)+sy(z,j)*sin(phi1))*sin(theta1) &
                        + sz(z,j)*cos(theta1))
                  h2(z,j) = th1*exp(xj*phases(l1))
                  h2(j,z) = conjg(th1)*exp(-xj*phases(l1))
                  end if
                end do
                do z = j+1, m
                  l1 = larray(i,z)
                  if (l1 >= lmin .and. l1 <= lmax) then
                  l1 = l1-lmin+1
                  phi1 = phis(l1)
                  theta1 = thetas(l1)
                  th1 = w1(l1)*((sx(z,j)*cos(phi1)+sy(z,j)*sin(phi1))*sin(theta1) &
                        + sz(z,j)*cos(theta1))
                  h2(z,j) = th1*exp(xj*phases(l1))
                  h2(j,z) = conjg(th1)*exp(-xj*phases(l1))
                  end if
                end do
              else
                h2(i,i) = h2(j,j) - de
                do z = 1, i-1
                  l1 = larray(z,j)
                  if (l1 >= lmin .and. l1 <= lmax) then
                  l1 = l1-lmin+1
                  phi1 = phis(l1)
                  theta1 = thetas(l1)
                  th1 = w1(l1)*((sx(z,j)*cos(phi1)+sy(z,j)*sin(phi1))*sin(theta1) &
                        + sz(z,j)*cos(theta1))
                  h2(z,i) = th1*exp(xj*phases(l1))
                  h2(i,z) = conjg(th1)*exp(-xj*phases(l1))
                  end if 
                end do
                do z = j+1, m
                  l1 = larray(j,z)
                  if (l1 >= lmin .and. l1 <= lmax) then
                  l1 = l1-lmin+1
                  phi1 = phis(l1)
                  theta1 = thetas(l1)
                  th1 = w1(l1)*((sx(z,j)*cos(phi1)+sy(z,j)*sin(phi1))*sin(theta1) &
                        + sz(z,j)*cos(theta1))
                  h2(z,i) = th1*exp(xj*phases(l1))
                  h2(i,z) = conjg(th1)*exp(-xj*phases(l1))
                  end if
                end do
              end if
            end if
          end if
        end do
      end do

      end subroutine broadband_build_matrix_combined

!-----------------------------------------------------------------------------!
!-----------------------------------------------------------------------------!
