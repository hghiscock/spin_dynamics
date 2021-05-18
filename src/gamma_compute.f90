!-----------------------------------------------------------------------------!

      subroutine sy_gamma_compute(num_cores, m, nt, jj, omega_rs, omega_rf, k, &
                       phi_s)
      implicit none

      !Calculates the singlet yield as final step in gamma-compute
      !algorithm

      real(8), intent(in) :: jj(:,:,:), omega_rs(:,:)
      real(8), intent(in) :: omega_rf, k
      integer, intent(in) :: m, num_cores, nt

      real(8), intent(out) :: phi_s(nt,m,m)
      real(8) :: ftrp

      integer :: l, i, ii

      call omp_set_num_threads(num_cores)

!$omp parallel do default(shared) private(ii, i, l, ftrp)
      do ii = 1, nt/2
        do l = 1,m
          do i = 1,m
            ftrp = (omega_rs(l,i) + dble(ii - nt/2 - 1)*omega_rf)**2.00d0/(k*k)
            phi_s(ii + nt/2,l,i) = jj(ii + nt/2,l,i)/(1.00d0+ftrp)
          end do
        end do

        do l = 1,m
          do i = 1,m
            ftrp = (omega_rs(l,i) + dble(ii - 1)*omega_rf)**2.00d0/(k*k)
            phi_s(ii,l,i) = jj(ii,l,i)/(1.00d0+ftrp)
          end do
        end do
      end do
!$omp end parallel do

      end subroutine sy_gamma_compute

!-----------------------------------------------------------------------------!

      subroutine complexgramschmidt(num_cores, m, x, x1)

      !Carries out gram schmidt orthogonalisation of eigenvectors
      !of non-Hermitian matrix
 
      integer, intent(in) :: m, num_cores
      complex(8), intent(in) :: x(:,:)

      complex(8), intent(out) :: x1(m,m)

      complex(8) :: ctmp
      integer :: i, j, k

      call omp_set_num_threads(num_cores)

      x1 = (0.00d0, 0.00d0)

      x1(:,1) = x(:,1)

      do i = 2,m
        do j = 1,i-1
          ctmp = (0.00d0, 0.00d0)
!$omp parallel do default(shared) private(k) reduction(+:ctmp)
          do k = 1, m
            ctmp = ctmp + conjg(x(k,j)) * x(k,i)
          end do
!$omp end parallel do
          x1(:,i) = x(:,i) - &
                ctmp*x(:,j)
          ctmp = (0.00d0, 0.00d0)
!$omp parallel do default(shared) private(k) reduction(+:ctmp)
          do k = 1, m
            ctmp = ctmp + conjg(x1(k,i)) * x1(k,i)
          end do
!$omp end parallel do
          x1(:,i) = x1(:,i) / &
                sqrt(ctmp)
        end do
      end do

      end subroutine complexgramschmidt

!-----------------------------------------------------------------------------!

      subroutine get_omega_rs(num_cores, m, wrf, lambda, wrs)

      !Finds ratio of pairs of eigenvalues

      complex(8), intent(in) :: lambda(:)
      real(8), intent(in) :: wrf
      integer, intent(in) :: m, num_cores

      complex(8), intent(out) :: wrs(m,m)

      integer :: i, j

      call omp_set_num_threads(num_cores)

!$omp parallel do default(shared) private(i, j)
      do i = 1, m
        do j = 1, m
          wrs(i,j) = -1.0*wrf*aimag(log(lambda(i)/lambda(j)))
        end do
      end do
!$omp end parallel do

      end subroutine get_omega_rs

!-----------------------------------------------------------------------------!
!-----------------------------------------------------------------------------!
