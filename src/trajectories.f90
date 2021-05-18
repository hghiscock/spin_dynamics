      module trajectories
      implicit none
      contains
       
!-----------------------------------------------------------------------------!

      subroutine sy_kmc(m, ntraj, e, evi, ps, pt, ks, kt, svecs, ncores, &
                        c0)

      !Compute the average of ntraj trajectories in a kinetic Monte
      !Carlo algorithm

      integer, intent(in) :: m, ntraj, ncores
      complex(8), intent(in) :: e(:), evi(:,:)
      complex(8), intent(in) :: ps(:,:), pt(:,:)
      complex(8), intent(in) :: svecs(:,:)
      
      real(8), intent(in) :: ks, kt

      real(8), intent(out) :: c0

      complex(8) :: psi(m)
      integer :: i
      real(8) :: c0tmp, ttmp, np
        
      call omp_set_num_threads(ncores)
      c0 = 0.00d0
!$omp parallel do default(shared) private(i,psi,c0tmp,ttmp) reduction(+:c0)
      do i = 1, ntraj
        call random_number(np)
        psi = svecs(:,int(np*m/4)+1)
        psi = matmul(evi, psi)
        call kmc(psi, ps, pt, ks, kt, e, m, c0tmp, ttmp)
        c0 = c0 + c0tmp
      end do
!$omp end parallel do

      end subroutine sy_kmc

!-----------------------------------------------------------------------------!

      subroutine kmc(psi, ps, pt, ks, kt, e, m, c0, time)

      !Does the kinetic Monte Carlo steps for a single trajectory

      complex(8), intent(in) :: psi(:), ps(:,:), pt(:,:), e(:)
      real(8), intent(in) :: ks, kt
      integer, intent(in) :: m

      real(8), intent(out) :: c0, time

      complex(8), parameter :: xj = (0.00d0, 1.00d0)
      complex(8) :: psi_tmp(m), prop(m)
      real(8) :: trand, dt, r_react, prob_r, r_success, qinv, s_react, modu, r_rf
      logical :: reacted

      reacted = .false.
      qinv = 1.00d0/(ks+kt)
      psi_tmp = psi
      s_react = ks/(ks+kt)
      time = 0.0d0

      do while (.not. reacted)
        call random_number(trand)
        dt = qinv*log(1.00d0/trand)
        time = time + dt
        prop = exp(-xj*dt*e)
        psi_tmp = prop*psi_tmp
        call random_number(r_react)
        if (r_react < s_react) then
          !singlet reaction
          prob_r = dot_product(psi_tmp,matmul(ps,psi_tmp))
          call random_number(r_success)
          if (r_success < prob_r) then
            call random_number(r_rf)
            c0 = 1
            reacted = .true.
          end if
        else
          !triplet reaction          
          prob_r = dot_product(psi_tmp,matmul(pt,psi_tmp))
          call random_number(r_success)
          if (r_success < prob_r) then
            c0 = 0.0d0
            reacted = .true.
          end if
        end if

      end do
      
      end subroutine kmc

!-----------------------------------------------------------------------------!

      end module trajectories
