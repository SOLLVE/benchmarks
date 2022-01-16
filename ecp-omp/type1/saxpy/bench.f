program main
      include 'omp_lib.h'
     
      use iso_fortran_env 
      integer i, n
      double precision x(1000), y(1000), s
c     real(kind=real32), dimension(n) :: x
c     real(kind=real32), dimension(n) :: y
      
      integer :: n, i, num_blocks
      real(kind=real32) :: a 
     
      n = 1000
      s = 123.456
      
      do i = 1, n
        x(i) = rand ( )
        y(i) = rand ( )
      end do
      
      
c$omp target teams distribute parallel simd map(to: x, a) map(tofrom: y) num_teams(num_blocks) thread_limit(112) do
      do i = 1, n
        y(i) = y(i) + s * x(i)
      end do
c$omp end teams distribute parallel

      stop
      end
