program main

      include 'omp_lib.h'

      integer i, n
      double precision x(1000), y(1000), s

      n = 1000
      s = 123.456

      do i = 1, n
        x(i) = rand ( )
        y(i) = rand ( )
      end do

c$omp parallel do
      do i = 1, n
        y(i) = y(i) + s * x(i)
      end do
c$omp end parallel do

      stop
      end
