!
! Description: 2D Jacobi relaxation example program used for experimenting with over-decomposition strategies along with GPU-ization.
! This program uses subroutines for different parts of the Jacobi relaxation computation.
!
! Author: Vivek Kale
!
! Last Edited: January 5th, 2022

      MODULE util
      contains

      subroutine printSample(arr, dimRow, dimCol, sampleSizeRow, &
      sampleSizeCol)

      real, pointer :: arr(:,:) ! assigning to m, m
      integer :: dimRow
      integer :: dimCol
      integer :: sampleSizeRow
      integer :: sampleSizeCol
      integer :: krow
      integer :: kcol

      integer i
      integer j

      krow = dimRow/sampleSizeRow
      kcol = dimCol/sampleSizeCol

      ! write(*,*), 'rank', rank, krow , dimRow
      do i = 1, dimRow, krow
         write(*,*) ''
         do j = 1, dimCol, kcol
            write(*,"(f8.3)", advance="no") arr(i,j)
         end do
      end do

      write(*,*) ''
      end subroutine printSample
      end MODULE util

      MODULE meshcomp
      contains

 subroutine sweep(a, b, N, M, FLOP, rank, p)

        #ifdef HAVE_OPENACC
        use openacc
        #endif

#ifdef USE_ALLOCATABLE
real, allocatable :: a(:,:)
#else
      real, pointer, intent(inout), contiguous :: a(:,:)
 !     real, pointer :: pa(:,:)
#endif

#ifdef USE_ALLOCATABLE
      real, allocatable :: b(:,:)
#else
      real, pointer, intent(inout), contiguous :: b(:,:)
  !    real, pointer :: pb(:,:)
#endif
      integer:: N
      integer:: M
      integer :: rank
      integer :: p
      integer :: i
      integer :: j

      integer :: ht
      integer :: FLOP

     ht =  2 + N/p

     ! FLOP = 4

     #ifdef USE_ALLOCATABLE
     if(.not.allocated(a) .eqv. .TRUE.) then
     allocate(a(ht, N+2))
     print *, 'Sweep(): rank ' , rank, ' allocated array a.'
     call flush(6)
     call flush(0)
     end if
     if(.not.allocated(b) .eqv. .TRUE.) allocate(b(ht, N+2))
     #else

     #endif


!Note that we don't do loop interchange for stride 1 access . TODO: change to stride-1 access .


#ifdef HAVE_OPENACC
!$ACC loop seq
#endif
     do k = 1, FLOP

#ifdef HAVE_OPENACC
!$ACC data pcopyin(a(1:ht, 1:N+2)) pcreate(b(2:ht-1, 2:N+1))
!$ACC kernels
#endif

!$ACC loop independent collapse(2)

        do i = 2, ht-1
           do j = 2, N+1
                 b(i,j) = 0.2*(a(i,j) + a(i-1,j) + &
                      a(i+1,j) + a(i,j-1) + a(i,j+1))
              end do
           end do
#ifdef HAVE_OPENACC
!$ACC loop independent collapse(2)
#endif

           do i = 2, ht-1
              do j = 2, N+1
                 a(i,j) = b(i,j)
              end do
           end do

#ifdef HAVE_OPENACC
!$ACC end kernels
!$ACC end data
#endif

end do

      end subroutine sweep

      subroutine boundarycopy(topBoundary, bottomBoundary, a, N, M, p)

#ifdef USE_ALLOCATABLE
      real, allocatable :: topBoundary(:)
      real, allocatable :: bottomBoundary(:)
      real, allocatable :: a(:,:)
#else
       real, pointer, intent(inout), contiguous :: topBoundary(:)
       real, pointer, intent(inout), contiguous :: bottomBoundary(:)
       real, pointer, intent(inout), contiguous :: a(:,:)
#endif

      integer:: N
      integer :: M

      integer  :: p
      integer :: ht
      integer :: ii
      ht = 2 + N/p

      #ifdef USE_ALLOCATABLE
      if(.not.allocated(topBoundary) .eqv. .TRUE.)  allocate(topBoundary(N+2))
      if(.not.allocated(bottomBoundary) .eqv. .TRUE. ) allocate(bottomBoundary(N+2))
      if(.not.allocated(a) .eqv. .TRUE.) allocate(a(ht, N+2))
      #else
      #endif

#ifdef HAVE_OPENACC
      !$ACC kernels loop independent pcopy(a(2,1:N+2)) copyout(topBoundary(1:N+2))
#endif
       do ii = 1, N+2
          topBoundary(ii) = a(2, ii)
       end do ! ii
#ifdef HAVE_OPENACC
!$ACC end kernels
#endif

#ifdef HAVE_OPENACC
     !$ACC kernels loop independent pcopy(a(ht-1,1:N+2)) copyout(bottomBoundary(1:N+2))
#endif
       do ii = 1, N+2
          bottomBoundary(ii) = a(ht-1, ii)
       end do ! ii
#ifdef HAVE_OPENACC
!$ACC end kernels
#endif

      end subroutine boundarycopy

      subroutine resetBoundaries(a, N, M, rank, p)

#ifdef HAVE_OPENACC
        use openacc
#endif

#ifdef USE_ALLOCATABLE
       real, allocatable :: a(:,:)
#else
       real, pointer, intent(inout), contiguous :: a(:,:)
#endif

       integer:: N
       integer :: M
       integer :: rank
       integer :: p

       integer :: ht
       integer :: ii
       ht = 2 + M/p

       #ifdef USE_ALLOCATABLE
       if(.not.allocated(a) .eqv. .TRUE. ) allocate(a(ht,N+2))
      #endif

#ifdef HAVE_OPENACC
!$ACC kernels present(a(1:ht, 1:N+2)) pcopyin(rank, p)
#endif

    ! Set values at boundary cells

    ! Set values at boundary cells for rank 0 and p-1
       if(rank == 0) then
          a(1, 2) = 0.0
          a(2, 1) = 0.0
       end if

       if (rank == (p - 1)) then
          a(ht, n+1) = 100.0
          a(ht-1, n+2) = 100.0
       end if


!$ACC loop
       do ii = 2, ht-1
          a(ii, 1) = a(ii, 2)
          a(ii, N+2) = a(ii, N+1)
       end do ! ii

#ifdef HAVE_OPENACC
      !$ACC end kernels
#endif
         end subroutine resetBoundaries

      subroutine haloWrite(topHalo, bottomHalo, a, N, M, rank, p)

#ifdef USE_ALLOCATABLE
      real, allocatable :: topHalo(:)
      real, allocatable :: bottomHalo(:)
      real, allocatable :: a(:,:)
#else
      real, pointer, intent(inout), contiguous :: topHalo(:)
      real, pointer, intent(inout), contiguous :: bottomHalo(:)
      real, pointer, intent(inout), contiguous :: a(:,:)
#endif

      integer :: N
      integer :: M
      integer :: rank
      integer  :: p

      integer :: ht
      integer :: ii

      ht = 2 + N/p
      #ifdef USE_ALLOCATABLE
      if(.not.allocated(a) .eqv. .TRUE.) allocate(a(ht, N+2))
      if(.not.allocated(topHalo) .eqv. .TRUE.) allocate(topHalo(N+2))
      if(.not.allocated(bottomHalo) .eqv. .TRUE.) allocate(bottomHalo(N+2))
      #endif

! TODO: ensure correctness of 1 to N+2

#ifdef HAVE_OPENACC
     if (rank == 0) then
        !$ACC kernels loop independent copy(topHalo(1:N+2)) present(a(2, 1:N+2))
        do ii = 1, N+2
              topHalo(ii) = a(2, ii)
          end do
          !$ACC end kernels
       end if

           if (rank == (p-1)) then
              !$ACC kernels loop independent copy(bottomHalo(1:N+2)) present(a(ht-1, 1:N+2))
              do ii = 1, N+2
                bottomHalo(ii) = a(ht-1, ii)
             end do
             !$ACC end kernels
   end if
#endif
      end subroutine haloWrite

      subroutine haloRead(topHalo, bottomHalo, a, N, M, rank, p)
#ifdef USE_ALLOCATABLE
        real, allocatable :: topHalo(:)
        real, allocatable :: bottomHalo(:)
        real, allocatable :: a(:,:)
#else
        real, pointer, intent(inout), contiguous :: topHalo(:)
        real, pointer, intent(inout), contiguous :: bottomHalo(:)
        real, pointer, intent(inout), contiguous :: a(:,:)
#endif
        integer :: N, M
        integer  :: rank
        integer  :: p

        integer :: ii
        integer :: ht
        ht = 2 + N/p

        #ifdef USE_ALLOCATABLE
        if(.not.allocated(topHalo) .eqv. .TRUE.) allocate(topHalo(N+2))
        if(.not.allocated(bottomHalo) .eqv. .TRUE.) allocate(bottomHalo(N+2))
        if(.not.allocated(a) .eqv. .TRUE.) allocate(a(ht, N+2))
        #endif

! TODO: check why we need independent to force parallelization, even though we have contiguous.

         ! if (rank .gt. 0) then
            #ifdef HAVE_OPENACC
            #ifdef USE_ALLOCATABLE
            !$ACC kernels loop independent pcopy(a(1,1:N+2)) copyin(topHalo(1:N+2))
            #else
            !$ACC kernels loop independent pcopy(a(1,1:N+2)) copyin(topHalo(1:N+2))
        ! !$ACC kernels loop independent pcopy(a) copyin(topHalo)
            #endif
            #endif
            do ii = 1, N+2
               a(1, ii)  = topHalo(ii)
            end do
            #ifdef HAVE_OPENACC
            !$ACC end kernels loop
            #endif
         ! end if

    !     if (rank .lt. p - 1) then
            #ifdef HAVE_OPENACC
            #ifdef USE_ALLOCATABLE
            !$ACC kernels loop independent present(a(ht,1:N+2)) copyin(bottomHalo(1:N+2))
            #else
            !$ACC kernels loop independent present(a(ht,1:N+2)) copyin(bottomHalo(1:N+2))
            #endif
            #endif
            do ii = 1, N+2
               a(ht,ii) = bottomHalo(ii)
            end do
            #ifdef HAVE_OPENACC
            !$ACC end kernels loop
            #endif
     !    end if
       end subroutine haloRead

      end MODULE meshcomp

      program stencil
      !use util
      use meshcomp
      use iso_c_binding, only: c_int, c_double, c_ptr, c_null_ptr

!!DEC$ IF DEFINED(HAVE_OPENACC)
#ifdef HAVE_OPENACC
      use openacc
#elif HAVE_OPENMP
      use omp_lib
#endif
!!DEC$ ENDIF

      implicit none

      ! Include the mpif.h file corresponding to the compiler that compiles this code.
      include 'mpif.h'

      ! Variables for MPI
      integer :: rank
      integer  :: size
      integer :: ierror
      integer :: numRequests
      INTEGER, allocatable :: ARRAY_OF_REQUESTS(:)
      integer :: requestCounter = 1

      ! Variables for OpenMP
      INTEGER NTHDS
      INTEGER MAX_THREADS
      INTEGER TID
      PARAMETER(MAX_THREADS=16)

      ! Variables for application code

      !     variables specifying parameters of application
      INTEGER TIMESTEP
      integer nSteps
      integer, PARAMETER :: probSize = 16
      integer, PARAMETER :: NUM_FLOP = 1
      integer :: N ! row dimension  of mesh  -TODO: change variable name to Nx
      integer :: Ny ! col dimension of mesh - TODO (priority): don't force Ny to be same as N
      integer :: FLOP ! number of repetitions to do for inner loop, to increase FLOP count
      integer, PARAMETER :: NUM_TIMESTEPS = 10  ! number of application timesteps

      !     data arrays for application code

      !TODO : check if this is correct
      #ifdef USE_ALLOCATABLE
      real, allocatable :: data_a(:,:)
      real, allocatable :: data_b(:,:)
      real, allocatable :: data_temp(:,:) ! can be used for pointer swapping
      #else
      real, pointer, contiguous :: data_a(:,:)
      real, pointer, contiguous :: data_b(:,:)
      real, pointer, contiguous :: data_temp(:,:) ! can be used for pointer swapping
      #endif

      !     used to check correctness
      integer :: xsum
      double precision checkSum
      integer r

      !  buffers and variables for border exchange
      integer ht ! height for the halo

      #ifdef USE_ALLOCATABLE
      real, allocatable :: data_topBoundary(:)
      real, allocatable :: data_bottomBoundary(:)
      real, allocatable :: data_topHalo(:)
      real, allocatable :: data_bottomHalo(:)
      #else
      real, pointer, contiguous :: data_topBoundary(:)
      real, pointer, contiguous :: data_bottomBoundary(:)
      real, pointer, contiguous :: data_topHalo(:)
      real, pointer, contiguous :: data_bottomHalo(:)
      #endif
      integer :: msgSize

      ! loop iteration variables
      INTEGER :: i, j

      integer :: k  ! used currently only for FLOP count

      !     performance timing variables
      double precision :: startTime, endTime, totalTime

      integer num_dev
      integer dev_id
      integer :: numRanksPerNode = 2

      ! used for input to program
      character*80 arg
      integer (kind = 4) num_args

      num_args = command_argument_count() ! TODO: fix this.
      num_args = 5
      if (num_args .gt. 3) then
          call getarg(1, arg)
          read(arg,*) N
          call getarg(2, arg)
          read(arg,*) Ny
          call getarg(3, arg)
          read(arg,*) FLOP
          call getarg(4, arg)
          read(arg,*) nSteps
       else
          N = probSize
          Ny = probSize
          FLOP = NUM_FLOP
          nSteps = NUM_TIMESTEPS
      end if

      call MPI_Init(ierror)
      call MPI_Comm_Size(MPI_COMM_WORLD, size, ierror)
      call MPI_Comm_Rank(MPI_COMM_WORLD, rank, ierror)

      allocate(ARRAY_OF_REQUESTS(size))
      do i = 1, size
         ARRAY_OF_REQUESTS(i) = MPI_REQUEST_NULL
      end do

      ht = 2 + N/size

      print *, 'jacobi.f: Allocating arrays.'
      call flush(6)
      call flush(0)
      ! TODO: change the below to Ny  (focused on square matrix now).
      allocate(data_a(ht, N+2))
      allocate(data_b(ht, N+2))
      allocate(data_topBoundary(N+2))
      allocate(data_bottomBoundary(N+2))
      allocate(data_topHalo(N+2))  !  need this for GPU to copy out
      allocate(data_bottomHalo(N+2))  ! need this for GPU to copy out

#ifdef HAVE_OPENACC

      call acc_init(acc_get_device_type())
      ! The below allows using multiple GPUs
      num_dev = acc_get_num_devices(acc_get_device_type())
      dev_id = mod(rank, num_dev)
      !  assign GPU to one MPI process
      call acc_set_device_num(dev_id, acc_get_device_type())

      print *, 'MPI rank ', rank, ' assigned to GPU ' , dev_id
      call flush(6)
      call flush(0)
      nthds = 2880 ! TODO: see if you can use OpenACC library to query number of cores on node

      if (rank == 0) then
         print *, 'jacobi.f: Initialized OpenACC.'
         call flush(6)
         call flush(0)
      end if

#elif HAVE_OPENMP
!$OMP PARALLEL
      nthds = omp_get_num_threads()
      tid = omp_get_thread_num()
      !      print *, 'Thread ' , tid + 1 , ' of ', nthds , ' active.'
       print *, 'Thread ' , omp_get_thread_num() + 1 , ' of ', nthds , ' active.'
       call flush(6)
       call flush(0)
       !$OMP END PARALLEL
#endif
       ! Set the message size for MPI buffers for isend/irecv/waitall
       msgSize = N+2
       !     Initial values assigned to matrix.
       !   All values of the matrix cells are 50.0.
       data_a = 50.0

! TODO: check whether we need to set  initial values, and if so make code comments explaning why.
       data_topBoundary = 50.0
       data_bottomBoundary = 50.0
       data_topHalo = 50.0
       data_bottomHalo = 50.0

       call MPI_BARRIER(MPI_COMM_WORLD, ierror)
       startTime = MPI_Wtime()

! TODO: check if we need to copy pointers to arrays instead of actual arrays
#ifdef HAVE_OPENACC
!$ACC data copy(data_a(1:ht, 1:N+2)) create(data_b(2:ht-1, 2:N+1))
#endif
      do timestep = 1, nSteps
         requestCounter = 1
         if (rank .ne. 0) then
            call MPI_Irecv(data_topHalo, msgSize, MPI_REAL, rank-1, 0, &
            MPI_COMM_WORLD, array_of_requests(requestCounter), ierror)
            requestCounter = requestCounter + 1
            call MPI_Isend(data_topBoundary, msgSize, MPI_REAL, rank-1, 0, &
            MPI_COMM_WORLD, array_of_requests(requestCounter), ierror)
            requestCounter = requestCounter + 1
         end if

         if (rank .ne. (size-1)) then
            call MPI_Irecv(data_bottomHalo, msgSize, MPI_REAL, rank + 1, 0, &
            MPI_COMM_WORLD, array_of_requests(requestCounter), ierror)
            requestCounter = requestCounter + 1
            call MPI_Isend(data_bottomBoundary, msgSize, MPI_REAL, rank + 1, 0, MPI_COMM_WORLD, &
            array_of_requests(requestCounter), ierror)
            requestCounter = requestCounter + 1
         end if

         call MPI_Waitall(requestCounter - 1, array_of_requests, &
         MPI_STATUSES_IGNORE, ierror)

! Note that when using MPI, each MPI process (within or across nodes) invokes the below.
#ifdef HAVE_OPENACC

           call haloRead(data_topHalo, data_bottomHalo, data_a, N, N, rank, size)
           ! write(*,*), "finished with haloRead"

           call haloWrite(data_topHalo, data_bottomHalo, data_a, N, N, rank, size)
            !write(*,*), "finished with haloWrite
           ! reset boundaries
           ! TODO:  may need to have one GPU thread do this (check).
           call resetBoundaries(data_a, N, N, rank, size)
            !write(*,*), "finished with resetBoundaries"

           call sweep(data_a, data_b, N, N, FLOP, rank, size)

            !write(*,*), "finished with sweep"
           call boundarycopy(data_topBoundary, data_bottomBoundary, data_a, N, N, size)

#elif HAVE_OPENMP
           ! read from Halo

!$OMP PARALLEL
!$OMP DO SCHEDULE(STATIC,4)
           do i = 1, N+2
              data_a(1, i) = data_topHalo(i)
           end do
!$OMP END DO
!$OMP END PARALLEL

!$OMP PARALLEL
!$OMP DO SCHEDULE(STATIC,4)
           do i = 1, N+2
              data_a(ht, i) = data_bottomHalo(i)
           end do
!$OMP END DO
!$OMP END PARALLEL

!$OMP PARALLEL
!$OMP DO SCHEDULE(STATIC, 1) ! do chunk size 1 to reduce capacity cache misses
           do i = 2, ht-1
              data_a(i, 1) = data_a(i, 2)
              data_a(i, N+2) = data_a(i, N+1)
           end do
!$OMP END DO
!$OMP END PARALLEL

           ! reset boundaries
           if(rank == 0) then
              data_a(1, 2) = 0.0
              data_a(2, 1) = 0.0
           end if

           if (rank == (size - 1)) then
              data_a(ht, n+1) = 100.0
              data_a(ht -1, n+2) = 100.0
           end if

 ! do the sweep
! TODO: find out whether loop interchange is needed.
do k = 1, FLOP
!$OMP PARALLEL
!$OMP DO SCHEDULE(STATIC, 4) collapse(2)
   do i = 2, ht-1
      do j = 2, N+1
                 data_b(i,j) = 0.2*(data_a(i,j) + data_a(i-1,j) + &
                 data_a(i+1,j) + data_a(i,j-1) + data_a(i,j+1))
              end do
           end do
!$OMP END DO

! TODO: find out whether loop interchange is needed.
!$OMP DO SCHEDULE(STATIC, 4) collapse(2)
           do i = 2, ht-1
              do j = 2, N+1
                 data_a(i,j) = data_b(i,j)
              end do
           end do
!$OMP END DO
!$OMP END PARALLEL

end do

!$OMP PARALLEL
!$OMP DO SCHEDULE(STATIC, 4)
           do i = 1, N+2
              data_topBoundary(i) = data_a(2, i)
           end do
!$OMP END DO

!$OMP DO SCHEDULE(STATIC, 4)
           do i = 1, N+2
              data_bottomBoundary(i) = data_a(ht-1, i)
           end do
!$OMP END DO
!$OMP END PARALLEL

#endif
        call MPI_BARRIER (MPI_COMM_WORLD, ierror)
      end do ! end timestep
#ifdef HAVE_OPENACC
!$ACC end data
!!$ACC exit data
#endif

      endTime = MPI_Wtime()


! print sample values of resulting matrix
      if (rank == (N/3)/(ht-2)) then
         r = mod((N/3), ht-2 )
         write(*, *) N/3,':', data_a(r, N/3), data_a(r,2*N/3)
      end if
      if (rank == (2*N/3)/(ht-2)) then
         r = mod((2*N/3), ht-2)
         write(*, *) 2*N/3,':', data_a(r, N/3), data_a(r,2*N/3)
      end if

! print timing data to output file
      if (rank == 0) then
         print *, 'That took ', endTime - startTime , ' seconds.'
         call flush(6)
         call flush(0)
         Open(unit=10, file="outfile-meshcomp-surface.dat", access="sequential", form="formatted", &
         status="unknown", position="append")
         ! TODO: may need to find a unsigned long for problem size

! TODO: change N*N to N*M

#ifdef HAVE_OPENACC
         #ifdef USE_ALLOCATABLE
         write(10, '(A, I7, I7, I7, A, A, I7, I7, A, f8.3, f8.3, f8.3)'), '\t jac', N*N, FLOP, nSteps, '\t ifp', '\t oac', size, nthds, '\t alc', endTime - startTime, 0.0, 0.0
         #else
         write(10, '(A, I7, I7, I7, A, A, I7, I7, A, f8.3, f8.3, f8.3)'), '\t jac', N*N, FLOP, nSteps, '\t ifp', '\t oac', size, nthds, '\t ptr', endTime - startTime, 0.0, 0.0
         #endif
#elif HAVE_OPENMP

         #ifdef USE_ALLOCATABLE
         write(10, '(A, I7, I7, I7, A, A, I7, I7, A, f8.3, f8.3, f8.3)'), '\t jac', N*N, FLOP, nSteps, '\t ifp', '\t omp', size, nthds, '\t alc', endTime - startTime, 0.0, 0.0
         #else
         write(10, '(A, I7, I7, I7, A, A, I7, I7, A, f8.3, f8.3, f8.3)'), '\t jac', N*N, FLOP, nSteps,'\t ifp', '\t omp', size, nthds, '\t ptr', endTime - startTime, 0.0, 0.0
         #endif
#endif
         close(10)

      end if ! rank == 0

      call MPI_Finalize(ierror)
      stop

      end program stencil
