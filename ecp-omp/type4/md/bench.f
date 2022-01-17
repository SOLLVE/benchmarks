


program main

!*****************************************************************************80
!
!! MAIN is the main program for MD.
!
!  Discussion:
!
!    MD implements a simple molecular dynamics simulation.
!
!    The velocity Verlet time integration scheme is used. 
!
!    The particles interact with a central pair potential.
!
!  Modified:
!
!    15 July 2008
!
!  Author:
!
!    FORTRAN90 original version by Bill Magro.
!    Modifications by John Burkardt
!    Modifications by Vivek Kale for OpenMP offload 
!
  use omp_lib

  integer, parameter :: nd = 3
  integer, parameter :: np = 500
  integer, parameter :: step_num = 100

  double precision acc(nd,np)
  double precision box(nd)
  double precision ctime
  double precision ctime1
  double precision ctime2
  double precision, parameter :: dt = 0.0001D+00
  double precision e0
  double precision ee(step_num)
  double precision force(nd,np)
  integer id
  double precision ke(step_num)
  double precision kinetic
  double precision, parameter :: mass = 1.0D+00
  double precision pe(step_num)
  double precision pos(nd,np)
  double precision potential
  integer seed
  integer step
  double precision vel(nd,np)
  double precision wtime

  call timestamp ( )

  write ( *, '(a)' ) ' '
  write ( *, '(a)' ) 'MD'
  write ( *, '(a)' ) '  FORTRAN90 version'
  write ( *, '(a)' ) ' '
  write ( *, '(a)' ) '  A molecular dynamics program.'
  write ( *, '(a)' ) ' '
  write ( *, '(a,i8)' ) '  NP, the number of particles in the simulation is ', np
  write ( *, '(a,i8)' ) '  STEP_NUM, the number of time steps, is ', step_num
  write ( *, '(a,g14.6)' ) '  DT, the size of each time step, is ', dt
!
!  Set the dimensions of the box.
!
  box(1:nd) = 10.0D+00
!
!  Set initial positions, velocities, and accelerations.
!
  write ( *, '(a)' ) ' '
  write ( *, '(a)' ) '  Initialize positions, velocities, and accelerations.'

  seed = 123456789
  call initialize ( np, nd, box, seed, pos, vel, acc )
!
!  Compute the forces and energies.
!
  write ( *, '(a)' ) ' '
  write ( *, '(a)' ) '  Compute initial forces and energies.'

  call compute ( np, nd, pos, vel, mass, force, potential, kinetic )
!
!  Save the initial total energy for use in the accuracy check.
!
  e0 = potential + kinetic

  write ( *, '(a)' ) ' '
  write ( *, '(a,g14.6)' ) '  Initial total energy E0 = ', e0
!
!  This is the main time stepping loop:
!    Compute forces and energies,
!    Update positions, velocities, accelerations.
!
  ctime1 = omp_get_wtime ( )

  do step = 1, step_num

    call compute ( np, nd, pos, vel, mass, force, potential, kinetic )

    pe(step) = potential 
    ke(step) = kinetic 
    ee(step) = ( potential + kinetic - e0 ) / e0

    call update ( np, nd, pos, vel, force, acc, mass, dt )

  end do
  ctime2 = omp_get_wtime ( )
!
!  Just for timing accuracy, we have moved the I/O out of the computational loop.
!
  write ( *, '(a)' ) ' '
  write ( *, '(a)' ) '  At each step, we report the potential and kinetic energies.'
  write ( *, '(a)' ) '  The sum of these energies should be a constant.'
  write ( *, '(a)' ) '  As an accuracy check, we also print the relative error'
  write ( *, '(a)' ) '  in the total energy.'
  write ( *, '(a)' ) ' '
  write ( *, '(a)' ) '      Step      Potential       Kinetic        (P+K-E0)/E0'
  write ( *, '(a)' ) '                Energy          Energy         Energy Error'
  write ( *, '(a)' ) ' '

  do step = 1, step_num

    write ( *, '(2x,i8,2x,g14.6,2x,g14.6,2x,g14.6)' ) &
      step, pe(step), ke(step), ee(step)

  end do

  ctime = ctime2 - ctime1
  write ( *, '(a)' ) ' '
  write ( *, '(a)' ) '  Elapsed cpu time for main computation:'
  write ( *, '(2x,g14.6,a)' ) ctime, ' seconds'

  write ( *, '(a)' ) ' '
  write ( *, '(a)' ) 'MD'
  write ( *, '(a)' ) '  Normal end of execution.'

  write ( *, '(a)' ) ' '
  call timestamp ( )

  stop
end
subroutine compute ( np, nd, pos, vel, mass, f, pot, kin )

!*****************************************************************************80
!
!! COMPUTE computes the forces and energies.
!
!  Discussion:
!
!    The computation of forces and energies is fully parallel.
!
!    The potential function V(X) is a harmonic well which smoothly
!    saturates to a maximum value at PI/2:
!
!      v(x) = ( sin ( min ( x, PI2 ) ) )**2
!
!    The derivative of the potential is:
!
!      dv(x) = 2.0D+00 * sin ( min ( x, PI2 ) ) * cos ( min ( x, PI2 ) )
!            = sin ( 2.0 * min ( x, PI2 ) )
!
!  Modified:
!
!    15 July 2008
!
!  Author:
!
!    FORTRAN90 original version by Bill Magro.
!    Modifications by John Burkardt
!
!  Parameters:
!
!    Input, integer NP, the number of particles.
!
!    Input, integer ND, the number of spatial dimensions.
!
!    Input, double precision POS(ND,NP), the position of each particle.
!
!    Input, double precision VEL(ND,NP), the velocity of each particle.
!
!    Input, double precision MASS, the mass of each particle.
!
!    Output, double precision F(ND,NP), the forces.
!
!    Output, double precision POT, the total potential energy.
!
!    Output, double precision KIN, the total kinetic energy.
!
  implicit none

  integer np
  integer nd

  double precision d
  double precision d2
  double precision f(nd,np)
  integer i
  integer j
  double precision kin
  double precision mass
  double precision, parameter :: PI2 = 3.141592653589793D+00 / 2.0D+00
  double precision pos(nd,np)
  double precision pot
  double precision rij(nd)
  double precision vel(nd,np)

  pot = 0.0D+00
  kin = 0.0D+00

!$omp target teams distribute parallel for simd num_teams(num_blocks) map(to: pos) map(tofrom:f, pe, ke, pot, kin) \ 
do private ( d, d2, rij, i, j ) shared ( f, nd, np, pos, vel ) &
!$omp reduction ( + : pot, kin )
  do i = 1, np
!
!  Compute the potential energy and forces.
!
    f(1:nd,i) = 0.0D+00

    do j = 1, np

      if ( i /= j ) then

        call dist ( nd, pos(1,i), pos(1,j), rij, d )
!
!  Attribute half of the potential energy to particle J.
!
        d2 = min ( d, PI2 )

        pot = pot + 0.5D+00 * sin ( d2 ) * sin ( d2 )

        f(1:nd,i) = f(1:nd,i) - rij(1:nd) * sin ( 2.0D+00 * d2 ) / d

      end if

    end do
!
!  Compute the kinetic energy.
!
    kin = kin + sum ( vel(1:nd,i)**2 )

  end do
!$omp end parallel do

  kin = kin * 0.5D+00 * mass
  
  return
end
subroutine dist ( nd, r1, r2, dr, d )

!*****************************************************************************80
!
!! DIST computes the displacement and distance between two particles.
!
!  Modified:
!
!    17 March 2002
!
!  Author:
!
!    FORTRAN90 original version by Bill Magro.
!    Modifications by John Burkardt
!
!  Parameters:
!
!    Input, integer ND, the number of spatial dimensions.
!
!    Input, double precision R1(ND), R2(ND), the positions of the particles.
!
!    Output, double precision DR(ND), the displacement vector.
!
!    Output, double precision D, the Euclidean norm of the displacement,
!    in other words, the distance between the two particles.
!
  implicit none

  integer nd

  double precision d
  double precision dr(nd)
  double precision r1(nd)
  double precision r2(nd)

  dr(1:nd) = r1(1:nd) - r2(1:nd)

  d = sqrt ( sum ( dr(1:nd)**2 ) )

  return
end
subroutine initialize ( np, nd, box, seed, pos, vel, acc )

!*****************************************************************************80
!
!! INITIALIZE initializes the positions, velocities, and accelerations.
!
!  Modified:
!
!    21 November 2007
!
!  Author:
!
!    FORTRAN90 original version by Bill Magro.
!    Modifications by John Burkardt
!
!  Parameters:
!
!    Input, integer NP, the number of particles.
!
!    Input, integer ND, the number of spatial dimensions.
!
!    Input, double precision BOX(ND), specifies the maximum position
!    of particles in each dimension.
!
!    Input/output, integer SEED, a seed for the random number generator.
!
!    Output, double precision POS(ND,NP), the position of each particle.
!
!    Output, double precision VEL(ND,NP), the velocity of each particle.
!
!    Output, double precision ACC(ND,NP), the acceleration of each particle.
!
  implicit none

  integer np
  integer nd

  double precision acc(nd,np)
  double precision box(nd)
  integer i
  integer j
  integer seed
  double precision pos(nd,np)
  double precision r8_uniform_01
  double precision vel(nd,np)
!
!  Start by setting the positions to random numbers between 0 and 1.
!
  call random_number ( harvest = pos(1:nd,1:np) )
!
!  Use these random values as scale factors to pick random locations
!  inside the box.
!
  do i = 1, nd
    pos(i,1:np) = box(i) * pos(i,1:np)
  end do
!
!  Velocities and accelerations begin at 0.
!
  vel(1:nd,1:np) = 0.0D+00
  acc(1:nd,1:np) = 0.0D+00

  return
end
subroutine timestamp ( )

!*****************************************************************************80
!
!! TIMESTAMP prints the current YMDHMS date as a time stamp.
!
!  Example:
!
!    May 31 2001   9:45:54.872 AM
!
!  Modified:
!
!    31 May 2001
!
!  Author:
!
!    John Burkardt
!
!  Parameters:
!
!    None
!
  implicit none

  character ( len = 8 ) ampm
  integer d
  character ( len = 8 ) date
  integer h
  integer m
  integer mm
  character ( len = 9 ), parameter, dimension(12) :: month = (/ &
    'January  ', 'February ', 'March    ', 'April    ', &
    'May      ', 'June     ', 'July     ', 'August   ', &
    'September', 'October  ', 'November ', 'December ' /)
  integer n
  integer s
  character ( len = 10 )  time
  integer values(8)
  integer y
  character ( len = 5 ) zone

  call date_and_time ( date, time, zone, values )

  y = values(1)
  m = values(2)
  d = values(3)
  h = values(5)
  n = values(6)
  s = values(7)
  mm = values(8)

  if ( h < 12 ) then
    ampm = 'AM'
  else if ( h == 12 ) then
    if ( n == 0 .and. s == 0 ) then
      ampm = 'Noon'
    else
      ampm = 'PM'
    end if
  else
    h = h - 12
    if ( h < 12 ) then
      ampm = 'PM'
    else if ( h == 12 ) then
      if ( n == 0 .and. s == 0 ) then
        ampm = 'Midnight'
      else
        ampm = 'AM'
      end if
    end if
  end if

  write ( *, '(a,1x,i2,1x,i4,2x,i2,a1,i2.2,a1,i2.2,a1,i3.3,1x,a)' ) &
    trim ( month(m) ), d, y, h, ':', n, ':', s, '.', mm, trim ( ampm )

  return
end
subroutine update ( np, nd, pos, vel, f, acc, mass, dt )

!*****************************************************************************80
!
!! UPDATE updates positions, velocities and accelerations.
!
!  Discussion:
!
!    The time integration is fully parallel.
!
!    A velocity Verlet algorithm is used for the updating.
!
!    x(t+dt) = x(t) + v(t) * dt + 0.5 * a(t) * dt * dt
!    v(t+dt) = v(t) + 0.5 * ( a(t) + a(t+dt) ) * dt
!    a(t+dt) = f(t) / m
!
!  Modified:
!
!    21 November 2007
!
!  Author:
!
!    FORTRAN90 original version by Bill Magro.
!    Modifications by John Burkardt
!
!  Parameters:
!
!    Input, integer NP, the number of particles.
!
!    Input, integer ND, the number of spatial dimensions.
!
!    Input/output, double precision POS(ND,NP), the position of each particle.
!
!    Input/output, double precision VEL(ND,NP), the velocity of each particle.
!
!    Input, double precision F(ND,NP), the force on each particle.
!
!    Input/output, double precision ACC(ND,NP), the acceleration of each
!    particle.
!
!    Input, double precision MASS, the mass of each particle.
!
!    Input, double precision DT, the time step.
!
  integer np
  integer nd

  double precision acc(nd,np)
  double precision dt
  double precision f(nd,np)
  integer i
  integer j
  double precision mass
  double precision pos(nd,np)
  double precision rmass
  double precision vel(nd,np)
  integer num_blocks 

  rmass = 1.0D+00 / mass
  num_blocks = 8

!$omp target teams distribute parallel do simd map(tofrom: pos, vel, acc, f, pe, ke) num_teams(num_blocks) private (i, j) shared (acc, dt, f, nd, np, pos, rmass, vel) 
  do j = 1, np
    do i = 1, nd
      pos(i,j) = pos(i,j) + vel(i,j) * dt + 0.5D+00 * acc(i,j) * dt * dt
      vel(i,j) = vel(i,j) + 0.5D+00 * dt * ( f(i,j) * rmass + acc(i,j) )
      acc(i,j) = f(i,j) * rmass
    end do
  end do
!$omp end parallel do

  return
end
