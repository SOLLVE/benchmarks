# include <stdlib.h>
# include <stdio.h>
# include <omp.h>

int main ( int argc, char *argv[] )
{
  int i, n = 1000;
  double x[1000], y[1000], s;

  s = 123.456;

  for ( i = 0; i < n; i++ )
  {
    x[i] = ( double ) rand ( ) / ( double ) RAND_MAX;
    y[i] = ( double ) rand ( ) / ( double ) RAND_MAX;
  }

//  void saxpy ( int n, float a, float* restrict x, float* restrict* y)
 #ifdef DEFAULT   
 #pragma omp teams distribute parallel for map(to: x[0:n], a) 
  
#elif TEAMS 
  #pragma omp target teams distribute parallel for map(to: x[0:n], a) map(tofrom: y[0:n]) map(tofrom: y[0:n]) 
 
#elif  THREADS
  #pragma omp target teams distribute parallel for map(to: x[0:n], a) map(tofrom: y[0:n]) map(tofrom: y[0:n]) thread_limit(112) 
  
#elif TEAMS_THREADS 
  #pragma omp target teams distribute parallel for map(to: x[0:n], a) map(tofrom: y[0:n]) map(tofrom: y[0:n]) num_teams(4) thread_limit(112) 
  
  #else   
  #pragma omp parallel for  
  #endif 
   
  for ( i = 0; i < n; i++ )
  {
    y[i] = y[i] + s * x[i];
  } 
  
  return 0;
}
