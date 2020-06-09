#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

#define N 100
#define MAXWORK 1000
#define GSZ 1

/*
 * work out which GPU to use
 */

inline unsigned
gpu_scheduler(const unsigned *occupancies, int ngpus)
{
  short looking = 1;
  unsigned chosen;

  //  return 0;
  //  return rand()%ngpus;
  //   return (i % ngpus);

  while (looking) {
    for (unsigned i = 0; i < ngpus; i++) {
      if (occupancies[i] == 0) {
	chosen = i;
	looking = 0;
	break;
      }
    }
  }
  return chosen;

}

#pragma omp declare target
int
work_generator(int work)
{
  const int NN = work;
  int a[MAXWORK][MAXWORK], b[MAXWORK][MAXWORK], c[MAXWORK][MAXWORK];
  int i, j;

  for (i = 0; i < NN; i++)
    {
      for (j = 0; j < NN; j++)
	{
	  c[i][j] = 0;
	  for (int k = 0; k < NN; k++)
	    {
	      c[i][j]+=a[i][k]*b[k][j];
	    }
	}
    }

  return c[i - 1][j - 1];     /* c[i][j] is bounds error */
}
#pragma omp end declare target

int
main()
{
  unsigned success[N];
  const int ndevs = omp_get_num_devices();
  unsigned *task_counts = NULL;
  unsigned *occupancies = NULL;
  double start_iterations, end_iterations;

  /* make sure we have some GPUs */
  assert(ndevs > 0);

  printf("There are %d GPUs\n", ndevs);

  task_counts = (unsigned *) calloc(ndevs, sizeof(*task_counts));
  assert(task_counts != NULL);
  occupancies = (unsigned *) calloc(ndevs, sizeof(*occupancies));
  assert(occupancies != NULL);

  srand((unsigned) time(NULL));

#pragma omp parallel
  {

#pragma omp single
    {
      start_iterations =  omp_get_wtime();

#pragma omp taskloop shared(success) grainsize(GSZ)
      for (unsigned i = 0; i < N; i++) {
	const int dev = gpu_scheduler(occupancies, ndevs);
	const int work = rand() % MAXWORK;

	//   printf("work=%d\n", work);
	
	#pragma omp task depend(out: success[i]) 
	{

	success[i] = 0;
#pragma omp atomic
	occupancies[dev]++; // may have to do atomic 
      }

#pragma omp target device(dev)                      \
  map(tofrom: success[i:1], occupancies[dev:1], task_counts[dev:1]) \
  depend(inout: success[i])
	{
	
	  task_counts[dev]++;
	  work_generator(work);
	  success[i] = 1;
	} // end target

	printf (" occ[dev] %d \n", occupancies[dev]); 
	#pragma omp task depend(in: success[i]) 
	{
	#pragma omp atomic
	  occupancies[dev]--; // may have to do atomic 
       	}
	
      } // end taskloop
    } // end of single

#pragma omp master
        {
	  int check = 0;
	  end_iterations =  omp_get_wtime();
	  for (unsigned i = 0; i < N; i++) {
	    check += success[i];
	  }
	  if (check != N) {
	    printf("failed\n");
	  }
	  
	  printf("Statistics for %d iterations:\n", N);
	  for (unsigned i = 0; i < ndevs; i++) {
	printf("# tasks of device[%d] = %d\n",
	       i,
	       task_counts[i]);
	  }
	  printf("Loop took %f seconds\n",
		 end_iterations - start_iterations);
	printf("Total number of CPU threads = %d\n",
	       omp_get_num_threads());
	}
  } // end parallel
  
  free(occupancies);
  free(task_counts);
  
  return 0;
} // end main
