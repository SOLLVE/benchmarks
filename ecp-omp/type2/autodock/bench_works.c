#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <math.h>
#include <omp.h>

#include "timing/ompvv_timing.h"

#define EPLB
#define N 500
#define MAXWORK 10


// Scheduling strategies, unset all to use the compact schedue
//#define SCHED_DYNAMIC
//#define SCHED_RANDOM
//#define SCHED_ROUNDROBIN

#define MAX_TIMESTEPS 100

//#define CPU_TEST

//#define RANDOM_SIZED_TASKS
#define INCREASING_SIZED_TASKS

//#define DEBUG 1

inline unsigned
gpu_scheduler_compact(unsigned *occupancies, int taskID, int ngpus, int numTasks)
{
  const unsigned chosen = (unsigned)(taskID * ngpus / (float)numTasks);
#pragma omp atomic
  occupancies[chosen]++;
  return chosen;
}

inline unsigned
gpu_scheduler_roundrobin(unsigned *occupancies, int taskID, int ngpus)
{
  const unsigned chosen = taskID % ngpus;
#pragma omp atomic
  occupancies[chosen]++;
  return chosen;
}

inline unsigned
gpu_scheduler_random(unsigned *occupancies, int ngpus)
{
  const unsigned chosen = rand() % ngpus;
#pragma omp atomic
  occupancies[chosen]++;
  return chosen;
}

inline unsigned
gpu_scheduler(unsigned *occupancies, int ngpus)
{
  short looking = 1;
  unsigned chosen;
  while (looking) {
    for (unsigned i = 0; i < ngpus; i++) {
      // But really, this should be a single atomic compare-and-swap
      unsigned occ_i;
      #pragma omp atomic read
      occ_i = occupancies[i];
      if (occ_i == 0) {
        chosen = i;
#pragma omp atomic
        occupancies[chosen]++;
        looking = 0;
        break;
      }
    }
  }
  return chosen;
}


inline unsigned
gpu_scheduler_multQueue(unsigned *occupancies, int omp_tid, int omp_nthrds, int ngpus)
{
  short looking = 1;
  unsigned chosen;
  while (looking) {
    for (unsigned i = (omp_tid/omp_nthrds)*ngpus; i < ((omp_tid + 1)/omp_nthrds)*ngpus; i++) {
      unsigned occ_i;
      #pragma omp atomic read
      occ_i = occupancies[i];
      if (occ_i == 0) {
        chosen = i;
#pragma omp atomic
        occupancies[chosen]++;
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
    
  if(NN> MAXWORK) printf("out of bounds NN\n");

  for (i = 0; i < NN; i++)
    {
      for (j = 0; j < NN; j++)
        {
          a[i][j] = i;
          b[i][j] = j;
        }
    }

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
  if(NN>0) 
    return c[NN-1][NN-1];     /* c[i][j] is bounds error */
  else
    return c[0][0];
}
#pragma omp end declare target

int main(int argc, char* argv[])
{
  OMPVV_INIT_TIMERS;
  int success[N];
  float output[N];
  float boundary[N]; // for a reg. mesh
  const int ndevs = omp_get_num_devices();
  int *devices = NULL;
  double *time_devices = NULL;
  double start_iterations, end_iterations;
  unsigned *task_counts = NULL;
  unsigned *occupancies = NULL;
  unsigned *lastGPU = NULL;
  task_counts = (unsigned *) calloc(ndevs, sizeof(*task_counts));
  assert(task_counts != NULL);
  occupancies = (unsigned *) calloc(ndevs, sizeof(*occupancies));
  assert(occupancies != NULL);
  int timestep = 0;
  int probSize = MAXWORK; 
  int num_timesteps = 1;
  int numThreads = 1;
  int numTasks = N;
  int gsz = 1;

  /* make sure we have some GPUs */
  assert(ndevs > 0);
  printf("There are %d GPUs\n", ndevs);
  devices = (int *) calloc(ndevs, sizeof(*devices));
  assert(devices != NULL);
  time_devices = (double *) calloc(ndevs, sizeof(*devices));
  assert(time_devices != NULL);
  srand((unsigned) time(NULL));
  if(argc <= 1) 
    {
      printf("Usage bench_works [pSize] [numTasks][gsz]  [numThreads]\n" );
      printf("Using default parameters\n" );
      probSize = MAXWORK; 
      num_timesteps = 1; 
#pragma omp parallel 
      numThreads = omp_get_num_threads();
      numTasks = N;
      gsz = 1;
    }
  else
    {
      if (argc > 1)
	probSize = atoi(argv[1]);
      if (argc > 2)
	numTasks = atoi(argv[2]);
      if (argc > 3)
	gsz = atoi(argv[3]);
      if (argc > 4)
	num_timesteps = atoi(argv[4]);
    } 
  printf("bench_works [pSize=%d] [numTasks=%d] [gsz=%d] [num_timesteps=%d] [numThreads=%d] \n", probSize, numTasks, gsz, num_timesteps, numThreads);
  int arrSize = probSize*probSize;
  float* a = malloc(sizeof(float)*arrSize);
  float* b = malloc(sizeof(float)*arrSize); 
  float* c = malloc(sizeof(float)*arrSize);
  int* taskWork = malloc(sizeof(int)*numTasks);
  int* taskWorkSquared = malloc(sizeof(int)*numTasks);
  // initialize 


#ifdef EPLB
  for(int i = 0; i< arrSize; i++) 
    {
      a[i] = 3.0;
      b[i] = 2.0;
      c[i] = 0.0;
    }
#endif

#ifdef MATMUL
  int imat, jmat;  
  for (imat = 0; imat < probSize; imat++)
    {
      for (jmat = 0; jmat < probSize; jmat++)
        {
          a[imat*probSize + jmat] = imat;
          b[imat*probSize + jmat] = jmat;
	  c[imat*probSize +jmat] = 0.0;
        }
    }
#endif

  int ctaskwork;
  for (int i =0 ; i < numTasks; i++)
    {
#ifdef RANDOM_SIZED_TASKS
      ctaskwork =  1 + (rand()%probSize -1); 
#else 
#ifdef INCREASING_SIZED_TASKS
      ctaskwork = 1 + i*((probSize - 1)/numTasks); // maybe could be MAXWORK/N rather than div by 2  - create an increasing sized tasks
#else
      ctaskwork = (probSize-1)/2; // maybe could be MAXWORK/TotWork rather than div by 2                              
#endif
#endif 
      taskWork[i] = ctaskwork;
      taskWorkSquared[i] = ctaskwork*ctaskwork;
    }
  double cpu_time = 0.0;
  double task_time = 0.0;

#ifdef CPU_TEST
  cpu_time = -omp_get_wtime();

#pragma omp parallel
  {
#pragma omp taskloop grainsize(gsz) private(task_time)
    for (int i = 0; i < numTasks; i++)
      {
	task_time = - omp_get_wtime();
	for (int j = 0; j < taskWorkSquared[i] ; j++)
	  c[j] = sqrt(a[j]*b[j]);
	task_time += omp_get_wtime();
#if defined DEBUG && DEBUG == 1
	printf("Mat size %d \t task_time for comp: %f\n", taskWork[i], task_time);
#endif
      }
  }
  cpu_time += omp_get_wtime();
  printf("cpu_time for comp: %f\n", cpu_time);
#endif

  while(timestep < num_timesteps)
    {
#pragma omp parallel
      {
	// only one thread deals out work to GPUs awith single - other threads could do it too though 
#pragma omp single
	{
	  start_iterations =  omp_get_wtime();
#pragma omp taskloop shared(success) grainsize(gsz)
	  for (int i = 0; i < numTasks; i++) {
	    
#if defined(SCHED_RANDOM)
	    const int dev = gpu_scheduler_random(occupancies, ndevs);
#elif defined(SCHED_ROUNDROBIN)
	    const int dev = gpu_scheduler_roundrobin(occupancies, i, ndevs);
#elif defined(SCHED_DYNAMIC)
	    const int dev = gpu_scheduler(occupancies, ndevs);
#else
	    const int dev = gpu_scheduler_compact(occupancies, i, ndevs, numTasks);
#endif

	    output[i] = 0;
#pragma omp task depend(out: success[i])
	    {
	      success[i] = 0;
	      // #pragma omp atomic
	      // occupancies[dev]++; // Moved to the inside of the scheduler functions
	      // #pragma omp atomic
	      // occupancies[dev] = 1
	    }
#pragma omp task depend(inout:success[i])
	    {
	      OMPVV_START_TIMER;
#pragma omp target device(dev)\
  map(to: a[0:arrSize], b[0:arrSize], c[0:arrSize]) map(tofrom: success[i:1], devices[dev:1], time_devices[dev:1], output[i:1], boundary[i:1], taskWork[i:1], occupancies[dev:1])
	      {
		devices[dev]++;
		if(taskWork[i] > probSize) taskWork[i] = probSize;
		const int NN = taskWork[i];
		double work_start = 0; //omp_get_wtime();

		// below are different computations. MATMUL is close to Autodock and can provide complex computation
#ifdef MATMUL 
		int mat_i, mat_j;

		for (mat_i = 0; mat_i < NN; mat_i++)
		  {
		    for (mat_j = 0; mat_j < NN; mat_j++)
		      {
			c[mat_i*NN + mat_j] = 0;
			for (int k = 0; k < NN; k++)
			  {
                            c[mat_i*NN +mat_j] +=a[mat_i*NN + k]*b[k*NN + mat_j];
			  }
		      }
		  }
		if(NN>0)
		  output[i] = c[NN*(NN-1)];     /* c[i][j] is bounds error */
		else
		  output[i] = c[0];

#endif

#ifdef EPLB
		for (int j = 0 ; j< NN*NN; j++)
		  c[j] = sqrt(a[j]*b[j]); 
		output[i] = c[NN];
#endif
		// note this doesn't communicate to GPUs at this time, need to return boundary values 
#ifdef GRIDLB 
		for (int j = 1 ; j<= NN*NN; j++)
		  a[j] = (b[j] +b[j-1] +b[j+1])/3.0;
		//swap pointers a an b for update
		c=b; 
		b=a;
		a=c;
		output[i] = a[1]; 
		boundary[i] = a[NN*NN -1];
#endif
		success[i] = 1;
		double work_end = 0; // omp_get_wtime(); 
		time_devices[dev]+= work_end - work_start;
	      } // end target
	      OMPVV_STOP_TIMER;
	      // printf("iter=%d, work_out=%d\n", i, work_out); 
	    } // end task 
#pragma omp task depend(in: success[i])
	    {
#if defined DEBUG && DEBUG == 1
	      printf (" occ[%d] %d \n", dev, occupancies[dev]);
#endif      
#pragma omp atomic
	      occupancies[dev]--; // may have to do atomic 
	    } 
	    } // end taskloop
	    } // end of single

#pragma omp master
	      {
	      int check = 0;
	      end_iterations =  omp_get_wtime(); 
	      int lastFail = 0;
	      //#if defined DEBUG && DEBUG
	      //#endif
	      {
	      for (int i = 0; i < numTasks; i++) {
	      check += success[i];
	      //printf("Output[%d]=%f\t success[%d]=%d\n",i, output[i], i, success[i]);
	      if(success[i] == 0) lastFail = i;
	    }    
	    }
	      if (check != numTasks) {
	      printf("failed! LastFailed %d output[%d]=%f\n", lastFail, lastFail, output[lastFail]);
            }    
	      printf("Statistics for %d iterations:\n", numTasks);
	      for (int i = 0; i < ndevs; i++) {
	      printf("# of tasks executed on device[%d]=%d, time=%f\n", i, devices[i], time_devices[i]);
            }
	      printf("Loop took %f seconds\n", end_iterations - start_iterations);
	      printf("Total number of CPU threads=%d\n", omp_get_num_threads());
	    }
	    } // end parallel
	      timestep++;
	    } // end while 
	      free(a);
	      free(b); 
	      free(c);
	      free(devices);
	      free(time_devices);
	      return 0;
	    } // end main
