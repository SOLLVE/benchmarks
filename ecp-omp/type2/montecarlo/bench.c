#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

#define N 100
#define MAXWORK 100000

int numOutputsReady;
double averageError;

#define GSZ 1
#define MAXOUTER 10

//#define DEBUG 

#define RANDOM_SIZED

//#define INCREASING_SIZED
/* Variables for strategy */ 

int currentGPUCount =0;

double totalOvhd;



/*
 * work out which GPU to use
 */
inline unsigned
gpu_scheduler_roundrobin(int taskID, int ngpus)
{
  return taskID%ngpus;
}

inline unsigned
gpu_scheduler_random(int ngpus)
{
     return rand()%ngpus;
}

inline unsigned
gpu_scheduler(const unsigned *occupancies, int ngpus)
{
  short looking = 1;
  unsigned chosen;
  //  return rand()%ngpus;
  //  return (i % ngpus);

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


// TODO: consider multiple queues for GPUs
//inline unsigned
//gpu_scheduler_fair(const unsigned *task_counts, int ngpus)
//{
// int minimum = task_counts[0];  
  
//  for ( ) 

//  return minimum;
// return indexWithMinVal(task_counts, ngpus);
//}

inline unsigned
gpu_scheduler_locality(const unsigned *occupancies, int ngpus)
{
     return rand()%ngpus;
}

// TODO: create timestep loop ? Can do locality-aware scheduling then .
#pragma omp declare target
int
work_generator(int work)
{
  const int NN = work;
  int a[MAXWORK][MAXWORK], b[MAXWORK][MAXWORK], c[MAXWORK][MAXWORK];
  int i, j, outer1;

  if(NN> MAXWORK) {printf("out of bounds NN\n");}

  for (i = 0; i < NN; i++)
    {
      for (j = 0; j < NN; j++)
	{	  
	  a[i][j] = i;
	  b[i][j] = j;
	}
    }

  for (outer1 = 0; outer1 < MAXOUTER; outer1++)
    {
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
    }

  if(NN>0)
    return c[NN-1][NN-1];     /* c[i][j] is bounds error */
  else
    return c[0][0];

  //  if(NN == 0)
    
  //return c[i - 1][j - 1];     /* c[i][j] is bounds error */
}

#pragma omp end declare target

int main()
{
  unsigned success[N];
  const int ndevs = omp_get_num_devices();
  unsigned *task_counts = NULL;
  unsigned *occupancies = NULL;
  unsigned *lastGPU = NULL;
  double start_iterations, end_iterations;

  int output;

  /* make sure we have some GPUs */
  assert(ndevs > 0);

  printf("There are %d GPUs\n", ndevs);

  task_counts = (unsigned *) calloc(ndevs, sizeof(*task_counts)); 
  assert(task_counts != NULL);
  occupancies = (unsigned *) calloc(ndevs, sizeof(*occupancies)); 
  assert(occupancies != NULL);

  lastGPU = (unsigned *) calloc(N/GSZ, sizeof(*lastGPU));
  assert(occupancies != NULL);

  srand((unsigned) time(NULL));
  /* Variables for Performance Profiling */
  double tasktime;
  double comptime;


#pragma omp parallel 
  {

#pragma omp single
    {
      start_iterations =  omp_get_wtime();

#pragma omp taskloop shared(success) grainsize(GSZ)  private(tasktime, comptime)
      for (unsigned i = 0; i < N; i++) {

	tasktime = - omp_get_wtime();

	/** Call scheduling strategy here **/ 
	// const int dev = ndevs/2;
	// const int dev = i%(ndevs); 
	// const int dev = rand()%(ndevs); 
	// const int dev = gpu_scheduler_roundrobin(i, ndevs);
	const int dev = gpu_scheduler(occupancies, ndevs);


#ifdef RANDOM_SIZED 
	const int work =  (rand() % MAXWORK);
#else

#ifdef INCREASING_SIZED
	const int work =  i*(MAXWORK/N); // maybe could be MAXWORK/N rather than div by 2
#else
	const int work = MAXWORK/2; // maybe could be MAXWORK/TotWork rather than div by 2
#endif
#endif

	#ifdef DEBUG
	 printf("work=%d\n", work);
	 #endif
	#pragma omp task depend(out: success[i]) 
	 {
	   success[i] = 0; 
#pragma omp atomic
	   occupancies[dev]++; // may have to do atomic 
	   //  #pragma omp atomic
	   //occupancies[dev] = 1
	 }
	 
#pragma omp task depend(inout:success[i])
	 {
#pragma omp target device(dev)						\
  map(tofrom: success[i:1], occupancies[dev:1], task_counts[dev:1], output)
	   {	
	     task_counts[dev]++; // get time of target region 
	     //	comptime = -omp_get_wtime();
	     output = work_generator(work);
	     //comptime += omp_get_wtime();
	     success[i] = 1; 
	   } // end target
	 }
	 
#pragma omp task depend(in: success[i]) 
	 {
	   
#ifdef DEBUG
	   printf (" occ[%d] %d \n", dev, occupancies[dev]); 
#endif
#pragma omp atomic
	   occupancies[dev]--; // may have to do atomic 
	   
	   //#pragma omp atomic
	   //numOutputsReady++;
#ifdef DEBUG
	   printf("Output%d\n", output);
#endif
	  //averageError = (output + averageError)/numOutputsReady;
	 }

	 tasktime += omp_get_wtime(); 

        #ifdef DEBUG
	 printf("Time to do task %d is : %f\n" , i, tasktime);
	 printf("Overhead of task %d is : %f\n" , i, tasktime - comptime);
	 printf("Comp time of task %d is : %f\n" , i, comptime);
#endif
	 totalOvhd += (tasktime - comptime); 
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

	  printf("Average error: %f \n", averageError);
	  printf("Total ovhd (dequeue and cache) was %f seconds\n",
		 totalOvhd );

	printf("Total number of CPU threads = %d\n",
	       omp_get_num_threads());
	}
  } // end parallel 

  free(occupancies);
  free(task_counts);
  free(lastGPU);
  
  return 0;
} // end main


/* Utility functions */ 

// Used for fair scheduling 
// int addressWithMinVal(int* myArr , int arrSz)
// {
// int minimum = myArr[0]; 
  //  minimum = myArr;
  //  *minimum = *myArr;

  //  for (int c = 1; c < arrSz; c++)
  // {
  //   if (*(myArr+c) < *minimum)
  //     {
//	  *minimum = *(array+c);
	  //  location = c+1;
  //      }
    //  }
//}

int indexWithMinIntVal(int* a, int n) 
{
  int c, min, index;  
  min = a[0];
  index = 0; 
 
  for (c = 1; c < n; c++) {
    if (a[c] < min) {
      index = c;
      min = a[c];
    }
  }
}
