/* 
* Author: Vivek Kale 
* This benchmark performs the stream triad of operation from the stream benchmark 
*/


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <math.h>
#include <omp.h>


#include "timing/ompvv_timing.h"

#define N 500
#define MAXWORK 10

#define MAX_TIMESTEPS 100

//#define CPU_TEST                                                                                                                                            

int main(int argc, char* argv[])
{
    OMPVV_INIT_TIMERS;
    float lboundary[N];
    float rboundary[N]; // for a reg. mesh                                                                                                                    
    const int ndevs = omp_get_num_devices();
    int *devices = NULL;
    double *time_devices = NULL;
    double start_iterations, end_iterations;
    int timestep = 0;
    int probSize = MAXWORK;
    int num_timesteps = 1;
    int numThreads = 1;
    int numTasks = N;
    int gsz = 1;

    /* make sure we have some GPUs */
    assert(ndevs > 0);
    printf("There are %d GPUs\n", ndevs);
    srand((unsigned) time(NULL));
    if(argc <= 1)
      {
        printf("Usage bench_stencil [pSize] [numBlocks] [chunkSize] [numTimesteps]\n" );
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
	  num_timesteps = atoi(argv[2]);
        if (argc > 3)
          numTasks = atoi(argv[3]);
        if (argc > 4)
          gsz = atoi(argv[4]);
      }
    printf("bench [pSize=%d] [numTasks=%d] [gsz=%d] [num_timesteps=%d] [numThreads=%d] \n", probSize, numTasks, gsz, num_timesteps, numThreads);

    int arrSize = probSize;
    int numBlocks = numTasks;
    float* a = malloc(sizeof(float)*arrSize);
    float* b = malloc(sizeof(float)*arrSize);
    float* c = malloc(sizeof(float)*arrSize);
    float alpha = 10.453;
    int* blockWork = malloc(sizeof(int)*numBlocks);
  
  
    for (int i = 0; i< arrSize; i++)
      {
        a[i] = 3.0;
        b[i] = 2.0;
        c[i] = 0.0;
      }

    int ctaskwork = (probSize - 1)/(numTasks);

 // TODO: consider blockWork[i], blockWorkStart[i] and blockWorkEnd[i] for true tasking support  
    // Or could use a binary search like algorithm for a random partition of the array 

//    for (int i = 1 ; i < numBlocks; i++)
 //   {
  //      ctaskwork = (probSize - 1)/(numTasks); // maybe could be MAXWORK/TotWork rather than div by 2   // uniform blockwork    
   //    blockWork[i] = ctaskwork;
    //  blockWorkStart[i] = blockWorkEnd[i-1];
    //  blockWorkEnd[i] = blockWork[i] + blockWorkStart[i]; 
    // }
  
    int numCores = 0;
    double cpu_time = 0.0;
    double task_time = 0.0;

#ifdef CPU_TEST
    cpu_time = -omp_get_wtime();

    float* temp;
#pragma omp parallel
    int numCores = omp_get_num_threads();

#pragma omp parallel
    {
#pragma omp for schedule(static, ctaskwork)
      {
        for (int j = 0; j < probSize; j++)
                 c[j] += a[j] + alpha*b[j];
       }
    }
      cpu_time += omp_get_wtime();
      printf("cpu_time for comp: %f\n", cpu_time);
#endif

while(timestep < num_timesteps)
  {
#pragma omp parallel
    { 
#pragma omp for schedule(static, gsz)
        for (int i = 0; i < numBlocks; i++) {
          //const int dev = (int) ((i/numBlocks)*ndevs); // use for static schedule 
          
	  const int dev = i%ndevs; 
	  printf("device chosen for iteration %d : %d\n" , i, dev);
          OMPVV_START_TIMER;
#pragma omp target teams distribute parallel for device(dev) map(alloc: c[0:arrSize], a[0:arrSize], b[0:arrSize], numBlocks, ndevs, ctaskwork) nowait
            {
       
              const int NN = ctaskwork;
              const int startInd = (i%(numBlocks/ndevs))*NN; // startInd depends on global task number (global across GPUs on a node)                         
              const int endInd = (i%(numBlocks/ndevs)+1)*NN;
              for (int j = startInd; j<= endInd; j++)
                c[j] += a[j] + alpha*b[j];
            } // end target 
            OMPVV_STOP_TIMER;
        } // end for      
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
