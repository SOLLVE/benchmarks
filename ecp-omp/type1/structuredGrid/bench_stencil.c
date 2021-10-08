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
    printf("bench_stencil [pSize=%d] [numTasks=%d] [gsz=%d] [num_timesteps=%d] [numThreads=%d] \n", probSize, numTasks, gsz, num_timesteps, numThreads);

    int arrSize = probSize;
    int numBlocks = numTasks;
    float* a = malloc(sizeof(float)*arrSize);
    float* b = malloc(sizeof(float)*arrSize);
    float* c = malloc(sizeof(float)*arrSize);
    int* blockWork = malloc(sizeof(int)*numBlocks);

    for (int i = 0; i< arrSize; i++)
      {
        a[i] = 3.0;
        b[i] = 2.0;
        c[i] = 0.0;
      }

    int ctaskwork;

    for (int i = 0 ; i < numBlocks; i++)
      {
        ctaskwork = (probSize - 1)/(numTasks); // maybe could be MAXWORK/TotWork rather than div by 2                                                         
        blockWork[i] = ctaskwork;
      }

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
#pragma omp for schedule(static, gsz)
      {
        for (int i = 0; i < numBlocks; i++)
          {
            int startInd = (i%(numBlocks/ndevs))*blockWork[i];
            int endInd = (i%(numBlocks/ndevs)+1)*blockWork[i];
            b[startInd] = lboundary[i];
            b[endInd-1] = rboundary[i];
            for (int j = startInd; j <= endInd ; j++)
              a[j] = (b[j] + b[j-1] + b[j+1])/3.0;
            //swap pointers a an b for update                                                                                                                 
            c=b;
            b=a;
            a=c;
            lboundary[i] = a[startInd];
            rboundary[i] = a[endInd-1];
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
#pragma omp target device(dev) map(alloc: a[0:arrSize], b[0:arrSize], numBlocks, ndevs) map(tofrom: lboundary[i:1], rboundary[i:1], blockWork[i:1]) nowait
            {
              const int NN = blockWork[i];
              const int startInd = (i%(numBlocks/ndevs))*NN; // startInd depends on global task number (global across GPUs on a node)                         
              const int endInd = (i%(numBlocks/ndevs)+1)*NN;
              // obtain boundaries for neighboring GPUs (needs to be fixed for multiple blocks for each GPU)
	      float* temp ; //temp variable
              b[startInd-1] = lboundary[i];
              b[endInd+1] = rboundary[i];
              for (int j = startInd; j<= endInd ; j++)
                a[j] = (b[j] + b[j-1] +b[j+1])/3.0;
              //swap pointers a an b for update                                                                                                               
              temp=b;
              b=a;
              a=temp;
              lboundary[i] = a[startInd-1];
              rboundary[i] = a[endInd+1];
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
