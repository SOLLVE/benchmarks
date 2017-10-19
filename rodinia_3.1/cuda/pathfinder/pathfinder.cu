#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <helper_timer.h>

#define BLOCK_SIZE 256
#define STR_SIZE 256
#define DEVICE 0
#define HALO 1 // halo width along one direction when advancing to the next iteration

#define CUDA_UVM

//#define BENCH_PRINT

void run(int argc, char** argv);

unsigned long long rows, cols;
int* data;
int** wall;
int* result;
#define M_SEED 9
int pyramid_height;

//#define BENCH_PRINT

void
init(int argc, char** argv)
{
	if(argc==4){
		cols = atoi(argv[1]);
		rows = atoi(argv[2]);
                pyramid_height=atoi(argv[3]);
	}else{
                printf("Usage: dynproc row_len col_len pyramid_height\n");
                exit(0);
        }
#ifndef CUDA_UVM
	data = new int[rows*cols];
#else
    cudaMallocManaged((void**)&data, sizeof(int)*rows*cols);
#endif
	wall = new int*[rows];
	for(unsigned long long n=0; n<rows; n++)
		wall[n]=data+cols*n;
#ifndef CUDA_UVM
	result = new int[cols];
#endif

	int seed = M_SEED;
	srand(seed);

	for (unsigned long long i = 0; i < rows; i++)
    {
        for (unsigned long long j = 0; j < cols; j++)
        {
            wall[i][j] = rand() % 10;
        }
    }
#ifdef BENCH_PRINT
    for (unsigned long long i = 0; i < rows; i++)
    {
        for (unsigned long long j = 0; j < cols; j++)
        {
            printf("%d ",wall[i][j]) ;
        }
        printf("\n") ;
    }
#endif
}

void 
fatal(char *s)
{
	fprintf(stderr, "error: %s\n", s);

}

#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))

__global__ void dynproc_kernel(
                int iteration, 
                int *gpuWall,
                int *gpuSrc,
                int *gpuResults,
                unsigned long long cols, 
                unsigned long long rows,
                int startStep,
                int border)
{

        __shared__ int prev[BLOCK_SIZE];
        __shared__ int result[BLOCK_SIZE];

	int bx = blockIdx.x;
	int tx=threadIdx.x;
	
        // each block finally computes result for a small block
        // after N iterations. 
        // it is the non-overlapping small blocks that cover 
        // all the input data

        // calculate the small block size
	int small_block_cols = BLOCK_SIZE-iteration*HALO*2;

        // calculate the boundary for the block according to 
        // the boundary of its small block
        int blkX = small_block_cols*bx-border;
        int blkXmax = blkX+BLOCK_SIZE-1;

        // calculate the global thread coordination
	int xidx = blkX+tx;
       
        // effective range within this block that falls within 
        // the valid range of the input data
        // used to rule out computation outside the boundary.
        int validXmin = (blkX < 0) ? -blkX : 0;
        int validXmax = (blkXmax > cols-1) ? BLOCK_SIZE-1-(blkXmax-cols+1) : BLOCK_SIZE-1;

        int W = tx-1;
        int E = tx+1;
        
        W = (W < validXmin) ? validXmin : W;
        E = (E > validXmax) ? validXmax : E;

        bool isValid = IN_RANGE(tx, validXmin, validXmax);

	if(IN_RANGE(xidx, 0, cols-1)){
            prev[tx] = gpuSrc[xidx];
	}
	__syncthreads(); // [Ronny] Added sync to avoid race on prev Aug. 14 2012
        bool computed;
        for (int i=0; i<iteration ; i++){ 
            computed = false;
            if( IN_RANGE(tx, i+1, BLOCK_SIZE-i-2) &&  \
                  isValid){
                  computed = true;
                  int left = prev[W];
                  int up = prev[tx];
                  int right = prev[E];
                  int shortest = MIN(left, up);
                  shortest = MIN(shortest, right);
                  int index = cols*(startStep+i)+xidx;
                  result[tx] = shortest + gpuWall[index];
	
            }
            __syncthreads();
            if(i==iteration-1)
                break;
            if(computed)	 //Assign the computation range
                prev[tx]= result[tx];
	    __syncthreads(); // [Ronny] Added sync to avoid race on prev Aug. 14 2012
      }

      // update the global memory
      // after the last iteration, only threads coordinated within the 
      // small block perform the calculation and switch on ``computed''
      if (computed){
          gpuResults[xidx]=result[tx];		
      }
}

/*
   compute N time steps
*/
int calc_path(int *gpuWall, int *gpuResult[2], unsigned long long rows, unsigned long long cols, \
	 int pyramid_height, int blockCols, int borderCols)
{
        dim3 dimBlock(BLOCK_SIZE);
        dim3 dimGrid(blockCols);  
	
        int src = 1, dst = 0;
	for (unsigned long long t = 0; t < rows-1; t+=pyramid_height) {
            int temp = src;
            src = dst;
            dst = temp;
            dynproc_kernel<<<dimGrid, dimBlock>>>(
                MIN(pyramid_height, rows-t-1), 
                gpuWall, gpuResult[src], gpuResult[dst],
                cols,rows, t, borderCols);
	}
        return dst;
}

int main(int argc, char** argv)
{
    int num_devices;
    cudaGetDeviceCount(&num_devices);
    if (num_devices > 1) cudaSetDevice(DEVICE);

    run(argc,argv);

    return EXIT_SUCCESS;
}

void run(int argc, char** argv)
{
    init(argc, argv);

    /* --------------- pyramid parameters --------------- */
    int borderCols = (pyramid_height)*HALO;
    int smallBlockCol = BLOCK_SIZE-(pyramid_height)*HALO*2;
    int blockCols = cols/smallBlockCol+((cols%smallBlockCol==0)?0:1);

    printf("pyramidHeight: %d\ngridSize: [%d]\nborder:[%d]\nblockSize: %d\nblockGrid:[%d]\ntargetBlock:[%d]\n",\
	pyramid_height, cols, borderCols, BLOCK_SIZE, blockCols, smallBlockCol);
	
    int *gpuWall, *gpuResult[2];
    unsigned long long size = rows*cols;

#ifndef CUDA_UVM
    cudaMalloc((void**)&gpuResult[0], sizeof(int)*cols);
    cudaMalloc((void**)&gpuResult[1], sizeof(int)*cols);
    cudaMalloc((void**)&gpuWall, sizeof(int)*(size-cols));
#else
    cudaMallocManaged((void**)&gpuResult[0], sizeof(int)*cols);
    cudaMallocManaged((void**)&gpuResult[1], sizeof(int)*cols);
    for (unsigned long long i = 0; i < cols; i++)
      gpuResult[0][i] = data[i];
    gpuWall = data+cols;
#endif
    unsigned long long total_size = sizeof(int)*(size-cols) + sizeof(int)*cols + sizeof(int)*cols;
    printf("Total size: %llu\n", total_size);
	StopWatchInterface *timer = 0;
	sdkCreateTimer(&timer); 
	sdkStartTimer(&timer); 
#ifndef CUDA_UVM
    cudaMemcpy(gpuResult[0], data, sizeof(int)*cols, cudaMemcpyHostToDevice);
    cudaMemcpy(gpuWall, data+cols, sizeof(int)*(size-cols), cudaMemcpyHostToDevice);
#endif


    int final_ret = calc_path(gpuWall, gpuResult, rows, cols, \
	 pyramid_height, blockCols, borderCols);

#ifndef CUDA_UVM
    cudaMemcpy(result, gpuResult[final_ret], sizeof(int)*cols, cudaMemcpyDeviceToHost);
#else
    cudaDeviceSynchronize();
    result = gpuResult[final_ret];
#endif
	sdkStopTimer(&timer); 
	printf("Time: %f\n", (sdkGetAverageTimerValue(&timer)/1000.0));


#ifdef BENCH_PRINT
    for (unsigned long long i = 0; i < cols; i++)
            printf("%d ",data[i]) ;
    printf("\n") ;
    for (unsigned long long i = 0; i < cols; i++)
            printf("%d ",result[i]) ;
    printf("\n") ;
#endif


#ifndef CUDA_UVM
    cudaFree(gpuWall);
    cudaFree(gpuResult[0]);
    cudaFree(gpuResult[1]);

    delete [] data;
    delete [] wall;
    delete [] result;
#else
    cudaFree(gpuResult[0]);
    cudaFree(gpuResult[1]);

    cudaFree(data);
    delete [] wall;
#endif

}

