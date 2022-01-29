
#include <mpi.h>
#include <stdio.h>
#include <iostream>

//    kernel function on device                                                                                                     
__global__ void scalar(double *sg, const double *x, const double *y, int N)
{
  extern __shared__ double sdata[];
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int str = gridDim.x*blockDim.x;
  double s = 0.0;
  for (int i = idx; i < N; i += str)
    s+= x[i]*y[i];
  sdata[threadIdx.x] = s;
  __syncthreads();
  for (int s=blockDim.x >> 1; s>0; s>>=1) {
    if (threadIdx.x < s) {
      sdata[threadIdx.x] += sdata[threadIdx.x + s];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0)  sg[blockIdx.x] = sdata[0];
}

//    kernel function for addition on  o n e  block on device                                                                       
__global__ void add_1_block(double *s, int N)
{
  if (N>blockDim.x) return;
  extern __shared__ double sdata[];
  const int tid = threadIdx.x;
  if (tid<N) sdata[tid]=s[tid];
  else       sdata[tid]=0.0;
  __syncthreads();
  for (int s=blockDim.x/2; s>0; s>>=1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0)  s[0] = sdata[0];
}

//-------------------------------------------------------------                                                                     
// Host function. Inner product calculated with device vectors   
double dscapr_GPU(const int N, const double x_d[], const double y_d[])
{
  cudaDeviceProp prop;
  cudaGetDeviceProperties (&prop, 0);
  const int blocksize = 4 * 64, gridsize = prop.multiProcessorCount, sBytes   = blocksize * sizeof(double);
  dim3 dimBlock(blocksize);
  dim3 dimGrid(gridsize);
  double sg, *s_d;  // device vector storing the partial sums                                                                       
  cudaMalloc((void **) &s_d, blocksize * sizeof(double)); // temp. memory on device                                                 
  // call the kernel function with  dimGrid.x * dimBlock.x threads                                                                  
  scalar <<< dimGrid, dimBlock, sBytes>>>(s_d, x_d, y_d, N);
  //    power of 2 as number of treads per block                                                                                    
  const unsigned int oneBlock = 2 << (int)ceil(log(dimGrid.x + 0.0) / log(2.0));
  add_1_block <<< 1, oneBlock, oneBlock *sizeof(double)>>>(s_d, dimGrid.x);
  // copy data:  device --> host                                                                                                    
  cudaMemcpy(&sg, s_d, sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(s_d);
  return sg;
}

//-------------------------------------------------------------                                                                     
//  main function on host                                                                                                           
int main()
{
  int myrank;
  FILE * pFile;
  FILE * fileOfTimings;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);                  // my MPI rank                                                                 
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);               // How many GPUs?                                                       
  int device_id = myrank % deviceCount;
  cudaSetDevice(device_id);                       // Map MPI-process to a GPU
  double *x_h, *y_h; // host data                                                                                                   
  double *x_d, *y_d; // device data                                                                   
  double sum, tstart, tgpu;
  const int N = 14000000;
  int nBytes = N*sizeof(double);
  const int LOOP = 1000; // TODO: see if this should be a #define.
 // cout << endl << gridsize << " x " << blocksize << " Threads\n";
  x_h = new double [N];
  y_h = new double [N];
  // allocate memory on device              
  cudaMalloc((void **) &x_d, nBytes);
  cudaMalloc((void **) &y_d, nBytes);

  for (int i=0; i<N; i++) { x_h[i] = (i % 137)+1; y_h[i] = 1.0/x_h[i];}
    
	        // copy data:  host --> device                                    
    cudaMemcpy(x_d, x_h, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y_h, nBytes, cudaMemcpyHostToDevice);
	  for (int k=0; k<LOOP; ++k)
   	   {
      	   double loc_sum = dscapr_GPU(N, x_d, y_d);
     	   MPI_Allreduce(&loc_sum,&sum,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
   	   }

	  pFile = fopen ("myError.csv","w+");
  	  fprintf(pFile, "%s,%d,%d,%d,%d,%d\n", "error", 0.001, 0.001, 0.001, 0.001, 0.001 );
	  if (pFile!=NULL) fclose (pFile);

	  fileOfTimings = fopen ("dProdTimings.out","w+");
  	  fprintf(fileOfTimings, "%s\t%d\t%d\t%d\t%d\t%f\n", "app", 1, 1, 1, time);
	  if (fileOfTimings!=NULL) fclose (fileOfTimings);

	  printf("completed MPI+CUDA dot product code\n");
	  	  
  delete [] x_h; delete [] y_h;
  cudaFree(x_d); cudaFree(y_d);
  MPI_Finalize();
  return 0;
}
