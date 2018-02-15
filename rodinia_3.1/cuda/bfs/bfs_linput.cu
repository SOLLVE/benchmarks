/***********************************************************************************
  Implementing Breadth first search on CUDA using algorithm given in HiPC'07
  paper "Accelerating Large Graph Algorithms on the GPU using CUDA"

  Copyright (c) 2008 International Institute of Information Technology - Hyderabad. 
  All rights reserved.

  Permission to use, copy, modify and distribute this software and its documentation for 
  educational purpose is hereby granted without fee, provided that the above copyright 
  notice and this permission notice appear in all copies of this software and that you do 
  not sell the software.

  THE SOFTWARE IS PROVIDED "AS IS" AND WITHOUT WARRANTY OF ANY KIND,EXPRESS, IMPLIED OR 
  OTHERWISE.

  Created by Pawan Harish.
 ************************************************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <helper_timer.h>

//#define CUDA_UVM
//#define CUDA_HST
#define CUDA_HYB
#define CUDA_HYB_HOST // This only works when CUDA_HYB is also defined
#define CUDA_HYB_L // This only works when CUDA_HYB and CUDA_HYB_HOST are also defined
//#define CUDA_HYS // Hybrid allocation for small inputs

//#define UVM_OVER

#define MAX_THREADS_PER_BLOCK 512

//int no_of_nodes;
//int edge_list_size;
//FILE *fp;
unsigned long long no_of_nodes;
unsigned long long edge_list_size;

//Structure to hold a node information
struct Node
{
	unsigned long long starting;
	unsigned long long no_of_edges;
};

#include "kernel.cu"
#include "kernel2.cu"

void BFSGraph(int argc, char** argv);

////////////////////////////////////////////////////////////////////////////////
// Main Program
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{
	no_of_nodes=0;
	edge_list_size=0;
	BFSGraph( argc, argv);
}

void Usage(int argc, char**argv){

fprintf(stderr,"Usage: %s <input_file>\n", argv[0]);

}
////////////////////////////////////////////////////////////////////////////////
//Apply BFS on a Graph using CUDA
////////////////////////////////////////////////////////////////////////////////
void BFSGraph( int argc, char** argv) 
{

    //char *input_f;
	if(argc!=2){
	Usage(argc, argv);
	exit(0);
	}
	
	//input_f = argv[1];
	//printf("Reading File\n");
	////Read in Graph from a file
	//fp = fopen(input_f,"r");
	//if(!fp)
	//{
	//	printf("Error Reading graph file\n");
	//	return;
	//}

	int source = 0;

	//fscanf(fp,"%d",&no_of_nodes);
    no_of_nodes = atoi(argv[1]);

	unsigned long long num_of_blocks = 1;
	unsigned long long num_of_threads_per_block = no_of_nodes;

	//Make execution Parameters according to the number of nodes
	//Distribute threads across multiple Blocks if necessary
	if(no_of_nodes>MAX_THREADS_PER_BLOCK)
	{
		num_of_blocks = (unsigned long long)ceil(no_of_nodes/(double)MAX_THREADS_PER_BLOCK); 
		num_of_threads_per_block = MAX_THREADS_PER_BLOCK; 
	}

#if defined (CUDA_UVM)
	Node* h_graph_nodes;
	cudaMallocManaged( (void**) &h_graph_nodes, sizeof(Node)*no_of_nodes) ;
	bool *h_graph_mask;
	cudaMallocManaged( (void**) &h_graph_mask, sizeof(bool)*no_of_nodes) ;
	bool *h_updating_graph_mask;
	cudaMallocManaged( (void**) &h_updating_graph_mask, sizeof(bool)*no_of_nodes) ;
	bool *h_graph_visited;
	cudaMallocManaged( (void**) &h_graph_visited, sizeof(bool)*no_of_nodes) ;
#elif defined (CUDA_HST)
	Node* h_graph_nodes;
	cudaMallocHost( (void**) &h_graph_nodes, sizeof(Node)*no_of_nodes) ;
	bool *h_graph_mask;
	cudaMallocHost( (void**) &h_graph_mask, sizeof(bool)*no_of_nodes) ;
	bool *h_updating_graph_mask;
	cudaMallocHost( (void**) &h_updating_graph_mask, sizeof(bool)*no_of_nodes) ;
	bool *h_graph_visited;
	cudaMallocHost( (void**) &h_graph_visited, sizeof(bool)*no_of_nodes) ;
#elif defined (CUDA_HYB)
	// allocate host memory
	bool *h_graph_mask = (bool*) malloc(sizeof(bool)*no_of_nodes);
	bool *h_updating_graph_mask = (bool*) malloc(sizeof(bool)*no_of_nodes);
#if defined (CUDA_HYB_L)
	Node* h_graph_nodes;
	cudaMallocManaged( (void**) &h_graph_nodes, sizeof(Node)*no_of_nodes) ;
	bool *h_graph_visited;
	cudaMallocManaged( (void**) &h_graph_visited, sizeof(bool)*no_of_nodes) ;
#else
	Node* h_graph_nodes = (Node*) malloc(sizeof(Node)*no_of_nodes);
	bool *h_graph_visited = (bool*) malloc(sizeof(bool)*no_of_nodes);
#endif
#elif defined (CUDA_HYS)
	Node* h_graph_nodes;
	cudaMallocManaged( (void**) &h_graph_nodes, sizeof(Node)*no_of_nodes) ;
	bool *h_graph_mask = (bool*) malloc(sizeof(bool)*no_of_nodes);
	bool *h_updating_graph_mask = (bool*) malloc(sizeof(bool)*no_of_nodes);
	bool *h_graph_visited;
	cudaMallocManaged( (void**) &h_graph_visited, sizeof(bool)*no_of_nodes) ;
#else
	// allocate host memory
	Node* h_graph_nodes = (Node*) malloc(sizeof(Node)*no_of_nodes);
	bool *h_graph_mask = (bool*) malloc(sizeof(bool)*no_of_nodes);
	bool *h_updating_graph_mask = (bool*) malloc(sizeof(bool)*no_of_nodes);
	bool *h_graph_visited = (bool*) malloc(sizeof(bool)*no_of_nodes);
#endif

	unsigned long long start, edgeno;   
    start = 0;
	// initalize the memory
	for( unsigned long long i = 0; i < no_of_nodes; i++) 
	{
		//fscanf(fp,"%d %d",&start,&edgeno);
        edgeno = rand()%10+1;
		h_graph_nodes[i].starting = start;
		h_graph_nodes[i].no_of_edges = edgeno;
		h_graph_mask[i]=false;
		h_updating_graph_mask[i]=false;
		h_graph_visited[i]=false;
        start += edgeno;
	}

	//read the source node from the file
	//fscanf(fp,"%d",&source);
	source=0;

	//set the source node as true in the mask
	h_graph_mask[source]=true;
	h_graph_visited[source]=true;

	//fscanf(fp,"%d",&edge_list_size);
    edge_list_size = start;

	//int id,cost;
#if defined (CUDA_UVM) || defined (CUDA_HYS)
	unsigned long long* h_graph_edges;
	cudaMallocManaged( (void**) &h_graph_edges, sizeof(unsigned long long)*edge_list_size) ;
#elif defined (CUDA_HST)
	unsigned long long* h_graph_edges;
	cudaMallocHost( (void**) &h_graph_edges, sizeof(unsigned long long)*edge_list_size) ;
#elif defined (CUDA_HYB)
    unsigned long long avail_size = 14 * 1024 * 1024 * 1024L - sizeof(bool)*no_of_nodes*3 - sizeof(int)*no_of_nodes - sizeof(Node)*no_of_nodes;
    unsigned long long edge_dev_size = avail_size / sizeof(unsigned long long);
    unsigned long long edge_um_size = 0;
    if (edge_list_size <= edge_dev_size) {
      edge_dev_size = edge_list_size;
      printf("Input is not large enough for hybrid allocation.\n");
    } else
      edge_um_size = edge_list_size - edge_dev_size;
	unsigned long long* h_graph_edges;
	h_graph_edges = (unsigned long long*) malloc(sizeof(unsigned long long)*edge_dev_size);
	unsigned long long* h_graph_edges_2 = NULL;
    if (edge_um_size)
#ifdef CUDA_HYB_HOST
	  cudaMallocHost( (void**) &h_graph_edges_2, sizeof(unsigned long long)*edge_um_size) ;
#else
	  cudaMallocManaged( (void**) &h_graph_edges_2, sizeof(unsigned long long)*edge_um_size) ;
#endif
#else
	unsigned long long* h_graph_edges = (unsigned long long*) malloc(sizeof(unsigned long long)*edge_list_size);
#endif
	for(unsigned long long i=0; i < edge_list_size ; i++)
	{
		//fscanf(fp,"%d",&id);
		//fscanf(fp,"%d",&cost);
		//h_graph_edges[i] = id;
#if defined (CUDA_HYB)
      if (i < edge_dev_size)
        h_graph_edges[i] = rand() % no_of_nodes;
      else
        h_graph_edges_2[i - edge_dev_size] = rand() % no_of_nodes;
#else
        h_graph_edges[i] = rand() % no_of_nodes;
#endif
	}

	//if(fp)
	//	fclose(fp);    

	//printf("Read File\n");
    unsigned long long total_size = sizeof(Node)*no_of_nodes + sizeof(unsigned long long)*edge_list_size;
    printf("Input size: %llu\n", total_size);
    total_size += sizeof(bool)*no_of_nodes*3 + sizeof(int)*no_of_nodes;
    printf("Total size: %llu\n", total_size);

	StopWatchInterface *timer = 0;
	  //	unsigned int timer = 0;

	// CUT_SAFE_CALL( cutCreateTimer( &timer));
	// CUT_SAFE_CALL( cutStartTimer( timer));
	sdkCreateTimer(&timer); 
	sdkStartTimer(&timer); 
#if defined (CUDA_HYB)
	//Copy the Edge List to device Memory
	unsigned long long* d_graph_edges;
	cudaMalloc( (void**) &d_graph_edges, sizeof(unsigned long long)*edge_dev_size) ;
	cudaMemcpy( d_graph_edges, h_graph_edges, sizeof(unsigned long long)*edge_dev_size, cudaMemcpyHostToDevice) ;

	//Copy the Mask to device memory
	bool* d_graph_mask;
	cudaMalloc( (void**) &d_graph_mask, sizeof(bool)*no_of_nodes) ;
	cudaMemcpy( d_graph_mask, h_graph_mask, sizeof(bool)*no_of_nodes, cudaMemcpyHostToDevice) ;

	bool* d_updating_graph_mask;
	cudaMalloc( (void**) &d_updating_graph_mask, sizeof(bool)*no_of_nodes) ;
	cudaMemcpy( d_updating_graph_mask, h_updating_graph_mask, sizeof(bool)*no_of_nodes, cudaMemcpyHostToDevice) ;

#if !defined (CUDA_HYB_L)
	//Copy the Node list to device memory
	Node* d_graph_nodes;
	cudaMalloc( (void**) &d_graph_nodes, sizeof(Node)*no_of_nodes) ;
	cudaMemcpy( d_graph_nodes, h_graph_nodes, sizeof(Node)*no_of_nodes, cudaMemcpyHostToDevice) ;

	//Copy the Visited nodes array to device memory
	bool* d_graph_visited;
	cudaMalloc( (void**) &d_graph_visited, sizeof(bool)*no_of_nodes) ;
	cudaMemcpy( d_graph_visited, h_graph_visited, sizeof(bool)*no_of_nodes, cudaMemcpyHostToDevice) ;
#endif
#elif defined (CUDA_HYS)
	//Copy the Mask to device memory
	bool* d_graph_mask;
	cudaMalloc( (void**) &d_graph_mask, sizeof(bool)*no_of_nodes) ;
	cudaMemcpy( d_graph_mask, h_graph_mask, sizeof(bool)*no_of_nodes, cudaMemcpyHostToDevice) ;

	bool* d_updating_graph_mask;
	cudaMalloc( (void**) &d_updating_graph_mask, sizeof(bool)*no_of_nodes) ;
	cudaMemcpy( d_updating_graph_mask, h_updating_graph_mask, sizeof(bool)*no_of_nodes, cudaMemcpyHostToDevice) ;
#elif !defined (CUDA_UVM) && !defined (CUDA_HST)
	//Copy the Node list to device memory
	Node* d_graph_nodes;
	cudaMalloc( (void**) &d_graph_nodes, sizeof(Node)*no_of_nodes) ;
	cudaMemcpy( d_graph_nodes, h_graph_nodes, sizeof(Node)*no_of_nodes, cudaMemcpyHostToDevice) ;

	//Copy the Edge List to device Memory
	unsigned long long* d_graph_edges;
	cudaMalloc( (void**) &d_graph_edges, sizeof(unsigned long long)*edge_list_size) ;
	cudaMemcpy( d_graph_edges, h_graph_edges, sizeof(unsigned long long)*edge_list_size, cudaMemcpyHostToDevice) ;

	//Copy the Mask to device memory
	bool* d_graph_mask;
	cudaMalloc( (void**) &d_graph_mask, sizeof(bool)*no_of_nodes) ;
	cudaMemcpy( d_graph_mask, h_graph_mask, sizeof(bool)*no_of_nodes, cudaMemcpyHostToDevice) ;

	bool* d_updating_graph_mask;
	cudaMalloc( (void**) &d_updating_graph_mask, sizeof(bool)*no_of_nodes) ;
	cudaMemcpy( d_updating_graph_mask, h_updating_graph_mask, sizeof(bool)*no_of_nodes, cudaMemcpyHostToDevice) ;

	//Copy the Visited nodes array to device memory
	bool* d_graph_visited;
	cudaMalloc( (void**) &d_graph_visited, sizeof(bool)*no_of_nodes) ;
	cudaMemcpy( d_graph_visited, h_graph_visited, sizeof(bool)*no_of_nodes, cudaMemcpyHostToDevice) ;
#endif

	// allocate mem for the result on host side
#if defined (CUDA_UVM) || defined (CUDA_HYS)
	int* h_cost;
	cudaMallocManaged( (void**) &h_cost, sizeof(int)*no_of_nodes);
#elif defined (CUDA_HST)
	int* h_cost;
	cudaMallocHost( (void**) &h_cost, sizeof(int)*no_of_nodes);
#elif defined (CUDA_HYB) && defined (CUDA_HYB_L)
	int* h_cost;
	cudaMallocManaged( (void**) &h_cost, sizeof(int)*no_of_nodes);
#else
	int* h_cost = (int*) malloc( sizeof(int)*no_of_nodes);
#endif
	for(unsigned long long i=0;i<no_of_nodes;i++)
		h_cost[i]=-1;
	h_cost[source]=0;
	
#if defined (CUDA_HYB) && defined (CUDA_HYB_L)
#elif !defined (CUDA_UVM) && !defined (CUDA_HST) && !defined (CUDA_HYS)
	// allocate device memory for result
	int* d_cost;
	cudaMalloc( (void**) &d_cost, sizeof(int)*no_of_nodes);
	cudaMemcpy( d_cost, h_cost, sizeof(int)*no_of_nodes, cudaMemcpyHostToDevice) ;
#endif

	//make a bool to check if the execution is over
	bool *d_over;
#if !defined (UVM_OVER)
	cudaMalloc( (void**) &d_over, sizeof(bool));
#elif defined (CUDA_UVM)
	cudaMallocManaged( (void**) &d_over, sizeof(bool));
#elif defined (CUDA_HST)
	cudaMallocHost( (void**) &d_over, sizeof(bool));
#else
	cudaMalloc( (void**) &d_over, sizeof(bool));
#endif

	printf("Copied Everything to GPU memory\n");

	// setup execution parameters
	dim3  grid( num_of_blocks, 1, 1);
	dim3  threads( num_of_threads_per_block, 1, 1);

	int k=0;
	printf("Start traversing the tree\n");
#if !defined (UVM_OVER)
	bool stop;
#endif
	//Call the Kernel untill all the elements of Frontier are not false
	do
	{
		//if no thread changes this value then the loop stops
#if !defined (UVM_OVER)
		stop=false;
		cudaMemcpy( d_over, &stop, sizeof(bool), cudaMemcpyHostToDevice) ;
#else
		*d_over=false;
#endif
#if defined (CUDA_UVM) || defined (CUDA_HST)
		Kernel<<< grid, threads, 0 >>>( h_graph_nodes, h_graph_edges, h_graph_mask, h_updating_graph_mask, h_graph_visited, h_cost, no_of_nodes);
#elif defined (CUDA_HYB) && defined (CUDA_HYB_L)
		Kernel<<< grid, threads, 0 >>>( h_graph_nodes, d_graph_edges, d_graph_mask, d_updating_graph_mask, h_graph_visited, h_cost, no_of_nodes, h_graph_edges_2, edge_dev_size);
#elif defined (CUDA_HYB)
		Kernel<<< grid, threads, 0 >>>( d_graph_nodes, d_graph_edges, d_graph_mask, d_updating_graph_mask, d_graph_visited, d_cost, no_of_nodes, h_graph_edges_2, edge_dev_size);
#elif defined (CUDA_HYS)
		Kernel<<< grid, threads, 0 >>>( h_graph_nodes, h_graph_edges, d_graph_mask, d_updating_graph_mask, h_graph_visited, h_cost, no_of_nodes);
#else
		Kernel<<< grid, threads, 0 >>>( d_graph_nodes, d_graph_edges, d_graph_mask, d_updating_graph_mask, d_graph_visited, d_cost, no_of_nodes);
#endif
		// check if kernel execution generated and error
		

#if defined (CUDA_UVM) || defined (CUDA_HST)
		Kernel2<<< grid, threads, 0 >>>( h_graph_mask, h_updating_graph_mask, h_graph_visited, d_over, no_of_nodes);
#elif defined (CUDA_HYB) && defined (CUDA_HYB_L)
		Kernel2<<< grid, threads, 0 >>>( d_graph_mask, d_updating_graph_mask, h_graph_visited, d_over, no_of_nodes);
#elif defined (CUDA_HYB)
		Kernel2<<< grid, threads, 0 >>>( d_graph_mask, d_updating_graph_mask, d_graph_visited, d_over, no_of_nodes);
#elif defined (CUDA_HYS)
		Kernel2<<< grid, threads, 0 >>>( d_graph_mask, d_updating_graph_mask, h_graph_visited, d_over, no_of_nodes);
#else
		Kernel2<<< grid, threads, 0 >>>( d_graph_mask, d_updating_graph_mask, d_graph_visited, d_over, no_of_nodes);
#endif
		// check if kernel execution generated and error
		

#if !defined (UVM_OVER)
		cudaMemcpy( &stop, d_over, sizeof(bool), cudaMemcpyDeviceToHost) ;
#else
        cudaDeviceSynchronize();
#endif
		k++;
	}
#if !defined (UVM_OVER)
	while(stop);
#else
	while(*d_over);
#endif


	printf("Kernel Executed %d times\n",k);

	sdkStopTimer(&timer); 
	printf("Time: %f\n", (sdkGetAverageTimerValue(&timer)/1000.0));

	// copy result from device to host
#if defined (CUDA_UVM) || defined (CUDA_HST) || defined (CUDA_HYS)
    cudaDeviceSynchronize();
#elif defined (CUDA_HYB) && defined (CUDA_HYB_L)
    cudaDeviceSynchronize();
#else
	cudaMemcpy( h_cost, d_cost, sizeof(int)*no_of_nodes, cudaMemcpyDeviceToHost) ;
#endif

	////Store the result into a file
	//FILE *fpo = fopen("result.txt","w");
	//for(int i=0;i<no_of_nodes;i++)
	//	fprintf(fpo,"%d) cost:%d\n",i,h_cost[i]);
	//fclose(fpo);
	//printf("Result stored in result.txt\n");


	// cleanup memory
#if defined (CUDA_UVM) || defined (CUDA_HST)
	cudaFree(h_graph_nodes);
	cudaFree(h_graph_edges);
	cudaFree(h_graph_mask);
	cudaFree(h_updating_graph_mask);
	cudaFree(h_graph_visited);
	cudaFree(h_cost);
#elif defined (CUDA_HYB)
	free( h_graph_mask);
	free( h_updating_graph_mask);
	cudaFree(d_graph_mask);
	cudaFree(d_updating_graph_mask);
	free( h_graph_edges);
	cudaFree(d_graph_edges);
#if defined (CUDA_HYB_L)
	cudaFree(h_graph_nodes);
	cudaFree(h_graph_visited);
	cudaFree(h_cost);
#else
	free( h_graph_nodes);
	free( h_graph_visited);
	free( h_cost);
	cudaFree(d_graph_nodes);
	cudaFree(d_graph_visited);
	cudaFree(d_cost);
#endif
    if (h_graph_edges_2)
	  cudaFree( h_graph_edges_2);
#elif defined (CUDA_HYS)
	cudaFree(h_graph_nodes);
	cudaFree(h_graph_edges);
	cudaFree(h_graph_visited);
	cudaFree(h_cost);
	free( h_graph_mask);
	free( h_updating_graph_mask);
	cudaFree(d_graph_mask);
	cudaFree(d_updating_graph_mask);
#else
	free( h_graph_nodes);
	free( h_graph_edges);
	free( h_graph_mask);
	free( h_updating_graph_mask);
	free( h_graph_visited);
	free( h_cost);
	cudaFree(d_graph_nodes);
	cudaFree(d_graph_edges);
	cudaFree(d_graph_mask);
	cudaFree(d_updating_graph_mask);
	cudaFree(d_graph_visited);
	cudaFree(d_cost);
#endif
}
