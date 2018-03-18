#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

#define OPEN

#ifdef OMP_GPU_OFFLOAD_UM
//#define CUDA_UM
//#define MAP_ALL
#include <cuda_runtime_api.h>
#endif

#define min(a,b) ((a) < (b) ? (a) : (b))

FILE *fp;

//Structure to hold a node information
struct Node {
    unsigned long long starting;
    unsigned long long no_of_edges;
};

void BFSGraph(int argc, char** argv);

void Usage(int argc, char**argv) {

fprintf(stderr,"Usage: %s <num_threads>\n", argv[0]);

}
////////////////////////////////////////////////////////////////////////////////
// Main Program
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{
    BFSGraph( argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//Apply BFS on a Graph using CUDA
////////////////////////////////////////////////////////////////////////////////
void BFSGraph( int argc, char** argv) 
{
    unsigned long long no_of_nodes = 0;
    unsigned long long edge_list_size = 0;
    int     num_omp_threads;
    
    if(argc!=2){
        Usage(argc, argv);
        exit(0);
    }
    srand(0);
    
    num_omp_threads = atoi(argv[1]);

    // allocate host memory
    //double start_time = omp_get_wtime();
    no_of_nodes = num_omp_threads;
#ifdef OMP_GPU_OFFLOAD_UM
#ifdef CUDA_UM
    Node* h_graph_nodes;
    cudaMallocManaged((void**)&h_graph_nodes, sizeof(Node)*no_of_nodes, cudaMemAttachGlobal);
    bool *h_graph_mask;
    cudaMallocManaged((void**)&h_graph_mask, sizeof(bool)*no_of_nodes, cudaMemAttachGlobal);
    bool *h_updating_graph_mask;
    cudaMallocManaged((void**)&h_updating_graph_mask, sizeof(bool)*no_of_nodes, cudaMemAttachGlobal);
    bool *h_graph_visited;
    cudaMallocManaged((void**)&h_graph_visited, sizeof(bool)*no_of_nodes, cudaMemAttachGlobal);
#else
    //Node* h_graph_nodes = (Node*) omp_target_alloc(sizeof(Node)*no_of_nodes, omp_get_default_device());
    //bool *h_graph_mask = (bool*) omp_target_alloc(sizeof(bool)*no_of_nodes, omp_get_default_device());
    //bool *h_updating_graph_mask = (bool*) omp_target_alloc(sizeof(bool)*no_of_nodes, omp_get_default_device());
    //bool *h_graph_visited = (bool*) omp_target_alloc(sizeof(bool)*no_of_nodes, omp_get_default_device());
    Node* h_graph_nodes = (Node*) omp_target_alloc(sizeof(Node)*no_of_nodes, -100);
    bool *h_graph_mask = (bool*) omp_target_alloc(sizeof(bool)*no_of_nodes, -100);
    bool *h_updating_graph_mask = (bool*) omp_target_alloc(sizeof(bool)*no_of_nodes, -100);
    bool *h_graph_visited = (bool*) omp_target_alloc(sizeof(bool)*no_of_nodes, -100);
#endif
#else
    Node* h_graph_nodes = (Node*) malloc(sizeof(Node)*no_of_nodes);
    bool *h_graph_mask = (bool*) malloc(sizeof(bool)*no_of_nodes);
    bool *h_updating_graph_mask = (bool*) malloc(sizeof(bool)*no_of_nodes);
    bool *h_graph_visited = (bool*) malloc(sizeof(bool)*no_of_nodes);
#endif
    unsigned long long total_size = sizeof(Node)*no_of_nodes + sizeof(bool)*no_of_nodes + sizeof(bool)*no_of_nodes + sizeof(bool)*no_of_nodes;
    unsigned long long start=0, edgeno;

    // initalize the memory
    for( unsigned int i = 0; i < no_of_nodes; i++)
    {
        edgeno = rand()%10+1;
        h_graph_nodes[i].starting = start;
        h_graph_nodes[i].no_of_edges = edgeno;
        h_graph_mask[i]=false;
        h_updating_graph_mask[i]=false;
        h_graph_visited[i]=false;
        start += edgeno;
    }
    edge_list_size = start;
    int source = 0;
    h_graph_mask[source]=true;
    h_graph_visited[source]=true;
#ifdef OMP_GPU_OFFLOAD_UM
#ifdef CUDA_UM
    unsigned long long* h_graph_edges;
    cudaMallocManaged((void**)&h_graph_edges, sizeof(unsigned long long)*edge_list_size, cudaMemAttachGlobal);
#else
    //unsigned long long* h_graph_edges = (unsigned long long*) omp_target_alloc(sizeof(unsigned long long)*edge_list_size, omp_get_default_device());
    unsigned long long* h_graph_edges = (unsigned long long*) omp_target_alloc(sizeof(unsigned long long)*edge_list_size, -100);
#endif
#else
    unsigned long long* h_graph_edges = (unsigned long long*) malloc(sizeof(unsigned long long)*edge_list_size);
#endif
    for(unsigned long long i=0; i < edge_list_size ; i++)
    {
        h_graph_edges[i] = rand()%no_of_nodes;
    }

    // allocate mem for the result on host side
    struct timeval st, et;
#ifdef OMP_GPU_OFFLOAD_UM
#ifdef CUDA_UM
    int* h_cost;
    cudaMallocManaged((void**)&h_cost, sizeof(int)*no_of_nodes, cudaMemAttachGlobal);
#else
    //int* h_cost = (int*) omp_target_alloc( sizeof(int)*no_of_nodes, omp_get_default_device());
    int* h_cost = (int*) omp_target_alloc( sizeof(int)*no_of_nodes, -100);
#endif
#else
    int* h_cost = (int*) malloc( sizeof(int)*no_of_nodes);
#endif
    total_size += sizeof(unsigned long long)*edge_list_size + sizeof(int)*no_of_nodes;
    printf("Total Size: %f GB\n", (float)((total_size/1024.0)/1024.0/1024.0));
    #ifdef OPEN
    // allocate mem for the result on host side
    for(int i=0;i<no_of_nodes;i++)
        h_cost[i]=-1;
    h_cost[source]=0;
    int counter = 0;
    
    int numDevice;
    cudaGetDeviceCount(&numDevice);
    printf("Device #: %d\n", numDevice);
    if (numDevice <= 0)
      return;
    //numDevice = 1;
    unsigned long long numNodesPerDev = (no_of_nodes + numDevice - 1) / numDevice;

    double start_time = omp_get_wtime();
    #if defined(OMP_GPU_OFFLOAD)
        #pragma omp target data \
            map(to: no_of_nodes, \
                h_graph_mask[0:no_of_nodes], \
                h_graph_nodes[0:no_of_nodes], \
                h_graph_edges[0:edge_list_size], \
                h_graph_visited[0:no_of_nodes], \
                h_updating_graph_mask[0:no_of_nodes]) \
            map(tofrom: h_cost[0:no_of_nodes])
    #elif defined(OMP_GPU_OFFLOAD_UM) && defined(MAP_ALL)
        #pragma omp target data \
            map(to: h_graph_mask[0:no_of_nodes], \
                h_graph_nodes[0:no_of_nodes], \
                h_graph_edges[0:edge_list_size], \
                h_graph_visited[0:no_of_nodes], \
                h_updating_graph_mask[0:no_of_nodes]) \
            map(tofrom: h_cost[0:no_of_nodes])
    #elif defined(OMP_GPU_OFFLOAD_UM)
    #endif
    {
#if defined(OMP_GPU_OFFLOAD)
    double trans_time =  omp_get_wtime();
    printf("Transfer time: %g\n", trans_time - start_time);
#endif
    //bool stop;
    bool *stop;
	//cudaMallocManaged( (void**) &stop, sizeof(bool));
    stop = (bool*) omp_target_alloc(sizeof(bool), -100);
    do 
    {
        //if no thread changes this value then the loop stops
        *stop=false;

        for (int d = 0; d < numDevice; d++) {
          unsigned long long start = d * numNodesPerDev;
          unsigned long long end = min((d+1) * numNodesPerDev, no_of_nodes);
          #pragma omp target teams distribute parallel for device(d) \
              is_device_ptr(h_graph_mask) \
              is_device_ptr(h_graph_nodes) \
              is_device_ptr(h_graph_edges) \
              is_device_ptr(h_graph_visited) \
              is_device_ptr(h_updating_graph_mask) \
              is_device_ptr(h_cost)
          for(unsigned long long tid = start; tid < end; tid++ )
          {
              if (h_graph_mask[tid] == true) 
              {
                  h_graph_mask[tid]=false;
                  for(unsigned long long i=h_graph_nodes[tid].starting; 
                          i<(h_graph_nodes[tid].no_of_edges + 
                              h_graph_nodes[tid].starting); i++)
                  {
                      unsigned long long id = h_graph_edges[i];
                      if(!h_graph_visited[id])
                      {
                          h_cost[id]=h_cost[tid]+1;
                          h_updating_graph_mask[id]=true;
                      }
                  }
              }
          }
        }
        for (int d = 0; d < numDevice; d++) {
          cudaSetDevice(d);
          cudaDeviceSynchronize();
        }

        for (int d = 0; d < numDevice; d++) {
          unsigned long long start = d * numNodesPerDev;
          unsigned long long end = min((d+1) * numNodesPerDev, no_of_nodes);
          #pragma omp target teams distribute parallel for device(d) \
              is_device_ptr(stop) \
              is_device_ptr(h_graph_mask) \
              is_device_ptr(h_graph_visited) \
              is_device_ptr(h_updating_graph_mask)
            for(unsigned long long tid=start; tid< end; tid++ )
            {
                if (h_updating_graph_mask[tid] == true){
                    h_graph_mask[tid]=true;
                    h_graph_visited[tid]=true;
                    *stop=true;
                    h_updating_graph_mask[tid]=false;
                }
            }
        }
        // FIXME: this does not produce the right results somehow
        for (int d = 0; d < numDevice; d++) {
          cudaSetDevice(d);
          cudaDeviceSynchronize();
        }
    } while(*stop);
    double end_time = omp_get_wtime();
    printf("Total time: %g\n", (end_time - start_time));
    }
    #endif

/*    FILE *fpo = fopen("result.txt","w");
    for(int i=0;i<no_of_nodes;i++)
        fprintf(fpo,"%d) cost:%d\n",i,h_cost[i]);
    fclose(fpo);
    printf("Result stored in result.txt\n");
*/
    // cleanup memory
    #ifdef OMP_GPU_OFFLOAD_UM
#ifdef CUDA_UM
    cudaFree(h_graph_nodes);
    cudaFree(h_graph_edges);
    cudaFree(h_graph_mask);
    cudaFree(h_updating_graph_mask);
    cudaFree(h_graph_visited);
    cudaFree(h_cost);
#else
    omp_target_free(h_graph_nodes, omp_get_default_device());
    omp_target_free(h_graph_edges, omp_get_default_device());
    omp_target_free(h_graph_mask, omp_get_default_device());
    omp_target_free(h_updating_graph_mask, omp_get_default_device());
    omp_target_free(h_graph_visited, omp_get_default_device());
    omp_target_free(h_cost, omp_get_default_device());
#endif
    #else 
    free(h_graph_nodes);
    free(h_graph_edges);
    free(h_graph_mask);
    free(h_updating_graph_mask);
    free(h_graph_visited);
    free(h_cost);
    #endif
}
