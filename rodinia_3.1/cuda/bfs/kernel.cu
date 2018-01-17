/*********************************************************************************
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

The CUDA Kernel for Applying BFS on a loaded Graph. Created By Pawan Harish
**********************************************************************************/
#ifndef _KERNEL_H_
#define _KERNEL_H_

__global__ void
#if defined (CUDA_HYB)
Kernel( Node* g_graph_nodes, unsigned long long* g_graph_edges, bool* g_graph_mask, bool* g_updating_graph_mask, bool *g_graph_visited, int* g_cost, unsigned long long no_of_nodes, unsigned long long* g_graph_edges_2, unsigned long long edge_dev_size) 
#else
Kernel( Node* g_graph_nodes, unsigned long long* g_graph_edges, bool* g_graph_mask, bool* g_updating_graph_mask, bool *g_graph_visited, int* g_cost, unsigned long long no_of_nodes) 
#endif
{
	unsigned long long tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if( tid<no_of_nodes && g_graph_mask[tid])
	{
		g_graph_mask[tid]=false;
		for(unsigned long long i=g_graph_nodes[tid].starting; i<(g_graph_nodes[tid].no_of_edges + g_graph_nodes[tid].starting); i++)
			{
#if defined (CUDA_HYB)
			unsigned long long id;
              if (i < edge_dev_size)
                id = g_graph_edges[i];
              else
                id = g_graph_edges_2[i - edge_dev_size];
#else
			unsigned long long id = g_graph_edges[i];
#endif
			if(!g_graph_visited[id])
				{
				g_cost[id]=g_cost[tid]+1;
				g_updating_graph_mask[id]=true;
				}
			}
	}
}

#endif 
