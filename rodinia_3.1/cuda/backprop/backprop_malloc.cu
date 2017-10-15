/*
 ******************************************************************
 * HISTORY
 * 15-Oct-94  Jeff Shufelt (js), Carnegie Mellon University
 *	Prepared for 15-681, Fall 1994.
 * Modified by Shuai Che
 ******************************************************************
 */

#include <stdio.h>
#include <stdlib.h>
#include "backprop.h"

//#define CUDA_UVM

/*** Allocate 1d array of floats ***/

extern "C"
float *alloc_1d_dbl(unsigned long long n)
{
  float *newmem;

#ifndef CUDA_UVM
  newmem = (float *) malloc ((n * sizeof (float)));
#else
  cudaMallocManaged((void**)&newmem, (n * sizeof (float)));
#endif
  if (newmem == NULL) {
    printf("ALLOC_1D_DBL: Couldn't allocate array of floats\n");
    return (NULL);
  }
  return newmem;
}


/*** Allocate 2d array of floats ***/

extern "C"
float **alloc_2d_dbl(unsigned long long m, unsigned long long n)
{
  int i;
  float **newmem;
  float *newmem_content;

#ifndef CUDA_UVM
  newmem = (float **) malloc ((m * sizeof (float *)));
  newmem_content = (float *) malloc ((m * n * sizeof (float)));
  if (newmem == NULL || newmem_content == NULL) {
    printf("ALLOC_2D_DBL: Couldn't allocate array of dbl ptrs\n");
    return (NULL);
  }

  for (i = 0; i < m; i++) {
    //newmem[i] = alloc_1d_dbl(n);
    newmem[i] = &newmem_content[i*n];
  }
#else
  cudaMallocManaged((void**)&newmem, (m * sizeof (float *)));
  cudaMallocManaged((void**)&newmem_content, (m * n * sizeof (float)));
  if (newmem == NULL || newmem_content == NULL) {
    printf("ALLOC_2D_DBL: Couldn't allocate array of dbl ptrs\n");
    return (NULL);
  }

  for (i = 0; i < m; i++) {
    newmem[i] = &newmem_content[i*n];
  }
#endif

  return (newmem);
}


extern "C"
BPNN *bpnn_internal_create(int n_in, int n_hidden, int n_out)
{
  BPNN *newnet;

#ifndef CUDA_UVM
  newnet = (BPNN *) malloc (sizeof (BPNN));
#else
  cudaMallocManaged((void**)&newnet, sizeof (BPNN));
#endif
  if (newnet == NULL) {
    printf("BPNN_CREATE: Couldn't allocate neural network\n");
    return (NULL);
  }

  newnet->input_n = n_in;
  newnet->hidden_n = n_hidden;
  newnet->output_n = n_out;
  newnet->input_units = alloc_1d_dbl(n_in + 1);
  newnet->hidden_units = alloc_1d_dbl(n_hidden + 1);
  newnet->output_units = alloc_1d_dbl(n_out + 1);

  newnet->hidden_delta = alloc_1d_dbl(n_hidden + 1);
  newnet->output_delta = alloc_1d_dbl(n_out + 1);
  newnet->target = alloc_1d_dbl(n_out + 1);

  newnet->input_weights = alloc_2d_dbl(n_in + 1, n_hidden + 1);
  newnet->hidden_weights = alloc_2d_dbl(n_hidden + 1, n_out + 1);

  newnet->input_prev_weights = alloc_2d_dbl(n_in + 1, n_hidden + 1);
  newnet->hidden_prev_weights = alloc_2d_dbl(n_hidden + 1, n_out + 1);

  return (newnet);
}


extern "C"
void bpnn_free(BPNN *net)
{
  int n1, n2, i;

  n1 = net->input_n;
  n2 = net->hidden_n;

#ifndef CUDA_UVM
  free((char *) net->input_units);
  free((char *) net->hidden_units);
  free((char *) net->output_units);

  free((char *) net->hidden_delta);
  free((char *) net->output_delta);
  free((char *) net->target);

  for (i = 0; i <= n1; i++) {
    free((char *) net->input_weights[i]);
    free((char *) net->input_prev_weights[i]);
  }
  free((char *) net->input_weights);
  free((char *) net->input_prev_weights);

  for (i = 0; i <= n2; i++) {
    free((char *) net->hidden_weights[i]);
    free((char *) net->hidden_prev_weights[i]);
  }
  free((char *) net->hidden_weights);
  free((char *) net->hidden_prev_weights);

  free((char *) net);
#else
  cudaFree((char *) net->input_units);
  cudaFree((char *) net->hidden_units);
  cudaFree((char *) net->output_units);

  cudaFree((char *) net->hidden_delta);
  cudaFree((char *) net->output_delta);
  cudaFree((char *) net->target);

  for (i = 0; i <= n1; i++) {
    cudaFree((char *) net->input_weights[i]);
    cudaFree((char *) net->input_prev_weights[i]);
  }
  cudaFree((char *) net->input_weights);
  cudaFree((char *) net->input_prev_weights);

  for (i = 0; i <= n2; i++) {
    cudaFree((char *) net->hidden_weights[i]);
    cudaFree((char *) net->hidden_prev_weights[i]);
  }
  cudaFree((char *) net->hidden_weights);
  cudaFree((char *) net->hidden_prev_weights);

  cudaFree((char *) net);
#endif
}

