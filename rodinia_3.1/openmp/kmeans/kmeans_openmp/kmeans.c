/*****************************************************************************/
/*IMPORTANT:  READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.         */
/*By downloading, copying, installing or using the software you agree        */
/*to this license.  If you do not agree to this license, do not download,    */
/*install, copy or use the software.                                         */
/*                                                                           */
/*                                                                           */
/*Copyright (c) 2005 Northwestern University                                 */
/*All rights reserved.                                                       */

/*Redistribution of the software in source and binary forms,                 */
/*with or without modification, is permitted provided that the               */
/*following conditions are met:                                              */
/*                                                                           */
/*1       Redistributions of source code must retain the above copyright     */
/*        notice, this list of conditions and the following disclaimer.      */
/*                                                                           */
/*2       Redistributions in binary form must reproduce the above copyright   */
/*        notice, this list of conditions and the following disclaimer in the */
/*        documentation and/or other materials provided with the distribution.*/ 
/*                                                                            */
/*3       Neither the name of Northwestern University nor the names of its    */
/*        contributors may be used to endorse or promote products derived     */
/*        from this software without specific prior written permission.       */
/*                                                                            */
/*THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS    */
/*IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED      */
/*TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY, NON-INFRINGEMENT AND         */
/*FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL          */
/*NORTHWESTERN UNIVERSITY OR ITS CONTRIBUTORS BE LIABLE FOR ANY DIRECT,       */
/*INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES          */
/*(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR          */
/*SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)          */
/*HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,         */
/*STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN    */
/*ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE             */
/*POSSIBILITY OF SUCH DAMAGE.                                                 */
/******************************************************************************/

/*************************************************************************/
/**   File:         example.c                                           **/
/**   Description:  Takes as input a file:                              **/
/**                 ascii  file: containing 1 data point per line       **/
/**                 binary file: first int is the number of objects     **/
/**                              2nd int is the no. of features of each **/
/**                              object                                 **/
/**                 This example performs a fuzzy c-means clustering    **/
/**                 on the data. Fuzzy clustering is performed using    **/
/**                 min to max clusters and the clustering that gets    **/
/**                 the best score according to a compactness and       **/
/**                 separation criterion are returned.                  **/
/**   Author:  Wei-keng Liao                                            **/
/**            ECE Department Northwestern University                   **/
/**            email: wkliao@ece.northwestern.edu                       **/
/**                                                                     **/
/**   Edited by: Jay Pisharath                                          **/
/**              Northwestern University.                               **/
/**                                                                     **/
/**   ================================================================  **/
/**																		**/
/**   Edited by: Sang-Ha  Lee											**/
/**				 University of Virginia									**/
/**																		**/
/**   Description:	No longer supports fuzzy c-means clustering;	 	**/
/**					only regular k-means clustering.					**/
/**					Simplified for main functionality: regular k-means	**/
/**					clustering.											**/
/**                                                                     **/
/*************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <sys/types.h>
#include <fcntl.h>
#include <omp.h>
#include <unistd.h>
#include "getopt.h"

#include "kmeans.h"

extern double wtime(void);

int num_omp_threads = 1;
double total_size = 0.0;

/*---< usage() >------------------------------------------------------------*/
void usage(char *argv0) {
    char *help =
        "Usage: %s [switches] -i filename\n"
        "       -i filename     		: file containing data to be clustered\n"
        "       -b                 	: input file is in binary format\n"
		"       -k                 	: number of clusters (default is 5) \n"
        "       -t threshold		: threshold value\n"
		"       -n no. of threads	: number of threads";
    fprintf(stderr, help, argv0);
    exit(-1);
}

/*---< main() >-------------------------------------------------------------*/
int main(int argc, char **argv) {
  int opt;
  extern char *optarg;
  extern int optind;
  int nclusters = 5;
  char *filename = 0;
  float **attributes;
  float **cluster_centres = NULL;
  unsigned long i, j;

  int numAttributes;
  unsigned long numObjects;
  char line[1024];
  int isBinaryFile = 0;
  int nloops = 1;
  float threshold = 0.001;
  unsigned long long num = 0;

  while ((opt = getopt(argc, argv, "i:k:t:b:n:l:?")) != EOF) {
    switch (opt) {
    case 'i':
      filename = optarg;
      break;
    case 'b':
      isBinaryFile = 1;
      break;
    case 't':
      threshold = atof(optarg);
      break;
    case 'k':
      nclusters = atoi(optarg);
      break;
    case 'n':
      num_omp_threads = atoi(optarg);
      break;
    case 'l':
      num = strtoul(optarg, NULL, 10);
      break;
    case '?':
      usage(argv[0]);
      break;
    default:
      usage(argv[0]);
      break;
    }
  }

    numAttributes = 34;
    if (num == 0)
      return 0;
    numObjects = num;
#ifdef OMP_GPU_OFFLOAD_UM
    attributes = (float **)omp_target_alloc(numObjects * sizeof(float *), -100);
    attributes[0] = (float *)omp_target_alloc(
        numObjects * numAttributes * sizeof(float), -100);
    for (i = 1; i < numObjects; i++) {
      attributes[i] = attributes[i - 1] + numAttributes;
    }
    for (i = 0; i < numObjects; i++)
      for (j = 0; j < numAttributes; j++)
        attributes[i][j] = ((float)rand() / (float)RAND_MAX);
    total_size += numObjects * sizeof(float *) +
                  numObjects * numAttributes * sizeof(float);
#else
    total_size += numObjects * sizeof(float *);
    attributes = (float **)malloc(numObjects * sizeof(float *));
    for (i = 0; i < numObjects; i++) {
      attributes[i] = (float *)malloc(numAttributes * sizeof(float));
      total_size += numAttributes * sizeof(float);
    }
#endif

    for (i=0; i<nloops; i++) {

        cluster_centres = NULL;
        cluster(numObjects,
                numAttributes,
                attributes,           /* [numObjects][numAttributes] */                
                nclusters,
                threshold,
                &cluster_centres
               );

    }
    printf("Size: %f\n", total_size / 1024 / 1024 / 1024);

#ifdef OMP_GPU_OFFLOAD_UM
    //for (i=0; i<numObjects; i++)
    //    omp_target_free(attributes[i], omp_get_default_device());
    //omp_target_free(attributes, omp_get_default_device());
    //for(i=0; i<nclusters; i++) 
    //    omp_target_free(cluster_centres[i], omp_get_default_device());
    //omp_target_free(cluster_centres, omp_get_default_device());
#else
    for (i=0; i<numObjects; i++)
        free(attributes[i]);
    free(attributes);
    for(i=0; i<nclusters; i++) 
        free(cluster_centres[i]);
    free(cluster_centres);
#endif
    return(0);
}

