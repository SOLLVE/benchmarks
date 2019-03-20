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
/**   File:         kmeans_clustering.c                                 **/
/**   Description:  Implementation of regular k-means clustering        **/
/**                 algorithm                                           **/
/**   Author:  Wei-keng Liao                                            **/
/**            ECE Department, Northwestern University                  **/
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
#include <float.h>
#include <math.h>
#include "kmeans.h"
#include <omp.h>

#define RANDOM_MAX 2147483647

#ifndef FLT_MAX
#define FLT_MAX 3.40282347e+38
#endif

extern double wtime(void);
extern int num_omp_threads;
extern unsigned long total_size;

#ifdef OMP_GPU_OFFLOAD_UM
#pragma omp declare target
#endif
int find_nearest_point(float  *pt,          /* [nfeatures] */
                       int     nfeatures,
                       float **pts,         /* [npts][nfeatures] */
                       int     npts)
{
    int index, i, j;
    float min_dist=FLT_MAX;

    /* find the cluster center id with min distance to pt */
    for (i=0; i<npts; i++) {
        float dist;
        //dist = euclid_dist_2(pt, pts[i], nfeatures);  /* no need square root */
        dist = 0;
        for (j=0; j<nfeatures; j++)
            dist += (pt[j]-pts[i][j]) * (pt[j]-pts[i][j]);
        if (dist < min_dist) {
            min_dist = dist;
            index    = i;
        }
    }
    return(index);
}

/*----< euclid_dist_2() >----------------------------------------------------*/
/* multi-dimensional spatial Euclid distance square */
__inline
float euclid_dist_2(float *pt1,
                    float *pt2,
                    int    numdims)
{
    int i;
    float ans=0.0;

    for (i=0; i<numdims; i++)
        ans += (pt1[i]-pt2[i]) * (pt1[i]-pt2[i]);

    return(ans);
}
#if defined(OMP_GPU_OFFLOAD_UM) || defined(OMP_GPU_OFFLOAD_UM_UM)
#pragma omp end declare target
#endif


/*----< kmeans_clustering() >---------------------------------------------*/
float** kmeans_clustering(float **feature,    /* in: [npoints][nfeatures] */
                          int     nfeatures,
                          unsigned long long     npoints,
                          int     nclusters,
                          float   threshold,
                          int    *membership) /* out: [npoints] */
{

    unsigned long i, j, k, n=0, index, loop=0;
    int     *new_centers_len;			/* [nclusters]: no. of points in each cluster */
	float   **new_centers;				/* [nclusters][nfeatures] */
	float   **clusters;					/* out: [nclusters][nfeatures] */
    float   delta;

    double  timing;

	int      nthreads;
    int    **partial_new_centers_len;
    float ***partial_new_centers;

    nthreads = num_omp_threads; 

    /* allocate space for returning variable clusters[] */
#if defined(OMP_GPU_OFFLOAD_UM)
    float *fstore =
        (float *)omp_target_alloc((unsigned long long)feature[0], -200);
    membership = (int *)omp_target_alloc((unsigned long long)membership, -200);
    clusters = (float **)omp_target_alloc(nclusters * sizeof(float *), -100);
    total_size += nclusters * sizeof(float *);
    for (i = 0; i < nclusters; i++) {
      clusters[i] = (float *)omp_target_alloc(nfeatures * sizeof(float), -100);
      total_size += nfeatures * sizeof(float);
  }
#else
    clusters = (float **)malloc(nclusters * sizeof(float *));
    total_size += nclusters * sizeof(float *);
    for (i = 0; i < nclusters; i++) {
      clusters[i] = (float *)malloc(nfeatures * sizeof(float));
      total_size += nfeatures * sizeof(float);
    }
#endif

  /* randomly pick cluster centers */
  for (i = 0; i < nclusters; i++) {
    // n = (int)rand() % npoints;
    for (j = 0; j < nfeatures; j++)
      clusters[i][j] = feature[n][j];
    n++;
  }

  for (i = 0; i < npoints; i++)
    membership[i] = -1;

  /* need to initialize new_centers_len and new_centers[0] to all 0 */
  total_size += nclusters * sizeof(float);
#if defined(OMP_GPU_OFFLOAD_UM)
  new_centers_len = (int *)omp_target_alloc(nclusters * sizeof(int), -100);
  new_centers = (float **)omp_target_alloc(nclusters * sizeof(float *), -100);
  for (i = 0; i < nclusters; i++) {
    new_centers[i] = (float *)omp_target_alloc(nfeatures * sizeof(float), -100);
    total_size += nfeatures * sizeof(float);
  }
    #else
    new_centers_len = (int*) calloc(nclusters, sizeof(int));
    new_centers    = (float**) malloc(nclusters *  sizeof(float*));
    for (i=0; i<nclusters; i++) {
        new_centers[i] = (float*)  calloc(nfeatures, sizeof(float));
        total_size += nfeatures * sizeof(float);
    }
    #endif

    total_size += nthreads * sizeof(int*);
    total_size += nthreads * sizeof(float**);
    #if defined(OMP_GPU_OFFLOAD_UM)
    partial_new_centers_len = (int**) omp_target_alloc(nthreads * sizeof(int*), -100);
	partial_new_centers = (float***) omp_target_alloc(nthreads * sizeof(float**), -100);
    for (i=0; i<nthreads; i++) {
        partial_new_centers_len[i] = (int*)  omp_target_alloc(nclusters * sizeof(int), -100);
        partial_new_centers[i] =(float**) omp_target_alloc(nclusters * sizeof(float*), -100);
        total_size += nclusters * sizeof(int);
        total_size += nclusters * sizeof(float*);
    }
    #else
    partial_new_centers_len    = (int**) malloc(nthreads * sizeof(int*));
	partial_new_centers    =(float***)malloc(nthreads * sizeof(float**));
    for (i=0; i<nthreads; i++) {
        partial_new_centers_len[i] = (int*)  calloc(nclusters, sizeof(int));
        partial_new_centers[i] =(float**) malloc(nclusters * sizeof(float*));
        total_size += nclusters * sizeof(int);
        total_size += nclusters * sizeof(float*);
    }
    #endif

	for (i=0; i<nthreads; i++)
	{
        for (j=0; j<nclusters; j++) {
            #if defined(OMP_GPU_OFFLOAD_UM)
            partial_new_centers[i][j] = (float*) omp_target_alloc(nfeatures * sizeof(float), -100);
            #else
            partial_new_centers[i][j] = (float*)calloc(nfeatures, sizeof(float));
            #endif
            total_size += nfeatures * sizeof(float);
        }
	}
	double start_time = omp_get_wtime();
    do {
        delta = 0.0;
        #ifndef OMP_GPU_OFFLOAD_UM
		omp_set_num_threads(num_omp_threads);
     		#pragma omp parallel \
                shared(feature,clusters,membership,partial_new_centers,partial_new_centers_len)
        #endif
        {
            #if defined(OMP_GPU_OFFLOAD_UM)
            int tid = 0;
            #else
            int tid = omp_get_thread_num();
            #endif
            #if defined(OMP_GPU_OFFLOAD_UM)
		    //#pragma omp target teams distribute parallel for \
                    //    private(i, j, index) is_device_ptr(feature) \
                    //    is_device_ptr(membership) \
                    //    is_device_ptr(partial_new_centers_len) is_device_ptr(partial_new_centers) \
                    //    is_device_ptr(clusters) \
                    //    reduction(+:delta) \
                    //    firstprivate(npoints, nclusters, nfeatures)
		    #pragma omp target teams distribute parallel for \
                        private(i, j, index) map(to:fstore[0:npoints*nfeatures]) \
                        map(tofrom: membership[0:npoints]) \
                        is_device_ptr(partial_new_centers_len) is_device_ptr(partial_new_centers) \
                        is_device_ptr(clusters) \
                        reduction(+:delta) \
                        firstprivate(npoints, nclusters, nfeatures)
            #else
            #pragma omp for \
                        private(i, j, index) \
                        firstprivate(npoints, nfeatures) \
                        schedule(static) \
                        reduction(+:delta)
            #endif
            for (i=0; i<npoints; i++) {
	            /* find the index of nestest cluster centers */					
            #if defined(OMP_GPU_OFFLOAD_UM)
    	        index = find_nearest_point(fstore+i*nfeatures,
            #else
    	        index = find_nearest_point(feature[i],
            #endif
    		             nfeatures,
    		             clusters,
    		             nclusters);				
    	        /* if membership changes, increase delta by 1 */
    	        if (membership[i] != index) delta += 1.0;

    	        /* assign the membership to object i */
    	        membership[i] = index;

    	        /* update new cluster centers : sum of all objects located
    		       within */
    	        partial_new_centers_len[tid][index]++;				
    	        for (j=0; j<nfeatures; j++)
            #if defined(OMP_GPU_OFFLOAD_UM)
    		       partial_new_centers[tid][index][j] += fstore[i*nfeatures+j];
            #else
    		       partial_new_centers[tid][index][j] += feature[i][j];
            #endif
            }
        } /* end of #pragma omp parallel */

        /* let the main thread perform the array reduction */
        for (i=0; i<nclusters; i++) {
            for (j=0; j<nthreads; j++) {
                new_centers_len[i] += partial_new_centers_len[j][i];
                partial_new_centers_len[j][i] = 0.0;
                for (k=0; k<nfeatures; k++) {
                    new_centers[i][k] += partial_new_centers[j][i][k];
                    partial_new_centers[j][i][k] = 0.0;
                }
            }
        }    

		/* replace old cluster centers with new_centers */
		for (i=0; i<nclusters; i++) {
            for (j=0; j<nfeatures; j++) {
                if (new_centers_len[i] > 0)
					clusters[i][j] = new_centers[i][j] / new_centers_len[i];
				new_centers[i][j] = 0.0;   /* set back to 0 */
			}
			new_centers_len[i] = 0;   /* set back to 0 */
		}
        
    } while (delta > threshold && loop++ < 10);

    double end_time = omp_get_wtime();
    printf("Time: %f\n", end_time - start_time);
    #ifdef OMP_GPU_OFFLOAD_UM 
    //omp_target_free(new_centers[0], 0);
    //omp_target_free(new_centers, 0);
    //omp_target_free(new_centers_len, 0);
    #else
    free(new_centers[0]);
    free(new_centers);
    free(new_centers_len);
    #endif
    
    return clusters;
}

