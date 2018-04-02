/*
 ******************************************************************
 * HISTORY
 * 15-Oct-94  Jeff Shufelt (js), Carnegie Mellon University
 *	Prepared for 15-681, Fall 1994.
 * Modified by Shuai Che
 ******************************************************************
 */

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h> // for open
#include <unistd.h> // for close
#include "backprop.h"
#ifdef OMP_GPU_OFFLOAD
#pragma omp  declare target
#endif
#include <math.h>
#ifdef OMP_GPU_OFFLOAD
#pragma omp end declare target
#endif
#define OPEN

#define ABS(x)  (((x) > 0.0) ? (x) : (-(x)))

#define fastcopy(to,from,len)\
{\
    register char *_to,*_from;\
    register int _i,_l;\
    _to = (char *)(to);\
    _from = (char *)(from);\
    _l = (len);\
    for (_i = 0; _i < _l; _i++) *_to++ = *_from++;\
}

extern unsigned long total_size;
/*** Return random number between 0.0 and 1.0 ***/
float drnd()
{
    return ((float) rand() / (float) BIGRND);
}

/*** Return random number between -1.0 and 1.0 ***/
float dpn1()
{
    return ((drnd() * 2.0) - 1.0);
}

#if defined(OMP_GPU_OFFLOAD) || defined(OMP_GPU_OFFLOAD_UM)
#pragma omp  declare target
#endif
/*** The squashing function.  Currently, it's a sigmoid. ***/
float squash(float x)
{
  float m;
  //x = -x;
  //m = 1 + x + x*x/2 + x*x*x/6 + x*x*x*x/24 + x*x*x*x*x/120;
  //return(1.0 / (1.0 + m));
  return (1.0 / (1.0 + exp(-x)));
}
#if defined(OMP_GPU_OFFLOAD) || defined(OMP_GPU_OFFLOAD_UM)
#pragma omp end declare target
#endif


/*** Allocate 1d array of floats ***/
float *alloc_1d_dbl(n)
    int n;
{
    float *new;

#if defined(OMP_GPU_OFFLOAD_UM)
    new = (float *) omp_target_alloc((unsigned) (n * sizeof (float)), omp_get_default_device());
#else
    new = (float *) malloc ((unsigned) (n * sizeof (float)));
#endif
    if (new == NULL) {
        printf("ALLOC_1D_DBL: Couldn't allocate array of floats\n");
        return (NULL);
    }
    return (new);
}


/*** Allocate 2d array of floats ***/
float **alloc_2d_dbl(m, n)
    int m, n;
{
    int i;
    float **new;

    #if defined(OMP_GPU_OFFLOAD_UM)
    new = (float **) omp_target_alloc((unsigned) (m * sizeof (float*)), omp_get_default_device());
    #else
    new = (float **) malloc ((unsigned) (m * sizeof (float *)));
    #endif
    if (new == NULL) {
        //printf("ALLOC_2D_DBL: Couldn't allocate array of dbl ptrs\n");
        return (NULL);
    }

    for (i = 0; i < m; i++) {
        new[i] = alloc_1d_dbl(n);
    }
    return (new);
}


void bpnn_randomize_weights(w, m, n)
    float **w;
    int m, n;
{
    int i, j;

#ifdef OMP_GPU_OFFLOAD
    for(i=0; i<=m*n; i++) {
        *w[i] = (float) rand()/RAND_MAX;
    }
#else
    for (i = 0; i <= m; i++) {
        for (j = 0; j <= n; j++) {
            w[i][j] = (float) rand()/RAND_MAX;
        }
    }
#endif
}

void bpnn_randomize_row(w, m)
    float *w;
    int m;
{
    int i;
    for (i = 0; i <= m; i++) {
        w[i] = 0.1;
    }
}


void bpnn_zero_weights(w, m, n)
    float **w;
    int m, n;
{
    int i, j;

#ifdef OMP_GPU_OFFLOAD
    for(i=0; i<=m*n; i++) {
        *w[i] = 0;
    }
#else
    for (i = 0; i <= m; i++) {
        for (j = 0; j <= n; j++) {
            w[i][j] = 0.0;
        }
    }
#endif
}


void bpnn_initialize(seed)
{
//    printf("Random number generator seed: %d\n", seed);
    srand(seed);
}


BPNN *bpnn_internal_create(n_in, n_hidden, n_out)
    int n_in, n_hidden, n_out;
{
    BPNN *newnet;

#if defined(OMP_GPU_OFFLOAD_UM)
    newnet = (BPNN *) omp_target_alloc((unsigned) (sizeof (BPNN)), omp_get_default_device());
#else
    newnet = (BPNN *) malloc (sizeof (BPNN));
#endif
    if (newnet == NULL) {
        //printf("BPNN_CREATE: Couldn't allocate neural network\n");
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

#if defined(OMP_GPU_OFFLOAD)
    float *temp1 = alloc_1d_dbl(n_in * n_hidden + 1);
    float *temp2 = alloc_1d_dbl(n_out * n_hidden + 1);
    float *temp3 = alloc_1d_dbl(n_in * n_hidden + 1);
    float *temp4 = alloc_1d_dbl(n_out * n_hidden + 1);
    newnet->input_weights = &temp1;
    newnet->hidden_weights = &temp2;
    newnet->input_prev_weights = &temp3;
    newnet->hidden_prev_weights =&temp4; 
#else
    newnet->input_weights = alloc_2d_dbl(n_in + 1, n_hidden + 1);
    newnet->hidden_weights = alloc_2d_dbl(n_hidden + 1, n_out + 1);
    newnet->input_prev_weights = alloc_2d_dbl(n_in + 1, n_hidden + 1);
    newnet->hidden_prev_weights = alloc_2d_dbl(n_hidden + 1, n_out + 1);
#endif

    return (newnet);
}


void bpnn_free(net)
    BPNN *net;
{
    int n1, n2, i;

    n1 = net->input_n;
    n2 = net->hidden_n;

#if defined(OMP_GPU_OFFLOAD_UM)
    omp_target_free((char *) net->input_units, omp_get_default_device());
    omp_target_free((char *) net->hidden_units, omp_get_default_device());
    omp_target_free((char *) net->output_units, omp_get_default_device());

    omp_target_free((char *) net->hidden_delta, omp_get_default_device());
    omp_target_free((char *) net->output_delta, omp_get_default_device());
    omp_target_free((char *) net->target, omp_get_default_device());

    for (i = 0; i <= n1; i++) {
        omp_target_free((char *) net->input_weights[i], omp_get_default_device());
        omp_target_free((char *) net->input_prev_weights[i], omp_get_default_device());
    }
    omp_target_free((char *) net->input_weights, omp_get_default_device());
    omp_target_free((char *) net->input_prev_weights, omp_get_default_device());

    for (i = 0; i <= n2; i++) {
        omp_target_free((char *) net->hidden_weights[i], omp_get_default_device());
        omp_target_free((char *) net->hidden_prev_weights[i], omp_get_default_device());
    }
    omp_target_free((char *) net->hidden_weights, omp_get_default_device());
    omp_target_free((char *) net->hidden_prev_weights, omp_get_default_device());

    omp_target_free((char *) net, omp_get_default_device());
#else
    free((char *) net->input_units);
    free((char *) net->hidden_units);
    free((char *) net->output_units);

    free((char *) net->hidden_delta);
    free((char *) net->output_delta);
    free((char *) net->target);

#ifdef OMP_GPU_OFFLOAD
    free((char *) *net->input_weights);
    free((char *) *net->input_prev_weights);
    free((char *) *net->hidden_weights);
    free((char *) *net->hidden_prev_weights);
#else
    for (i = 0; i <= n1; i++) {
        free((char *) net->input_weights[i]);
        free((char *) net->input_prev_weights[i]);
    }
    for (i = 0; i <= n2; i++) {
        free((char *) net->hidden_weights[i]);
        free((char *) net->hidden_prev_weights[i]);
    }
#endif
    free((char *) net->input_weights);
    free((char *) net->input_prev_weights);
    free((char *) net->hidden_weights);
    free((char *) net->hidden_prev_weights);

    free((char *) net);
#endif
}


/*** Creates a new fully-connected network from scratch,
  with the given numbers of input, hidden, and output units.
  Threshold units are automatically included.  All weights are
  randomly initialized.

  Space is also allocated for temporary storage (momentum weights,
  error computations, etc).
 ***/

BPNN *bpnn_create(n_in, n_hidden, n_out)
    int n_in, n_hidden, n_out;
{

    BPNN *newnet;

    newnet = bpnn_internal_create(n_in, n_hidden, n_out);

#ifdef INITZERO
    bpnn_zero_weights(newnet->input_weights, n_in, n_hidden);
#else
    bpnn_randomize_weights(newnet->input_weights, n_in, n_hidden);
#endif
    bpnn_randomize_weights(newnet->hidden_weights, n_hidden, n_out);
    bpnn_zero_weights(newnet->input_prev_weights, n_in, n_hidden);
    bpnn_zero_weights(newnet->hidden_prev_weights, n_hidden, n_out);
    bpnn_randomize_row(newnet->target, n_out);
    return (newnet);
}


void bpnn_layerforward(l1, l2, conn, n1, n2)
    float *l1, *l2, **conn;
    int n1, n2;
{
    float sum;
    int j, k;

    /*** Set up thresholding unit ***/
    l1[0] = 1.0;
    #ifdef OPEN
    //omp_set_num_threads(NUM_THREAD);
    double start_time = omp_get_wtime();
    total_size += sizeof(int)*2    // n1, n2
                + sizeof(float)*n1 + sizeof(float)*n2 + sizeof(float)*n1*n2;
 //   printf("n1=%lu, n2=%lu\n", n1, n2);
    #ifdef OMP_GPU_OFFLOAD
 //   #pragma omp target data map(to: conn[0:n1][0:n2], n1, n2, l1[0:n1]) map(tofrom: sum, l2[0:n2])
 //   #pragma omp target teams distribute parallel for \
            reduction(+: sum) shared(conn, n1, n2, l1)//\
            schedule(static)
    #elif defined(OMP_GPU_OFFLOAD_UM)
 //   #pragma omp target teams distribute parallel for private(j,k) \
            shared(conn, n1, n2, l1) schedule(static) \
            is_device_ptr(conn) is_device_ptr(l1) is_device_ptr(l2) \
            map(to: n1, n2) map(tofrom: sum) reduction(+: sum) 
    #else
    #pragma omp parallel for \
            shared(conn, n1, n2, l1) \
            reduction(+: sum) \
            schedule(static)
    #endif
    #endif  //end OPEN
        /*** For each unit in second layer ***/
        for (int j = 1; j <= n2; j++) {
            float temp = 0.0;

            /*** Compute weighted sum of its inputs ***/
    #ifdef OMP_GPU_OFFLOAD
    #pragma omp target data map(to: conn[0:n1][0:n2], l1[0:n1]) 
    #pragma omp target teams distribute parallel for \
            firstprivate(j, n1) private(k) \
            reduction(+: temp) 
    #elif defined(OMP_GPU_OFFLOAD_UM)
    #pragma omp target teams distribute parallel for \
            firstprivate(j, n1) private(k) \
            is_device_ptr(conn) is_device_ptr(l1) reduction(+:temp)
    #endif
            for (int k = 0; k <= n1; k++) {	
                temp += conn[k][j] * l1[k]; 
            }
            l2[j] = squash(temp);
        }
    double end_time = omp_get_wtime();
    compute_time += end_time - start_time;
//    printf("%f\n",  end_time - start_time);
//    printf("Compute_time = %f\n", compute_time);
}

//extern "C"
void bpnn_output_error(delta, target, output, nj, err)  
    float *delta, *target, *output, *err;
    int nj;
{
    int j;
    float o, t, errsum;
    errsum = 0.0;
    for (j = 1; j <= nj; j++) {
        o = output[j];
        t = target[j];
        delta[j] = o * (1.0 - o) * (t - o);
        errsum += ABS(delta[j]);
    }
    *err = errsum;
}


void bpnn_hidden_error(delta_h,   
        nh, 
        delta_o, 
        no, 
        who, 
        hidden, 
        err)
    float *delta_h, *delta_o, *hidden, **who, *err;
    int nh, no;
{
    int j, k;
    float h, sum, errsum;

    errsum = 0.0;
    for (j = 1; j <= nh; j++) {
        h = hidden[j];
        sum = 0.0;
        for (k = 1; k <= no; k++) {
            sum += delta_o[k] * who[j][k];
        }
        delta_h[j] = h * (1.0 - h) * sum;
        errsum += ABS(delta_h[j]);
    }
    *err = errsum;
}


void bpnn_adjust_weights(delta, ndelta, ly, nly, w, oldw)
    float *delta, *ly, **w, **oldw;
{
    float new_dw;
    int k, j;
    ly[0] = 1.0;
    //eta = 0.3;
    //momentum = 0.3;

    #ifdef OPEN
 //   omp_set_num_threads(NUM_THREAD);
    total_size += sizeof(int)*2        // ndelta, nly
                + sizeof(float)*ndelta              // detla
                + sizeof(float)*nly                 // ly
                + sizeof(float)*nly*ndelta          // w
                + sizeof(float)*nly*ndelta;         // oldw
    double start_time = omp_get_wtime();
  //  printf("ndelta=%lu, nly=%lu\n", ndelta, nly);
    #ifdef OMP_GPU_OFFLOAD
    #pragma omp target data map(to: delta[0:ndelta], ly[0:nly]) map(tofrom: w[0:nly][0:ndelta], oldw[0:nly][0:ndelta])
    #pragma omp target teams distribute parallel for \
            private(j, k, new_dw) \
            firstprivate(ndelta, nly) 
    #elif defined(OMP_GPU_OFFLOAD_UM)
//    #pragma omp target data map(to: delta, ndelta, nly) map(tofrom: w, oldw)
    #pragma omp target teams distribute parallel for shared(oldw, w, delta) \
            private(j, k, new_dw) firstprivate(ndelta, nly) \
            is_device_ptr(delta) is_device_ptr(ly) is_device_ptr(w) is_device_ptr(oldw) 
    #else
    #pragma omp parallel for \
        shared(oldw, w, delta) \
        private(j, k, new_dw) \
        firstprivate(ndelta, nly) 
 //   printf("##################################################################################%s -- %d\n", __FILE__, __LINE__);
    #endif
    #endif // end OPEN 
    for (k = 0; k <= nly; k++) {
        for (j = 1; j <= ndelta; j++) {
        //for (k = 0; k <= nly; k++) {
            new_dw = ((ETA * delta[j] * ly[k]) + (MOMENTUM * oldw[k][j]));
            w[k][j] += new_dw;
            oldw[k][j] = new_dw;
        }
    }
    double end_time = omp_get_wtime();
    compute_time += end_time - start_time;
//    printf("##################################################################################%s -- %d\n", __FILE__, __LINE__);
//    printf("done %f\n", end_time - start_time);
}


void bpnn_save(net, filename)
    BPNN *net;
    char *filename;
{
    int n1, n2, n3, i, j, memcnt;
    float dvalue, **w;
    char *mem;
    ///add//
    FILE *pFile;
    pFile = fopen( filename, "w+" );
    ///////
    /*
       if ((fd = creat(filename, 0644)) == -1) {
       printf("BPNN_SAVE: Cannot create '%s'\n", filename);
       return;
       }
       */

    n1 = net->input_n;  n2 = net->hidden_n;  n3 = net->output_n;
//    printf("Saving %dx%dx%d network to '%s'\n", n1, n2, n3, filename);
    //fflush(stdout);

    //write(fd, (char *) &n1, sizeof(int));
    //write(fd, (char *) &n2, sizeof(int));
    //write(fd, (char *) &n3, sizeof(int));

    fwrite( (char *) &n1 , sizeof(char), sizeof(char), pFile);
    fwrite( (char *) &n2 , sizeof(char), sizeof(char), pFile);
    fwrite( (char *) &n3 , sizeof(char), sizeof(char), pFile);



    memcnt = 0;
    w = net->input_weights;
    mem = (char *) malloc ((unsigned) ((n1+1) * (n2+1) * sizeof(float)));
    for (i = 0; i <= n1; i++) {
        for (j = 0; j <= n2; j++) {
            dvalue = w[i][j];
            fastcopy(&mem[memcnt], &dvalue, sizeof(float));
            memcnt += sizeof(float);
        }
    }
    //write(fd, mem, (n1+1) * (n2+1) * sizeof(float));
    fwrite( mem , (unsigned)(sizeof(float)), (unsigned) ((n1+1) * (n2+1) * sizeof(float)) , pFile);
    free(mem);

    memcnt = 0;
    w = net->hidden_weights;
    mem = (char *) malloc ((unsigned) ((n2+1) * (n3+1) * sizeof(float)));
    for (i = 0; i <= n2; i++) {
        for (j = 0; j <= n3; j++) {
            dvalue = w[i][j];
            fastcopy(&mem[memcnt], &dvalue, sizeof(float));
            memcnt += sizeof(float);
        }
    }
    //write(fd, mem, (n2+1) * (n3+1) * sizeof(float));
    fwrite( mem , sizeof(float), (unsigned) ((n2+1) * (n3+1) * sizeof(float)) , pFile);
    free(mem);

    fclose(pFile);
    return;
}


BPNN *bpnn_read(filename)
    char *filename;
{
    char *mem;
    BPNN *new;
    int fd, n1, n2, n3, i, j, memcnt;

    if ((fd = open(filename, 0, 0644)) == -1) {
        return (NULL);
    }

    //printf("Reading '%s'\n", filename);  //fflush(stdout);

    read(fd, (char *) &n1, sizeof(int));
    read(fd, (char *) &n2, sizeof(int));
    read(fd, (char *) &n3, sizeof(int));
    new = bpnn_internal_create(n1, n2, n3);

    //printf("'%s' contains a %dx%dx%d network\n", filename, n1, n2, n3);
    //printf("Reading input weights...");  //fflush(stdout);

    memcnt = 0;
    mem = (char *) malloc ((unsigned) ((n1+1) * (n2+1) * sizeof(float)));
    read(fd, mem, (n1+1) * (n2+1) * sizeof(float));
    for (i = 0; i <= n1; i++) {
        for (j = 0; j <= n2; j++) {
            fastcopy(&(new->input_weights[i][j]), &mem[memcnt], sizeof(float));
            memcnt += sizeof(float);
        }
    }
    free(mem);

    //printf("Done\nReading hidden weights...");  //fflush(stdout);

    memcnt = 0;
    mem = (char *) malloc ((unsigned) ((n2+1) * (n3+1) * sizeof(float)));
    read(fd, mem, (n2+1) * (n3+1) * sizeof(float));
    for (i = 0; i <= n2; i++) {
        for (j = 0; j <= n3; j++) {
            fastcopy(&(new->hidden_weights[i][j]), &mem[memcnt], sizeof(float));
            memcnt += sizeof(float);
        }
    }
    free(mem);
    close(fd);

    //printf("Done\n");  //fflush(stdout);

    bpnn_zero_weights(new->input_prev_weights, n1, n2);
    bpnn_zero_weights(new->hidden_prev_weights, n2, n3);

    return (new);
}
