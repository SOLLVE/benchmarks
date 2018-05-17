LOCAL_CC = clang -g -O3 -Wall
CC = clang++
CFLAGS = -lm -fopenmp -Wall


all : nn

clean :
	rm -rf *.o nn nn_gpu nn_um

gpu:
	$(CC) -o nn_gpu nn_omp.cpp $(LDFLAGS) $(CFLAGS) -DOMP_GPU_OFFLOAD -fopenmp-targets=nvptx64-nvidia-cuda

um:
	$(CC) -o nn_um nn_omp.cpp $(LDFLAGS) $(CFLAGS) -DOMP_GPU_OFFLOAD_UM -fopenmp-targets=nvptx64-nvidia-cuda

nn : nn_openmp.c
	$(CC) -o $@ $< $(LDFLAGS) $(CFLAGS) 

hurricane_gen : hurricane_gen.c
	$(LOCAL_CC) -o $@ $< -lm

