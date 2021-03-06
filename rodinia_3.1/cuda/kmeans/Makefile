include ../../common/make.config.summitdev

# C compiler
CC = g++
CC_FLAGS = -g -fopenmp -O2

# CUDA compiler
NVCC = $(CUDA_DIR)/bin/nvcc
NVCC_FLAGS = -I$(CUDA_DIR)/include 

# 'make dbg=1' enables NVCC debugging
ifeq ($(dbg),1)
	NVCC_FLAGS += -g -O0
else
	NVCC_FLAGS += -O2
endif

# 'make emu=1' compiles the CUDA kernels for emulation
ifeq ($(emu),1)
	NVCC_FLAGS += -deviceemu
endif

NVCC_FLAGS += -arch=sm_60

all: kmeans kmeans_lin

#kmeans: cluster.o getopt.o kmeans.o kmeans_clustering.o kmeans_cuda.o rmse.o
#	$(CC) $(CC_FLAGS) cluster.o getopt.o kmeans.o kmeans_clustering.o kmeans_cuda.o rmse.o -o kmeans -L$(CUDA_LIB_DIR) -lcuda -lcudart -lm
#kmeans: cluster.o getopt.o kmeans.o kmeans_clustering.o rmse.o
#	$(CC) $(CC_FLAGS) cluster.o getopt.o kmeans.o kmeans_clustering.o rmse.o -o kmeans -L$(CUDA_LIB_DIR) -lcuda -lcudart -lm
kmeans: cluster.o getopt.o kmeans.o rmse.o
	$(CC) $(CC_FLAGS) cluster.o getopt.o kmeans.o rmse.o -o kmeans -L$(CUDA_LIB_DIR) -lcuda -lcudart -lm

kmeans_lin: cluster.o getopt.o kmeans_lin.o rmse.o
	$(CC) $(CC_FLAGS) cluster.o getopt.o kmeans_lin.o rmse.o -o kmeans_lin -L$(CUDA_LIB_DIR) -lcuda -lcudart -lm

%.o: %.[c]
	$(CC) $(CC_FLAGS) $< -c
#	$(NVCC) $(NVCC_FLAGS) $< -c

kmeans.o: kmeans.cu kmeans_clustering.cu kmeans_cuda.cu
	$(NVCC) $(NVCC_FLAGS) -c kmeans.cu

kmeans_lin.o: kmeans_lin.cu kmeans_clustering.cu kmeans_cuda.cu
	$(NVCC) $(NVCC_FLAGS) -c kmeans_lin.cu

#kmeans_clustering.o: kmeans_clustering.cu kmeans_cuda.cu
#	$(NVCC) $(NVCC_FLAGS) -c kmeans_clustering.cu

#kmeans_cuda.o: kmeans_cuda.cu 
#	$(NVCC) $(NVCC_FLAGS) -c kmeans_cuda.cu

clean:
	rm -f *.o *~ kmeans kmeans_cuda.linkinfo
