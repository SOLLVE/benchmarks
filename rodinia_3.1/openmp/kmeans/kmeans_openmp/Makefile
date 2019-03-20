# C compiler
CC = clang
CC_FLAGS = -g -fopenmp -O2
PLUGIN_FLAG = -Xclang -load -Xclang ~/git/unifiedmem/code/llvm-pass/build/uvm/libOMPPass.so
GPU_FLAGS_UM = -DOMP_GPU_OFFLOAD_UM -fopenmp-targets=nvptx64-nvidia-cuda

all: kmeans um

kmeans: cluster.o kmeans.o kmeans_clustering.o 
	$(CC) $(CC_FLAGS) cluster.o kmeans.o kmeans_clustering.o  -o kmeans

um: cluster_um.o kmeans_um.o kmeans_clustering_um.o 
	$(CC) $(CC_FLAGS) $(GPU_FLAGS_UM) cluster_um.o kmeans_um.o kmeans_clustering_um.o  -o kmeans_um

%.o: %.[ch]
	$(CC) $(CC_FLAGS) $< -c

cluster.o: cluster.c 
	$(CC) $(CC_FLAGS) cluster.c -c

cluster_um.o: cluster.c 
	$(CC) $(CC_FLAGS) $(GPU_FLAGS_UM) cluster.c -c -o cluster_um.o
	
kmeans.o: kmeans.c 
	$(CC) $(CC_FLAGS) kmeans.c -c

kmeans_um.o: kmeans.c 
	$(CC) $(CC_FLAGS) $(GPU_FLAGS_UM) kmeans.c -c -o kmeans_um.o

kmeans_clustering.o: kmeans_clustering.c kmeans.h
	$(CC) $(CC_FLAGS) kmeans_clustering.c -c

kmeans_clustering_um.o: kmeans_clustering.c kmeans.h
	$(CC) $(CC_FLAGS) $(GPU_FLAGS_UM) kmeans_clustering.c -c -o kmeans_clustering_um.o $(PLUGIN_FLAG)

clean:
	rm -f *.o *~ kmeans kmeans_gpu kmeans_um
