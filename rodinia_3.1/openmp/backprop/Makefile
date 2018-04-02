# C compiler
CC = clang
#PLUGIN_FLAG = -Xclang -load -Xclang ~/git/unifiedmem/code/llvm-pass/build/uvm/libOMPPass.so
CC_FLAGS = -g -fopenmp -O2 -fopenmp-targets=nvptx64-nvidia-cuda $(PLUGIN_FLAG)

all: backprop um

backprop: backprop.o facetrain.o imagenet.o backprop_kernel.o 
	$(CC) $(CC_FLAGS) backprop.o facetrain.o imagenet.o backprop_kernel.o -o backprop -lm

#gpu: backprop_gpu.o facetrain.o imagenet.o backprop_kernel.o 
#	$(CC) $(CC_FLAGS) -fopenmp-targets=nvptx64-nvidia-cuda backprop_gpu.o facetrain.o imagenet.o backprop_kernel.o -o backprop_gpu -lm

um: backprop_gpu_um.o facetrain.o imagenet.o backprop_kernel.o 
	$(CC) $(CC_FLAGS) backprop_gpu_um.o facetrain.o imagenet.o backprop_kernel.o -o backprop_gpu_um  -lm

%.o: %.[ch]
	$(CC) $(CC_FLAGS) $< -c

facetrain.o: facetrain.c backprop.h
	$(CC) $(CC_FLAGS) facetrain.c -c
	
backprop.o: backprop_gpu.c backprop.h
	$(CC) $(CC_FLAGS) backprop_gpu.c -c -o backprop.o

#backprop_gpu.o: backprop_gpu.c backprop.h
#	$(CC) $(CC_FLAGS) -DOMP_GPU_OFFLOAD -fopenmp-targets=nvptx64-nvidia-cuda backprop_gpu.c -c -o backprop_gpu.o

backprop_gpu_um.o: backprop_gpu.c backprop.h
	$(CC) $(CC_FLAGS) -DOMP_GPU_OFFLOAD_UM -fopenmp-targets=nvptx64-nvidia-cuda backprop_gpu.c -c -o backprop_gpu_um.o

backprop_kernel.o: backprop_kernel.c backprop.h
	$(CC) $(CC_FLAGS) backprop_kernel.c -c

imagenet.o: imagenet.c backprop.h
	$(CC) $(CC_FLAGS) imagenet.c -c

clean:
	rm -f *.o *~ backprop backprop_gpu backprop_gpu_um backprop_cuda.linkinfo
