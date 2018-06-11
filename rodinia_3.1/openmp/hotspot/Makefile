# C compiler
CC = clang++
ICC = icc
CC_FLAGS = -g -fopenmp -O3
OFFLOAD_CC_FLAGS = -offload-option,mic,compiler,"-no-opt-prefetch"
PLUGIN_FLAG = -Xclang -load -Xclang ~/git/unifiedmem/code/llvm-pass/build/uvm/libOMPPass.so

all: hotspot hotspot_offload 


hotspot: hotspot_openmp.cpp Makefile 
	$(CC) $(CC_FLAGS) hotspot_openmp.cpp -o hotspot 

hotspot_offload: hotspot_openmp.cpp Makefile
	$(ICC) $(CC_FLAGS) $(OFFLOAD_CC_FLAGS) -DOMP_OFFLOAD hotspot_openmp.cpp -o hotspot_offload

gpu: hotspot_openmp.cpp Makefile 
	$(CC) $(CC_FLAGS) hotspot_openmp.cpp -o hotspot_gpu -DOMP_GPU_OFFLOAD -fopenmp-targets=nvptx64-nvidia-cuda

um: hotspot_openmp.cpp Makefile 
	$(CC) $(CC_FLAGS) hotspot_openmp.cpp -o hotspot_um -DOMP_GPU_OFFLOAD_UM -fopenmp-targets=nvptx64-nvidia-cuda $(PLUGIN_FLAG)

umll: hotspot_openmp.cpp Makefile 
	$(CC) $(CC_FLAGS) hotspot_openmp.cpp -c -S -emit-llvm -o hotspot_um.ll -DOMP_GPU_OFFLOAD_UM -fopenmp-targets=nvptx64-nvidia-cuda $(PLUGIN_FLAG)

clean:
	rm -f hotspot hotspot_offload
