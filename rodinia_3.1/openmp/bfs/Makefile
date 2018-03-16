# C compiler
CC = clang
CXX = clang++
ICC = clang
#CC_FLAGS = -fopenmp -O3 -I/usr/local/cuda-9.0/include -lcudart
CC_FLAGS = -fopenmp -O3 -I/sw/summitdev/cuda/9.0.69/include -lcudart -I/sw/summitdev/cuda/9.0.69/include -L/sw/summitdev/cuda/9.0.69/lib64
GPU_FLAGS = -DOMP_GPU_OFFLOAD -fopenmp-targets=nvptx64-nvidia-cuda 
UM_FLAGS = -DOMP_GPU_OFFLOAD_UM -fopenmp-targets=nvptx64-nvidia-cuda 

all: bfs bfs_offload gpu um

bfs: bfs.cpp 
	$(CC) $(CC_FLAGS) bfs_gpu.cpp -o bfs 

bfs_offload: bfs.cpp
	$(ICC) $(CC_FLAGS) -DOMP_OFFLOAD bfs.cpp -o bfs_offload

gpu: bfs_gpu.cpp
	$(CXX) $(CC_FLAGS) $(GPU_FLAGS) bfs_gpu.cpp -o bfs_gpu

um: bfs_gpu.cpp
	$(CXX) $(CC_FLAGS) $(UM_FLAGS) bfs_gpu.cpp -o bfs_gpu_um

mgpu: bfs_mulgpu.cpp
	$(CXX) $(CC_FLAGS) $(UM_FLAGS) bfs_mulgpu.cpp -o bfs_mulgpu

clean:
	rm -f bfs bfs_offload bfs_gpu bfs_gpu_um result.txt
