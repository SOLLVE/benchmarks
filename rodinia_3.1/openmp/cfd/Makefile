CC = clang
CXX = clang++
CC_FLAGS = -g -fopenmp -O2 -fmodules
OFFLOAD_CC_FLAGS = -DOMP_OFFLOAD -offload-option,mic,compiler,"-no-opt-prefetch"
GPU_FLAGS = -DOMP_GPU_OFFLOAD -fopenmp-targets=nvptx64-nvidia-cuda
#UM_FLAGS = -DOMP_GPU_OFFLOAD_UM -fopenmp-targets=nvptx64-nvidia-cuda -I/sw/summit/cuda/9.0.184/include -lcudart
UM_FLAGS = -DOMP_GPU_OFFLOAD_UM -fopenmp-targets=nvptx64-nvidia-cuda -I/sw/summitdev/cuda/9.0.69/include -lcudart
PLUGIN_FLAG = -Xclang -load -Xclang ~/git/unifiedmem/code/llvm-pass/build/uvm/libOMPPass.so

OMP_NUM_THREADS = 8
MIC_OMP_NUM_THREADS = 216

all:  euler double pre_euler pre_double

euler:	euler3d_cpu gpu um 

double:	euler3d_cpu_double euler3d_cpu_double_offload gpu_double

pre_euler: pre_euler3d_cpu pre_euler3d_cpu_offload gpu_pre

pre_double: pre_euler3d_cpu_double 

euler3d_cpu: euler3d_gpu.cpp
	$(CXX) $(CC_FLAGS) -Dblock_length=$(OMP_NUM_THREADS) euler3d_gpu.cpp -o euler3d_cpu

euler3d_cpu_offload:
	$(CXX) $(CC_FLAGS) $(OFFLOAD_CC_FLAGS) -Dblock_length=$(MIC_OMP_NUM_THREADS) euler3d_cpu.cpp -o euler3d_cpu_offload

gpu:
	$(CXX) $(CC_FLAGS) $(GPU_FLAGS) -Dblock_length=$(OMP_NUM_THREADS) euler3d_omp.cpp -o euler3d_cpu_gpu 

um:
	$(CXX) $(CC_FLAGS) $(UM_FLAGS) -Dblock_length=$(OMP_NUM_THREADS) euler3d_omp.cpp -o euler3d_cpu_gpu_um $(PLUGIN_FLAG)

euler3d_cpu_double: euler3d_cpu_double.cpp
	$(CXX) $(CC_FLAGS) -Dblock_length=$(OMP_NUM_THREADS) euler3d_cpu_double.cpp -o euler3d_cpu_double

euler3d_cpu_double_offload: euler3d_cpu_double.cpp
	$(CXX) $(CC_FLAGS) $(OFFLOAD_CC_FLAGS) -Dblock_length=$(OMP_NUM_THREADS) euler3d_cpu_double.cpp -o euler3d_cpu_double_offload

gpu_double: euler3d_cpu_double.cpp
	$(CXX) $(CC_FLAGS) $(GPU_FLAGS) -Dblock_length=$(OMP_NUM_THREADS) euler3d_cpu_double.cpp -o euler3d_cpu_double_gpu

pre_euler3d_cpu: pre_euler3d_cpu.cpp
	$(CXX) $(CC_FLAGS) -Dblock_length=$(OMP_NUM_THREADS) pre_euler3d_cpu.cpp -o pre_euler3d_cpu

pre_euler3d_cpu_offload: pre_euler3d_cpu.cpp
	$(CXX) $(CC_FLAGS) $(OFFLOAD_CC_FLAGS) -Dblock_length=$(OMP_NUM_THREADS) pre_euler3d_cpu.cpp -o pre_euler3d_cpu_offload

gpu_pre: pre_euler3d_cpu.cpp
	$(CXX) $(CC_FLAGS) $(GPU_FLAGS) -Dblock_length=$(OMP_NUM_THREADS) pre_euler3d_cpu.cpp -o pre_euler3d_cpu_gpu

pre_euler3d_cpu_double: pre_euler3d_cpu_double.cpp
	$(CXX) -O3 -Dblock_length=$(OMP_NUM_THREADS) -fopenmp pre_euler3d_cpu_double.cpp -o pre_euler3d_cpu_double

clean:
	rm -f euler3d_cpu euler3d_cpu_offload euler3d_cpu_gpu euler3d_cpu_gpu_um
	rm -f euler3d_cpu_double euler3d_cpu_double_offload euler3d_cpu_double_gpu
	rm -f pre_euler3d_cpu  pre_euler3d_cpu_offload  pre_euler3d_cpu_gpu
	rm -f pre_euler3d_cpu_double 
	rm -f density density_energy momentum
