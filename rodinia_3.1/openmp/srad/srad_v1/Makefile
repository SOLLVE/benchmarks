# Example
# target: dependencies
	# command 1
	# command 2
          # .
          # .
          # .
	# command n

# link objects(binaries) together

PLUGIN_FLAG = -Xclang -load -Xclang ~/git/unifiedmem/code/llvm-pass/build/uvm/libOMPPass.so

all: a.out um gpu

a.out:	main.o
	clang	main.o -lm -fopenmp -o srad

um: main_um.o
	clang	-O2 main_um.o -lm -fopenmp -o srad_um -fopenmp-targets=nvptx64-nvidia-cuda

gpu: main_gpu.o
	clang	main_gpu.o -lm -fopenmp -o srad_gpu -fopenmp-targets=nvptx64-nvidia-cuda

# compile main function file into object (binary)
main.o: 	main.c resize.c define.c graphics.c
	clang	main.c -c -O3 -fopenmp

main_um.o: 	main.c resize.c define.c graphics.c
	clang	main.c -c -O3 -fopenmp -DOMP_GPU_OFFLOAD_UM -fopenmp-targets=nvptx64-nvidia-cuda -o main_um.o $(PLUGIN_FLAG)

main_gpu.o: 	main.c resize.c define.c graphics.c
	clang	main.c -c -O3 -fopenmp -DOMP_GPU_OFFLOAD -fopenmp-targets=nvptx64-nvidia-cuda -o main_gpu.o

# delete all object files
clean:
	rm *.o srad*
