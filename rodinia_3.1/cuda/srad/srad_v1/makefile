include ../../../common/make.config.summitdev

CC := $(CUDA_DIR)/bin/nvcc

INCLUDE := $(CUDA_DIR)/include

# Example
# target: dependencies
	# command 1
	# command 2
          # .
          # .
          # .
	# command n

# link objects(binaries) together
a.out:		main.o
	$(CC)	main.o \
				-I$(INCLUDE) \
				-L$(CUDA_LIB_DIR)  \
				-lm -lcuda -lcudart -o srad

# compile main function file into object (binary)
main.o: 	main_lin.cu \
				define.c \
				graphics.c \
				extract_kernel.cu \
				prepare_kernel.cu \
				reduce_kernel.cu \
				srad_kernel.cu \
				srad2_kernel.cu \
				compress_kernel.cu
	nvcc	main_lin.cu \
				-c -O3 -arch sm_60 -o main.o

# delete all object files
clean:
	rm *.o srad
