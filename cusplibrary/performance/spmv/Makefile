ifndef INPUT
  INPUT = cant
endif
ifndef TESTVERSION
  TESTVERSION = um
endif

ifdef TESTV_EXTRA
  FULLTESTVERSION = $(TESTVERSION)-$(TESTV_EXTRA)
else
  FULLTESTVERSION = $(TESTVERSION)
endif

ifdef INPUTTIME
  NVCCINTRAFLAGS = -Xcompiler -DINPUT_TIME=$(INPUTTIME)
  INPUTNAME = $(INPUT)-$(INPUTTIME)
else
  NVCCINTRAFLAGS =
  INPUTNAME = $(INPUT)
endif

MATRIXPATH = /ccs/home/lld/data/matrices/
GRAPHPATH = /ccs/home/lld/data/graphs/
PROFILEPATH = /ccs/home/lld/profiling/
INPUTFILE = $(MATRIXPATH)$(INPUT).mtx
BINARYFILE = spmv_$(TESTVERSION)
NVCCCOMMONFLAGS = $(NVCCEXTRAFLAGS) $(NVCCINTRAFLAGS) -c -arch=sm_60 -Xcompiler -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA -Xcompiler -DCUSP_DEVICE_BLAS_SYSTEM=CUSP_DEVICE_BLAS_GENERIC -Xcompiler -DCUSP_HOST_BLAS_SYSTEM=CUSP_HOST_BLAS_GENERIC -Xcompiler -Wall -I /ccs/home/lld/benchmarks/cusplibrary -I /sw/summitdev/cuda/8.0.61-1/include/
NVCCFLAGS = $(NVCCCOMMONFLAGS) -Xcompiler -O3
GCCFLAGS = -L/sw/summitdev/cuda/8.0.61-1/lib64 -lcudart -lstdc++ -lm
DATE = `date +'%m.%d.%y'`
CURPATH = `pwd`

all: num um umpf host

num:
	nvcc -o spmv.o $(NVCCFLAGS) spmv.cu
	gcc -o spmv_num spmv.o $(GCCFLAGS)

um:
	nvcc -o spmv.o -Xcompiler -DCUDA_UM_ALLOC $(NVCCFLAGS) spmv.cu
	gcc -o spmv_um spmv.o $(GCCFLAGS)

umpf:
	nvcc -o spmv.o -Xcompiler -DCUDA_UM_ALLOC -Xcompiler -DCUDA_UM_PREFETCH $(NVCCFLAGS) spmv.cu
	gcc -o spmv_umpf spmv.o $(GCCFLAGS)

h:
	nvcc -o spmv.o -Xcompiler -DCUDA_HOST_ALLOC $(NVCCFLAGS) spmv.cu
	gcc -o spmv_h spmv.o $(GCCFLAGS)

hyb:
	nvcc -o spmv.o -Xcompiler -DCUDA_HYBRID_ALLOC $(NVCCFLAGS) spmv.cu
	gcc -o spmv_hyb spmv.o $(GCCFLAGS)

thyb:
	nvcc -o spmv.o -Xcompiler -DCUDA_TRUE_HYBRID_ALLOC $(NVCCFLAGS) spmv.cu
	gcc -o spmv_thyb spmv.o $(GCCFLAGS)

dbg:
	nvcc -o spmv.o -Xcompiler -g $(NVCCCOMMONFLAGS) spmv.cu
	gcc -g -o spmv_dbg spmv.o $(GCCFLAGS)

clean:
	-rm spmv.o spmv_*


# code version
codehyb:
	-rm benchmark.h
	ln -s $(CURPATH)/code/benchmark.hybrid.h benchmark.h
	-rm ../../cusp/system/cuda/detail/multiply/csr_block_spmv.h
	ln -s $(CURPATH)/code/csr_block_spmv.my.h ../../cusp/system/cuda/detail/multiply/csr_block_spmv.h

codeori:
	-rm benchmark.h
	ln -s $(CURPATH)/code/benchmark.ori.h benchmark.h
	-rm ../../cusp/system/cuda/detail/multiply/csr_block_spmv.h
	ln -s $(CURPATH)/code/csr_block_spmv.my.h ../../cusp/system/cuda/detail/multiply/csr_block_spmv.h

codetruehyb:
	-rm benchmark.h
	ln -s $(CURPATH)/code/benchmark.hybrid.h benchmark.h
	-rm ../../cusp/system/cuda/detail/multiply/csr_block_spmv.h
	ln -s $(CURPATH)/code/csr_block_spmv.hyb.h ../../cusp/system/cuda/detail/multiply/csr_block_spmv.h


# test
test:
	./$(BINARYFILE) $(INPUTFILE)

testprofile:
	nvprof --export-profile $(PROFILEPATH)$(FULLTESTVERSION)-$(INPUTNAME)-$(DATE).prof -f ./$(BINARYFILE) $(INPUTFILE)

testall:
	@echo ""
#	-@$(MAKE) test TESTVERSION=num
	-@$(MAKE) test TESTVERSION=um
#	-@$(MAKE) test TESTVERSION=umpf
#	-@$(MAKE) test TESTVERSION=h

experiment:
	-@$(MAKE) test &> /dev/null
	-@$(MAKE) testall
	-@$(MAKE) testall
	-@$(MAKE) testall
	-@$(MAKE) testall
	-@$(MAKE) testall
