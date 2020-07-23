CC         = xlc
#CC         = clang++
LD         = $(CC)


DEBUG      = -g4

#DEBUG = -g
OPT        = -O2 -std=c++11		# O3 crashes

# for xlc
OMPOFF = -qoffload
OMPFL  = -fopenmp
OMP        = $(OMPOFF) $(OMPFL)

# for clang

#OMP = -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda --cuda-path=${OLCF_CUDA_ROOT} 

CFLAGS     = $(OMP) $(DEBUG) $(OPT)
LDFLAGS    = $(CFLAGS)

.PHONY: all tidy clean

all:	bench bench_stencil

bench:	bench.o
	$(LD) $(LDFLAGS) -o $@ $^


bench_stencil:	bench_stencil.o
	$(LD) $(LDFLAGS) -o $@ $^

tidy:
	rm -f bench.o bench_stencil.o

clean:	tidy
	rm -f bench bench_stencil
