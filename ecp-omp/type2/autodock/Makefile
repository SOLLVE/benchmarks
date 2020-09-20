CC         = xlc
CC         = clang
CXX        = clang++
LD         = $(CXX)
DEBUG      = -g4

#DEBUG = -g
OPT        = -O2 -std=c++11		# O3 crashes
OMPOFF = -qoffload
OMPFL  = -fopenmp
OMP        = $(OMPOFF) $(OMPFL)

# for clang

#OMP = -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda --cuda-path=${OLCF_CUDA_ROOT} 

CFLAGS     = $(OMP) $(DEBUG) $(OPT)
LDFLAGS    = $(CFLAGS)

.PHONY: all tidy clean

all:	bench bench_works

bench:	bench.o
	$(LD) $(LDFLAGS) -o $@ $^

bench_works:	bench_works.o
	$(LD) $(LDFLAGS) -o $@ $^

# ignore this one right now, for some diagnostic tests by Vivek on cpu
bench_ignore:	bench_ignore.o
	xlc $(LDFLAGS) -o $@ $^

tidy:
	rm -f bench.o bench_works.o bench_ignore.o

clean:	tidy
	rm -f bench bench_works bench_ignore
