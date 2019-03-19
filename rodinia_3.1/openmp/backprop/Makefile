# C compiler
CC = clang
PLUGIN_FLAG = -Xclang -load -Xclang ~/git/unifiedmem/code/llvm-pass/build/uvm/libOMPPass.so
CC_FLAGS = -g -DOMP_GPU_OFFLOAD_UM -fopenmp -O2 -fopenmp-targets=nvptx64-nvidia-cuda $(PLUGIN_FLAG)
BC_FLAGS = -cc1 -DOMP_GPU_OFFLOAD_UM -fopenmp -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc
LL_DEV_FLAGS = -cc1 -DOMP_GPU_OFFLOAD_UM -fopenmp -triple nvptx64-nvidia-cuda -fopenmp-targets=nvptx64-nvidia-cuda -target-cpu sm_70 -emit-llvm -I. -fopenmp-is-device -fopenmp-host-ir-file-path
ALL_FLAGS = -DOMP_GPU_OFFLOAD_UM -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm -save-temps

all: backprop um

backprop: backprop.o facetrain.o imagenet.o backprop_kernel.o 
	$(CC) $(CC_FLAGS) backprop.o facetrain.o imagenet.o backprop_kernel.o -o backprop -lm

#gpu: backprop_gpu.o facetrain.o imagenet.o backprop_kernel.o 
#	$(CC) $(CC_FLAGS) backprop_gpu.o facetrain.o imagenet.o backprop_kernel.o -o backprop_gpu -lm

um: backprop_gpu_um.o facetrain.o imagenet.o backprop_kernel.o 
	$(CC) $(CC_FLAGS) backprop_gpu_um.o facetrain.o imagenet.o backprop_kernel.o -o backprop_gpu_um  -lm

%.o: %.[ch]
	$(CC) $(CC_FLAGS) $< -c

facetrain.o: facetrain.c backprop.h
	$(CC) $(CC_FLAGS) facetrain.c -c

#backprop.o: backprop_gpu.c backprop.h
#	$(CC) $(CC_FLAGS) backprop_gpu.c -c -o backprop.o

backprop.o: backprop.c backprop.h
	$(CC) $(CC_FLAGS) backprop.c -c -o backprop.o

backprop_gpu.o: backprop_gpu.c backprop.h
	$(CC) $(CC_FLAGS) -DOMP_GPU_OFFLOAD backprop_gpu.c -c -o backprop_gpu.o

backprop_gpu_um.o: backprop_gpu.c backprop.h
	$(CC) $(CC_FLAGS) -DOMP_GPU_OFFLOAD_UM backprop_gpu.c -c -o backprop_gpu_um.o

backprop_kernel.o: backprop_kernel.c backprop.h
	$(CC) $(CC_FLAGS) backprop_kernel.c -c

imagenet.o: imagenet.c backprop.h
	$(CC) $(CC_FLAGS) imagenet.c -c

facetrain.bc: facetrain.c backprop.h
	$(CC) $(BC_FLAGS) facetrain.c

backprop.bc: backprop.c backprop.h
	$(CC) $(BC_FLAGS) backprop.c

backprop_kernel.bc: backprop_kernel.c backprop.h
	$(CC) $(BC_FLAGS) backprop_kernel.c

imagenet.bc: imagenet.c backprop.h
	$(CC) $(BC_FLAGS) imagenet.c

host.bc: facetrain.bc backprop.bc backprop_kernel.bc imagenet.bc
	llvm-link facetrain.bc backprop.bc backprop_kernel.bc imagenet.bc -o host.bc

facetrain-dev.ll: facetrain.c backprop.h facetrain.bc
	$(CC) $(LL_DEV_FLAGS) facetrain.bc facetrain.c -o facetrain-dev.ll

backprop-dev.ll: backprop.c backprop.h backprop.bc
	$(CC) $(LL_DEV_FLAGS) backprop.bc backprop.c -o backprop-dev.ll

backprop_kernel-dev.ll: backprop_kernel.c backprop.h backprop_kernel.bc
	$(CC) $(LL_DEV_FLAGS) backprop_kernel.bc backprop_kernel.c -o backprop_kernel-dev.ll

imagenet-dev.ll: imagenet.c backprop.h imagenet.bc
	$(CC) $(LL_DEV_FLAGS) imagenet.bc imagenet.c -o imagenet-dev.ll

dev.bc: facetrain-dev.ll backprop-dev.ll backprop_kernel-dev.ll imagenet-dev.ll
	llvm-link facetrain-dev.ll backprop-dev.ll backprop_kernel-dev.ll imagenet-dev.ll -o dev.bc

gpu: host.bc dev.bc
#	$(CC) $(CC_FLAGS) host.bc dev.bc -o backprop_gpu -lm
	clang-offload-bundler -type=bc -inputs=host.bc,dev.bc -targets=host-powerpc64le-unknown-unknown,openmp-nvptx64-nvidia-cuda -outputs=/ccs/home/lld/benchmarks/rodinia_3.1/openmp-summit/backprop/main.bc
	clang "-cc1" "-triple" "powerpc64le-unknown-linux-gnu" "-emit-llvm-bc" "-emit-llvm-uselists" "-disable-free" "-disable-llvm-verifier" "-discard-value-names" "-main-file-name" main.bc "-mrelocation-model" "static" "-mthread-model" "posix" "-fmath-errno" "-masm-verbose" "-mconstructor-aliases" "-fuse-init-array" "-target-cpu" "ppc64le" "-mfloat-abi" "hard" "-target-abi" "elfv2" "-dwarf-column-info" "-debugger-tuning=gdb" "-momit-leaf-frame-pointer" "-resource-dir" "/autofs/nccs-svm1_home1/lld/software/sollve/lib/clang/9.0.0" "-O2" "-fdebug-compilation-dir" "/ccs/home/lld/benchmarks/rodinia_3.1/openmp-summit/backprop" "-ferror-limit" "19" "-fmessage-length" "0" "-fopenmp" "-fno-signed-char" "-fobjc-runtime=gcc" "-fdiagnostics-show-option" "-vectorize-loops" "-vectorize-slp" "-o" host2.bc "-x" "ir" host.bc "-fopenmp-targets=nvptx64-nvidia-cuda" "-faddrsig"
	clang -cc1 -DOMP_GPU_OFFLOAD -fopenmp -triple nvptx64-nvidia-cuda -fopenmp-targets=nvptx64-nvidia-cuda -target-cpu sm_70 -I. -fopenmp-is-device -fopenmp-host-ir-file-path host2.bc dev.bc -S -o dev.s
	ptxas "-m64" "-O2" "--gpu-name" "sm_70" "--output-file" "dev.cubin" dev.s "-c"
	nvlink "-o" dev.out "-arch" "sm_70" "-L/ccs/home/lld/software/sollve/libexec" "-L/ccs/home/lld/software/sollve/lib" "-L/autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/gcc-6.4.0/spectrum-mpi-10.2.0.11-20190201-6p364zmj3lab7ezyc4prkmm4gogxf5om/lib" "-L/sw/summit/cuda/9.2.148/lib64" "-L/autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/gcc-4.8.5/darshan-runtime-3.1.7-uwak6exr43zvhu2vasiddrov57baf37t/lib" "-L/autofs/nccs-svm1_home1/lld/software/sollve/lib" "-lomptarget-nvptx" dev.cubin 
	clang "-cc1" "-triple" "powerpc64le-unknown-linux-gnu" "-emit-obj" "-disable-free" "-disable-llvm-verifier" "-discard-value-names" "-main-file-name" main.bc "-mrelocation-model" "static" "-mthread-model" "posix" "-fmath-errno" "-masm-verbose" "-mconstructor-aliases" "-fuse-init-array" "-target-cpu" "ppc64le" "-mfloat-abi" "hard" "-target-abi" "elfv2" "-dwarf-column-info" "-debugger-tuning=gdb" "-momit-leaf-frame-pointer" "-resource-dir" "/autofs/nccs-svm1_home1/lld/software/sollve/lib/clang/9.0.0" "-O2" "-fdebug-compilation-dir" "/ccs/home/lld/benchmarks/rodinia_3.1/openmp-summit/backprop" "-ferror-limit" "19" "-fmessage-length" "0" "-fopenmp" "-fno-signed-char" "-fobjc-runtime=gcc" "-fdiagnostics-show-option" "-vectorize-loops" "-vectorize-slp" "-o" "host.o" "-x" "ir" host.bc "-fopenmp-targets=nvptx64-nvidia-cuda" "-faddrsig"
	ld "--hash-style=gnu" "--no-add-needed" "--eh-frame-hdr" "-m" "elf64lppc" "-dynamic-linker" "/lib64/ld64.so.2" "-o" backprop_gpu "/usr/lib/gcc/ppc64le-redhat-linux/4.8.5/../../../../lib64/crt1.o" "/usr/lib/gcc/ppc64le-redhat-linux/4.8.5/../../../../lib64/crti.o" "/usr/lib/gcc/ppc64le-redhat-linux/4.8.5/crtbegin.o" "-L/usr/lib/gcc/ppc64le-redhat-linux/4.8.5" "-L/usr/lib/gcc/ppc64le-redhat-linux/4.8.5/../../../../lib64" "-L/lib/../lib64" "-L/usr/lib/../lib64" "-L/usr/lib/gcc/ppc64le-redhat-linux/4.8.5/../../.." "-L/autofs/nccs-svm1_home1/lld/software/sollve/bin/../lib" "-L/lib" "-L/usr/lib" "host.o" "-L/ccs/home/lld/software/sollve/libexec" "-L/ccs/home/lld/software/sollve/lib" "-L/autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/gcc-6.4.0/spectrum-mpi-10.2.0.11-20190201-6p364zmj3lab7ezyc4prkmm4gogxf5om/lib" "-L/sw/summit/cuda/9.2.148/lib64" "-L/autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/gcc-4.8.5/darshan-runtime-3.1.7-uwak6exr43zvhu2vasiddrov57baf37t/lib" "-lomp" "-lomptarget" "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed" "-lpthread" "-lc" "-lgcc" "--as-needed" "-lgcc_s" "--no-as-needed" "/usr/lib/gcc/ppc64le-redhat-linux/4.8.5/crtend.o" "/usr/lib/gcc/ppc64le-redhat-linux/4.8.5/../../../../lib64/crtn.o" "-T" "a.lk"

facetrain.all: facetrain.c backprop.h
	$(CC) $(ALL_FLAGS) facetrain.c -c -o facetrain.bc

backprop.all: backprop.c backprop.h
	$(CC) $(ALL_FLAGS) backprop.c -c -o backprop.bc

backprop_kernel.all: backprop_kernel.c backprop.h
	$(CC) $(ALL_FLAGS) backprop_kernel.c -c -o backprop_kernel.bc

imagenet.all: imagenet.c backprop.h
	$(CC) $(ALL_FLAGS) imagenet.c -c -o imagenet.bc

gpu2: facetrain.all backprop.all backprop_kernel.all imagenet.all
	llvm-link facetrain-host-powerpc64le-unknown-linux-gnu.tmp.bc backprop-host-powerpc64le-unknown-linux-gnu.tmp.bc backprop_kernel-host-powerpc64le-unknown-linux-gnu.tmp.bc imagenet-host-powerpc64le-unknown-linux-gnu.tmp.bc -o host.bc
	llvm-link facetrain-openmp-nvptx64-nvidia-cuda.tmp.bc backprop-openmp-nvptx64-nvidia-cuda.tmp.bc backprop_kernel-openmp-nvptx64-nvidia-cuda.tmp.bc imagenet-openmp-nvptx64-nvidia-cuda.tmp.bc -o dev.bc
	clang-offload-bundler -type=bc -inputs=host.bc,dev.bc -targets=host-powerpc64le-unknown-linux-gnu,openmp-nvptx64-nvidia-cuda -outputs=/ccs/home/lld/benchmarks/rodinia_3.1/openmp/backprop/main.bc
	$(CC) $(CC_FLAGS) main.bc -o backprop_gpu -lm 

clean:
	rm -f *.o *~ backprop backprop_gpu backprop_gpu_um backprop_cuda.linkinfo *.ll *.bc
