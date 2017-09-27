include ../../common/make.config.summitdev

CC := $(CUDA_DIR)/bin/nvcc

INCLUDE := $(CUDA_DIR)/include

SRC = bfs_linput.cu

EXE = bfs

release: $(SRC)
	$(CC) -arch=sm_60 $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR)  -I$(SDK_DIR)/common/inc  -L$(SDK_DIR)/lib

enum: $(SRC)
	$(CC) -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

debug: $(SRC)
	$(CC) -g $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

debugenum: $(SRC)
	$(CC) -g -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result.txt
