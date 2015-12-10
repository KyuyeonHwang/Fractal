## Makefile

.PHONY: clean realclean

OUTNAME_BIN=langModel

FLAG_ENABLE_OMP=0
FLAG_ENABLE_CUDA=1
FLAG_ENABLE_ATLAS=0

CUDA_PATH=/usr/local/cuda
GPU_ARCH=-gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35

OPTFLAGS=-m64 -Ofast -flto -march=native -funroll-loops
CPPFLAGS=-Wall -std=c++11
NVCCFLAGS=-m64 -O3 -arch=$(GPU_ARCH)

LDFLAGS=-Wl,-rpath $(shell pwd)/../../../build/lib

BUILDDIR_BIN=../bin
OBJDIR=../obj
SRCDIR=./
SRCDIR_HDRS=./
SRCDIR_CUDA=$(SRCDIR)


LIBS=-lfractal
INCDIR=../../../build/include
LIBDIR=../../../build/lib


#########################################################################################

ifeq ($(FLAG_ENABLE_OMP),1) 
  DEFS+=-D__ENABLE_OMP
  CPPFLAGS+=-fopenmp
  LDFLAGS+=-fopenmp
endif

ifeq ($(FLAG_ENABLE_ATLAS),1) 
  DEFS+=-D__ENABLE_ATLAS
  LIBS+=-lcblas -latlas
endif


ifeq ($(FLAG_ENABLE_CUDA),1) 
  INCDIR+=$(CUDA_PATH)/include
  LIBDIR+=$(CUDA_PATH)/lib64
  LIBS+=-lcuda 
  DEFS+=-D__ENABLE_CUDA
  LDFLAGS+=-Wl,-rpath $(CUDA_PATH)/lib64
  NVCC=nvcc
  OBJDIR_CUDA=$(OBJDIR)/cuda
  OBJS=$(patsubst $(SRCDIR_CUDA)/%.cu,$(OBJDIR_CUDA)/%.o, $(wildcard $(SRCDIR_CUDA)/**/*.cu))
endif


INCLUDES+=$(patsubst %,-I%,$(INCDIR))
LDFLAGS+=$(patsubst %,-L%,$(LIBDIR))

CPPFLAGS+=$(OPTFLAGS)
LDFLAGS+=$(OPTFLAGS)

HDRS=$(wildcard $(SRCDIR_HDRS)/**/*.h)
HDRS+=$(wildcard $(SRCDIR_HDRS)/*.h)
HDRS+=$(wildcard $(SRCDIR_HDRS)/**/*.hxx)
HDRS+=$(wildcard $(SRCDIR_HDRS)/*.hxx)
OBJS+=$(patsubst $(SRCDIR)/%.cc,$(OBJDIR)/%.o, $(wildcard $(SRCDIR)/**/*.cc))
OBJS+=$(patsubst $(SRCDIR)/%.cc,$(OBJDIR)/%.o, $(wildcard $(SRCDIR)/*.cc))

CC=g++

TARGET_BIN=$(BUILDDIR_BIN)/$(OUTNAME_BIN)


all:$(TARGET_BIN)


$(TARGET_BIN):$(OBJS) 
	@mkdir -p $(@D)
	$(CC) -o $(TARGET_BIN)    $(LDFLAGS) $(OBJS) $(LIBS)

# dependencies
$(OBJDIR)/%.o:$(SRCDIR)/%.cc $(HDRS)
	@mkdir -p $(@D)
	$(CC) -o $@    $(DEFS) $(CPPFLAGS) $(INCLUDES) -c $<

$(OBJDIR_CUDA)/%.o:$(SRCDIR_CUDA)/%.cu $(HDRS)
	@mkdir -p $(@D)
	$(NVCC) -o $@    $(DEFS) $(NVCCFLAGS) $(INCLUDES) -c $<

## other options
clean:
	rm -rf $(OBJS)

realclean:
	rm -rf $(OBJDIR) $(TARGET_BIN) 

