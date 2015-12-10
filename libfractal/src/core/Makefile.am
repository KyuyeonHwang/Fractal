noinst_LTLIBRARIES = libcore.la

libcore_la_SOURCES = Connection.cc \
                     Engine.cc \
                     Layer.cc \
                     Matrix.cc \
                     Mem.cc \
                     MultiTypeMatrix.cc \
                     Probe.cc \
                     Rnn.cc \
                     TrainableProbe.cc \
                     CudaKernels.cu

includesubdir = $(includedir)/fractal/core

includesub_HEADERS = FractalCommon.h \
                     Connection.h \
                     Engine.h \
                     InitWeightParam.h \
                     Layer.h \
                     Matrix.h \
                     Mem.h \
                     MultiTypeMatrix.h \
                     Probe.h \
                     Rnn.h \
                     TrainableProbe.h \
                     CudaKernels.h


CLEANFILES = nvcc_wrapper.sh

.cu.lo:
	@echo '#!/bin/bash' > nvcc_wrapper.sh
	@echo 'ARGS=`echo $$@ | sed "s/-W/-Xcompiler -W/g"`' >> nvcc_wrapper.sh
	@echo 'ARGS=`echo $$ARGS | sed "s/-f/-Xcompiler -f/g"`' >> nvcc_wrapper.sh
	@echo 'nvcc $$ARGS' >> nvcc_wrapper.sh
	$(LIBTOOL) --tag=CXX --mode=compile /bin/bash nvcc_wrapper.sh -c $(NVCCFLAGS) -o $@ $<
	@rm nvcc_wrapper.sh

