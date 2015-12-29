/*
   Copyright 2015 Kyuyeon Hwang (kyuyeon.hwang@gmail.com)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#include "Engine.h"

#ifdef FRACTAL_USE_CUDA

#include <random>
#include "CudaKernels.h"

#define CUDA_CHUNK_SIZE (2 * sizeof(FLOAT)) /* In bytes. curandGenerateNormal requires even number of elements */

#ifdef FRACTAL_DOUBLE_PRECISION
    #define GEAM cublasDgeam
    #define GEMV cublasDgemv
    #define GEMM cublasDgemm
    #define AXPY cublasDaxpy
    #define COPY cublasDcopy
    #define RANDN curandGenerateNormalDouble
    #define RANDU curandGenerateUniformDouble
#elif defined(FRACTAL_SINGLE_PRECISION)
    #define GEAM cublasSgeam
    #define GEMV cublasSgemv
    #define GEMM cublasSgemm
    #define AXPY cublasSaxpy
    #define COPY cublasScopy
    #define RANDN curandGenerateNormal
    #define RANDU curandGenerateUniform
#endif /* FRACTAL_DOUBLE_PRECISION */

#else /* FRACTAL_USE_CUDA */

#include <cstdlib>
#include <cstring>

#endif /* FRACTAL_USE_CUDA */


#ifdef FRACTAL_USE_ATLAS
extern "C"
{
#include <cblas.h>
}
#endif /* FRACTAL_USE_ATLAS */


namespace fractal
{


Engine::Engine()
{
    memCount = 0;
    memAllocCount = 0;
    hostLoc = 0;
    eventCount = 0;
    streamCount = 0;
    currentLoc = 0;

#ifdef FRACTAL_USE_CUDA
    int n = 0;

    verify(cudaGetDeviceCount(&n) == cudaSuccess);

    numLoc = n + 1;

    /* Initialize CUBLAS and CURAND */

    cublasHandle.resize(n);
    curandGen.resize(n);

    for(unsigned long i = 0; i < (unsigned long) n; i++)
    {
        verify(cudaSetDevice(i) == cudaSuccess);
        verify(cublasCreate(&cublasHandle[i]) == CUBLAS_STATUS_SUCCESS);
        verify(curandCreateGenerator(&curandGen[i], CURAND_RNG_PSEUDO_DEFAULT) == CURAND_STATUS_SUCCESS);
        verify(curandSetPseudoRandomGeneratorSeed(curandGen[i], i) == CURAND_STATUS_SUCCESS);
    }


    /* Enable peer access */

    for(unsigned long i = 0; i < (unsigned long) n; i++)
    {
        for(unsigned long j = 0; j < (unsigned long) n; j++)
        {
            if(i == j) continue;

            int accessible = 0;
            verify(cudaDeviceCanAccessPeer(&accessible, i, j) == cudaSuccess);

            if(accessible != 0)
            {
                verify(cudaSetDevice(i) == cudaSuccess);
                verify(cudaDeviceEnablePeerAccess(j, 0) == cudaSuccess);
            }
        }
    }
#else
    numLoc = 1;
#endif /* FRACTAL_USE_CUDA */
}


Engine::~Engine()
{
    verify(memCount == 0);
    verify(memAllocCount == 0);
    verify(eventCount == 0);
    verify(streamCount == 0);

#ifdef FRACTAL_USE_CUDA

    for(unsigned long i = 0; i < GetNumComputeLocs(); i++)
    {
        verify(cudaSetDevice(i) == cudaSuccess);
        verify(cudaDeviceSynchronize() == cudaSuccess);
        verify(curandDestroyGenerator(curandGen[i]) == CURAND_STATUS_SUCCESS);
        verify(cublasDestroy(cublasHandle[i]) == CUBLAS_STATUS_SUCCESS);
    }

    verify(cudaGetLastError() == cudaSuccess);
#endif /* FRACTAL_USE_CUDA */
}


void Engine::MemAdd(Mem *mem)
{
    mtxMem.lock();

    verify(mem->GetEngine() == this);

    memCount++;

    mtxMem.unlock();
}


void Engine::MemDel(Mem *mem)
{
    mtxMem.lock();

    verify(mem->GetEngine() == this);

    memCount--;

    mtxMem.unlock();
}


void Engine::MemAlloc(Mem *mem, unsigned long loc)
{
    mtxMem.lock();
    mem->Lock();

    verify(mem->GetEngine() == this);

    size_t size;
    void **ptr;

    ptr = mem->GetPtrs();
    if(ptr[loc] != NULL)
    {
        mtxMem.unlock();
        mem->Unlock();
        return;
    }

    size = mem->GetSize();
    verify(size > 0);


#ifdef FRACTAL_USE_CUDA
    if(loc == hostLoc)
    {
        verify(cudaMallocHost(ptr + loc, size) == cudaSuccess);
    }
    else
    {
        SetComputeLoc(loc);
        size = (size + CUDA_CHUNK_SIZE - 1) / CUDA_CHUNK_SIZE * CUDA_CHUNK_SIZE;
        verify(cudaMalloc(ptr + loc, size) == cudaSuccess);
    }
#else
    verify((ptr[loc] = malloc(size)) != NULL);
#endif /* FRACTAL_USE_CUDA */

    memAllocCount++;

    mtxMem.unlock();
    mem->Unlock();
}


void Engine::MemDealloc(Mem *mem)
{
    mtxMem.lock();
    mem->Lock();

    verify(mem->GetEngine() == this);

    void **ptr;

    ptr = mem->GetPtrs();

#ifdef FRACTAL_USE_CUDA
    for(unsigned long i = 0; i < numLoc; i++)
    {
        if(ptr[i] != NULL)
        {
            if(i == hostLoc)
            {
                verify(cudaFreeHost(ptr[i]) == cudaSuccess);
            }
            else
            {
                verify(cudaFree(ptr[i]) == cudaSuccess);
            }

            ptr[i] = NULL;
            memAllocCount--;
        }
    }
#else
    for(unsigned long i = 0; i < numLoc; i++)
    {
        if(ptr[i] != NULL)
        {
            free(mem->GetPtr(0));

            ptr[i] = NULL;
            memAllocCount--;
        }
    }
#endif /* FRACTAL_USE_CUDA */

    mem->Invalidate();

    mtxMem.unlock();
    mem->Unlock();
}


void Engine::MemPull(Mem *mem, const unsigned long loc, PStream &stream)
{
    /* TODO: Do not copy the entire mem
     * If mem is sub, do not validate, copy parital data */
    /* DO NOT link data between GPUs */
    mtxMem.lock();
    mem->Lock();

    verify(mem->GetEngine() == this);

    unsigned long recent;

    if(mem->IsValid(loc) == true)
    {
        mtxMem.unlock();
        mem->Unlock();
        return;
    }

    recent = mem->GetRecentLoc();

    //verify(mem->IsValid(recent) == true);

    MemAlloc(mem, loc);
    if(mem->IsValid(recent) == true)
        MemCopy(mem, 0, recent, mem, 0, loc, mem->GetSize(), stream);

    mem->Validate(loc);

    mtxMem.unlock();
    mem->Unlock();
}


void Engine::MemCopy(const Mem *memSrc, const size_t offsetSrc, Mem *memDst, const size_t offsetDst, const size_t size, PStream &stream)
{
    verify(memSrc != memDst);
    verify(memSrc->GetEngine() == this);
    verify(memDst->GetEngine() == this);

    unsigned long locSrc, locDst;

    locSrc = memSrc->GetRecentLoc();
    locDst = memDst->GetRecentLoc();
    verify(memSrc->IsValid(locSrc) == true);

    //if(memDst->GetSize() != size) /* Partial copy */
    //    memDst->Pull(locDst);
    //else
    MemAlloc(memDst, locDst);

    MemCopy(memSrc, offsetSrc, locSrc, memDst, offsetDst, locDst, size, stream);

    memDst->Push(locDst);
}


void Engine::MemCopyFromHost(Mem *memDst, const size_t offsetDst, const void *ptrSrc, const size_t size, PStream &stream)
{
    verify(memDst->GetEngine() == this);

    unsigned long locSrc, locDst;
    void *ptrDst;

    locSrc = GetHostLoc();

    locDst = memDst->GetRecentLoc();
    MemAlloc(memDst, locDst);

    ptrDst = (unsigned char *)memDst->GetPtr(locDst) + offsetDst;
    MemCopy(ptrSrc, locSrc, ptrDst, locDst, size, stream);

    memDst->Push(locDst);
}


void Engine::MemCopyToHost(const Mem *memSrc, const size_t offsetSrc, void *ptrDst, const size_t size, PStream &stream)
{
    verify(memSrc->GetEngine() == this);

    unsigned long locSrc, locDst;
    void *ptrSrc;

    locDst = GetHostLoc();

    locSrc = memSrc->GetRecentLoc();
    verify(memSrc->IsValid(locSrc) == true);

    ptrSrc = (unsigned char *)memSrc->GetPtr(locSrc) + offsetSrc;
    MemCopy(ptrSrc, locSrc, ptrDst, locDst, size, stream);
}


void Engine::MemCopy(const Mem *memSrc, const size_t offsetSrc, const unsigned long locSrc,
        Mem *memDst, const size_t offsetDst, const unsigned long locDst, const size_t size, PStream &stream)
{
/*
    if(memSrc == memDst && locSrc == locDst)
    {
        verify(offsetSrc == offsetDst);
        return;
    }
*/
    if(memSrc == memDst && locSrc == locDst && offsetSrc == offsetDst) return;

    void *ptrSrc, *ptrDst;

    ptrSrc = (unsigned char *)memSrc->GetPtr(locSrc) + offsetSrc;
    ptrDst = (unsigned char *)memDst->GetPtr(locDst) + offsetDst;

    MemCopy(ptrSrc, locSrc, ptrDst, locDst, size, stream);
}


void Engine::MemCopy(const void *ptrSrc, const unsigned long locSrc,
        void *ptrDst, const unsigned long locDst, const size_t size, PStream &stream)
{
    if(ptrSrc == ptrDst) return;;

#ifdef FRACTAL_USE_CUDA
    if(stream.loc == hostLoc)
    {
        if(locSrc != hostLoc) SetComputeLoc(locSrc);
        if(locDst != hostLoc) SetComputeLoc(locDst);

        verify(cudaMemcpy(ptrDst, ptrSrc, size, cudaMemcpyDefault) == cudaSuccess);
    }
    else
    {
        SetComputeLoc(stream.loc);
        verify(cudaMemcpyAsync(ptrDst, ptrSrc, size, cudaMemcpyDefault, stream.cudaStream) == cudaSuccess);
    }
#else
    verify(stream.loc == hostLoc);
    memcpy(ptrDst, ptrSrc, size);
#endif /* FRACTAL_USE_CUDA */
}


void Engine::MatMult(Matrix<FLOAT> &A, const bool transA, Matrix<FLOAT> &B, const bool transB, Matrix<FLOAT> &C, const FLOAT alpha, const FLOAT beta, PStream &stream)
{
    verify(A.GetEngine() == this);
    verify(B.GetEngine() == this);
    verify(C.GetEngine() == this);

    verify((transA == false ? A.GetNumCols() : A.GetNumRows()) == (transB == false ? B.GetNumRows() : B.GetNumCols()));
    verify(C.GetNumRows() == (transA == false ? A.GetNumRows() : A.GetNumCols()));
    verify(C.GetNumCols() == (transB == false ? B.GetNumCols() : B.GetNumRows()));

    FLOAT *ptrA, *ptrB, *ptrC;

    ptrA = A.GetPtrForReadWrite(stream);
    ptrB = B.GetPtrForReadWrite(stream);

    if(beta == (FLOAT) 0)
        ptrC = C.GetPtrForWrite(stream);
    else
        ptrC = C.GetPtrForReadWrite(stream);


#ifdef FRACTAL_USE_CUDA
    SetComputeLoc(stream.loc);
    verify(cublasSetStream(cublasHandle[GetComputeLocIdx(stream.loc)], stream.cudaStream) == CUBLAS_STATUS_SUCCESS);

    if(C.GetNumCols() == 1) /* If C is a vector */
    {
        verify(GEMV(cublasHandle[GetComputeLocIdx(stream.loc)],
                    transA == true ? CUBLAS_OP_T : CUBLAS_OP_N,
                    A.GetNumRows(),
                    A.GetNumCols(),
                    &alpha,
                    ptrA,
                    A.GetLeadingDim(),
                    ptrB,
                    1,
                    &beta,
                    ptrC,
                    1)
                == CUBLAS_STATUS_SUCCESS);
    }
    else
    {
        verify(GEMM(cublasHandle[GetComputeLocIdx(stream.loc)],
                    transA == true ? CUBLAS_OP_T : CUBLAS_OP_N,
                    transB == true ? CUBLAS_OP_T : CUBLAS_OP_N,
                    C.GetNumRows(),
                    C.GetNumCols(),
                    transA == true ? A.GetNumRows() : A.GetNumCols(),
                    &alpha,
                    ptrA,
                    A.GetLeadingDim(),
                    ptrB,
                    B.GetLeadingDim(),
                    &beta,
                    ptrC,
                    C.GetLeadingDim())
                == CUBLAS_STATUS_SUCCESS);
    }
#else
    verify(false); /* CPU computation is not supported */
#endif /* FRACTAL_USE_CUDA */

    C.FinishWrite(stream);
}


void Engine::MatElemMult(Matrix<FLOAT> &A, Matrix<FLOAT> &B, Matrix<FLOAT> &C, PStream &stream)
{
    verify(A.GetEngine() == this);
    verify(B.GetEngine() == this);
    verify(C.GetEngine() == this);
    verify(A.GetNumRows() == B.GetNumRows());
    verify(A.GetNumCols() == B.GetNumCols());
    verify(A.GetNumRows() == C.GetNumRows());
    verify(A.GetNumCols() == C.GetNumCols());

    FLOAT *ptrA, *ptrB, *ptrC;

    ptrA = A.GetPtrForReadWrite(stream);
    ptrB = B.GetPtrForReadWrite(stream);
    ptrC = C.GetPtrForWrite(stream);

#ifdef FRACTAL_USE_CUDA
    SetComputeLoc(stream.loc);
    cudaKernels::MatElemMult<FLOAT>(ptrA, A.GetLeadingDim(),
            ptrB, B.GetLeadingDim(),
            ptrC, C.GetLeadingDim(),
            A.GetNumRows(), B.GetNumCols(),
            stream.cudaStream);
#else
    verify(false); /* CPU computation is not supported */
#endif /* FRACTAL_USE_CUDA */

    C.FinishWrite(stream);
}


void Engine::MatAdd(Matrix<FLOAT> &A, Matrix<FLOAT> &B, const FLOAT alpha, PStream &stream)
{
    verify(A.GetEngine() == this);
    verify(B.GetEngine() == this);
    verify(A.GetNumRows() == B.GetNumRows());
    verify(A.GetNumCols() == B.GetNumCols());

    verify(A.GetNumRows() == A.GetLeadingDim());
    verify(B.GetNumRows() == B.GetLeadingDim());

    FLOAT *ptrA, *ptrB;

    ptrA = A.GetPtrForReadWrite(stream);
    ptrB = B.GetPtrForReadWrite(stream);

#ifdef FRACTAL_USE_CUDA
    SetComputeLoc(stream.loc);
    verify(cublasSetStream(cublasHandle[GetComputeLocIdx(stream.loc)], stream.cudaStream) == CUBLAS_STATUS_SUCCESS);

    verify(AXPY(cublasHandle[GetComputeLocIdx(stream.loc)],
                A.GetNumRows() * A.GetNumCols(),
                &alpha,
                ptrA,
                1,
                ptrB,
                1)
            == CUBLAS_STATUS_SUCCESS);
#else
    verify(false); /* CPU computation is not supported */
#endif /* FRACTAL_USE_CUDA */

    B.FinishWrite(stream);
}


void Engine::MatAdd(Matrix<FLOAT> &A, Matrix<FLOAT> &B, Matrix<FLOAT> &C, PStream &stream)
{
    verify(A.GetEngine() == this);
    verify(B.GetEngine() == this);
    verify(C.GetEngine() == this);
    verify(A.GetNumRows() == B.GetNumRows());
    verify(A.GetNumCols() == B.GetNumCols());
    verify(A.GetNumRows() == C.GetNumRows());
    verify(A.GetNumCols() == C.GetNumCols());

    FLOAT *ptrA, *ptrB, *ptrC;

    ptrA = A.GetPtrForReadWrite(stream);
    ptrB = B.GetPtrForReadWrite(stream);
    ptrC = C.GetPtrForWrite(stream);

#ifdef FRACTAL_USE_CUDA
    SetComputeLoc(stream.loc);
    cudaKernels::MatAdd<FLOAT>(ptrA, A.GetLeadingDim(),
            ptrB, B.GetLeadingDim(),
            ptrC, C.GetLeadingDim(),
            A.GetNumRows(), B.GetNumCols(),
            stream.cudaStream);
#else
    verify(false); /* CPU computation is not supported */
#endif /* FRACTAL_USE_CUDA */

    C.FinishWrite(stream);
}


void Engine::MatSub(Matrix<FLOAT> &A, Matrix<FLOAT> &B, Matrix<FLOAT> &C, PStream &stream)
{
    verify(A.GetEngine() == this);
    verify(B.GetEngine() == this);
    verify(C.GetEngine() == this);
    verify(A.GetNumRows() == B.GetNumRows());
    verify(A.GetNumCols() == B.GetNumCols());
    verify(A.GetNumRows() == C.GetNumRows());
    verify(A.GetNumCols() == C.GetNumCols());

    FLOAT *ptrA, *ptrB, *ptrC;

    ptrA = A.GetPtrForReadWrite(stream);
    ptrB = B.GetPtrForReadWrite(stream);
    ptrC = C.GetPtrForWrite(stream);

#ifdef FRACTAL_USE_CUDA
    SetComputeLoc(stream.loc);
    cudaKernels::MatSub<FLOAT>(ptrA, A.GetLeadingDim(),
            ptrB, B.GetLeadingDim(),
            ptrC, C.GetLeadingDim(),
            A.GetNumRows(), B.GetNumCols(),
            stream.cudaStream);
#else
    verify(false); /* CPU computation is not supported */
#endif /* FRACTAL_USE_CUDA */

    C.FinishWrite(stream);
}


void Engine::MatAddToDiag(Matrix<FLOAT> &mat, const FLOAT val, const unsigned long shift, PStream &stream)
{
    verify(mat.GetEngine() == this);
    verify(mat.GetNumRows() == mat.GetNumCols());

    verify(mat.GetNumRows() == mat.GetLeadingDim());

    FLOAT *ptr;

    ptr = mat.GetPtrForReadWrite(stream);

#ifdef FRACTAL_USE_CUDA
    SetComputeLoc(stream.loc);
    cudaKernels::MatAddToDiag<FLOAT>(ptr, val, shift, mat.GetNumRows(), stream.cudaStream);
#else
    verify(false); /* CPU computation is not supported */
#endif /* FRACTAL_USE_CUDA */

    mat.FinishWrite(stream);
}


void Engine::MatMakeTri(Matrix<FLOAT> &mat, const bool upper, PStream &stream)
{
    verify(mat.GetEngine() == this);
    verify(mat.GetNumRows() == mat.GetNumCols());

    verify(mat.GetNumRows() == mat.GetLeadingDim());

    FLOAT *ptr;

    ptr = mat.GetPtrForReadWrite(stream);

#ifdef FRACTAL_USE_CUDA
    SetComputeLoc(stream.loc);
    cudaKernels::MatMakeTri<FLOAT>(ptr, upper, mat.GetNumRows(), stream.cudaStream);
#else
    verify(false); /* CPU computation is not supported */
#endif /* FRACTAL_USE_CUDA */

    mat.FinishWrite(stream);
}


void Engine::MatSet(Matrix<FLOAT> &mat, const FLOAT val, PStream &stream)
{
    verify(mat.GetEngine() == this);

    FLOAT *ptr;

    ptr = mat.GetPtrForWrite(stream);

#ifdef FRACTAL_USE_CUDA
    SetComputeLoc(stream.loc);
    cudaKernels::MatSet<FLOAT>(ptr, mat.GetLeadingDim(), val,
            mat.GetNumRows(), mat.GetNumCols(), stream.cudaStream);
#else
    verify(false); /* CPU computation is not supported */
#endif /* FRACTAL_USE_CUDA */

    mat.FinishWrite(stream);
}


void Engine::MatSet(Matrix<INT> &mat, const INT val, PStream &stream)
{
    verify(mat.GetEngine() == this);

    INT *ptr;

    ptr = mat.GetPtrForWrite(stream);

#ifdef FRACTAL_USE_CUDA
    SetComputeLoc(stream.loc);
    cudaKernels::MatSet<INT>(ptr, mat.GetLeadingDim(), val,
            mat.GetNumRows(), mat.GetNumCols(), stream.cudaStream);
#else
    verify(false); /* CPU computation is not supported */
#endif /* FRACTAL_USE_CUDA */

    mat.FinishWrite(stream);
}


void Engine::MatRandN(Matrix<FLOAT> &mat, const FLOAT mean, const FLOAT stdev, PStream &stream)
{
    verify(mat.GetEngine() == this);
    verify(stdev >= (FLOAT) 0);
    verify(mat.GetMem()->GetSize() == sizeof(FLOAT) * mat.GetNumCols() * mat.GetNumRows());

    verify(mat.GetNumRows() == mat.GetLeadingDim());

    FLOAT *ptr;

    ptr = mat.GetPtrForWrite(stream);

#ifdef FRACTAL_USE_CUDA
    /* curandGenerateNormal requires even number of elements */
    unsigned long n;
    verify(CUDA_CHUNK_SIZE >= 2 * sizeof(FLOAT));
    n = mat.GetNumRows() * mat.GetNumCols();
    n = (n + 1) / 2 * 2;

    SetComputeLoc(stream.loc);
    verify(curandSetStream(curandGen[GetComputeLocIdx(stream.loc)], stream.cudaStream) == CURAND_STATUS_SUCCESS);
    verify(RANDN(curandGen[GetComputeLocIdx(stream.loc)], ptr, n, mean, stdev) == CURAND_STATUS_SUCCESS);
#else
    verify(false); /* CPU computation is not supported */
#endif /* FRACTAL_USE_CUDA */

    mat.FinishWrite(stream);
}


void Engine::MatCopy(Matrix<FLOAT> &A, Matrix<FLOAT> &B, PStream &stream)
{
    verify(A.GetEngine() == this);
    verify(B.GetEngine() == this);
    verify(A.GetNumRows() == B.GetNumRows());
    verify(A.GetNumCols() == B.GetNumCols());

    verify(A.GetNumRows() == A.GetLeadingDim());
    verify(B.GetNumRows() == B.GetLeadingDim());

    unsigned long locSrc, locDst;
    size_t offsetSrc, offsetDst, size;
    Mem *memSrc, *memDst;

    B.GetPtrForWrite(stream);

    memSrc = A.GetMem();
    memDst = B.GetMem();

    offsetSrc = A.GetOffset() * sizeof(FLOAT);
    offsetDst = B.GetOffset() * sizeof(FLOAT);
    size = A.GetNumRows() * A.GetNumCols() * sizeof(FLOAT);

    locSrc = memSrc->GetRecentLoc();
    locDst = stream.loc;

    verify(memSrc->IsValid(locSrc) == true);

    MemAlloc(memDst, locDst);

    MemCopy(memSrc, offsetSrc, locSrc, memDst, offsetDst, locDst, size, stream);

    memDst->Push(locDst);
#ifdef FRACTAL_USE_CUDA
    /* inefficient */
    /*
    //FLOAT *ptrA, *ptrB;

    ptrA = A.GetPtrForReadWrite(stream);
    ptrB = B.GetPtrForWrite(stream);
    
    SetComputeLoc(stream.loc);
    verify(cublasSetStream(cublasHandle[GetComputeLocIdx(stream.loc)], stream.cudaStream) == CUBLAS_STATUS_SUCCESS);
    */
    /*
    verify(COPY(cublasHandle[GetComputeLocIdx(stream.loc)],
                A.GetNumRows() * A.GetNumCols(),
                ptrA,
                1,
                ptrB,
                1)
            == CUBLAS_STATUS_SUCCESS);
    */
#else
    verify(false); /* CPU computation is not supported */
#endif /* FRACTAL_USE_CUDA */

    B.FinishWrite(stream);
}


void Engine::MatTranspose(Matrix<FLOAT> &A, Matrix<FLOAT> &B, PStream &stream)
{
    verify(A.GetEngine() == this);
    verify(B.GetEngine() == this);
    verify(A.GetNumRows() == B.GetNumCols());
    verify(A.GetNumCols() == B.GetNumRows());

    verify(A.GetNumRows() == A.GetLeadingDim());
    verify(B.GetNumRows() == B.GetLeadingDim());

    FLOAT *ptrA, *ptrB;

    ptrA = A.GetPtrForReadWrite(stream);
    ptrB = B.GetPtrForWrite(stream);

#ifdef FRACTAL_USE_CUDA
    FLOAT alpha = (FLOAT) 1;
    FLOAT beta = (FLOAT) 0;

    SetComputeLoc(stream.loc);
    verify(cublasSetStream(cublasHandle[GetComputeLocIdx(stream.loc)], stream.cudaStream) == CUBLAS_STATUS_SUCCESS);
    verify(GEAM(cublasHandle[GetComputeLocIdx(stream.loc)],
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                B.GetNumRows(),
                B.GetNumCols(),
                &alpha,
                ptrA,
                A.GetNumRows(),
                &beta,
                ptrB,
                B.GetNumRows(),
                ptrB,
                B.GetNumRows())
            == CUBLAS_STATUS_SUCCESS);
#else
    verify(false); /* CPU computation is not supported */
#endif /* FRACTAL_USE_CUDA */

    B.FinishWrite(stream);
}


void Engine::FuncSigmoid(Matrix<FLOAT> &X, Matrix<FLOAT> &Y, PStream &stream)
{
    verify(X.GetEngine() == this);
    verify(Y.GetEngine() == this);
    verify(X.GetNumRows() == Y.GetNumRows());
    verify(X.GetNumCols() == Y.GetNumCols());

    FLOAT *ptrX, *ptrY;

    ptrX = X.GetPtrForReadWrite(stream);
    ptrY = Y.GetPtrForWrite(stream);

#ifdef FRACTAL_USE_CUDA
    SetComputeLoc(stream.loc);
    cudaKernels::FuncSigmoid(ptrX, X.GetLeadingDim(),
            ptrY, Y.GetLeadingDim(),
            Y.GetNumRows(), Y.GetNumCols(), stream.cudaStream);
#else
    verify(false); /* CPU computation is not supported */
#endif /* FRACTAL_USE_CUDA */

    Y.FinishWrite(stream);
}


void Engine::FuncTanh(Matrix<FLOAT> &X, Matrix<FLOAT> &Y, PStream &stream)
{
    verify(X.GetEngine() == this);
    verify(Y.GetEngine() == this);
    verify(X.GetNumRows() == Y.GetNumRows());
    verify(X.GetNumCols() == Y.GetNumCols());

    FLOAT *ptrX, *ptrY;

    ptrX = X.GetPtrForReadWrite(stream);
    ptrY = Y.GetPtrForWrite(stream);

#ifdef FRACTAL_USE_CUDA
    SetComputeLoc(stream.loc);
    cudaKernels::FuncTanh(ptrX, X.GetLeadingDim(),
            ptrY, Y.GetLeadingDim(),
            Y.GetNumRows(), Y.GetNumCols(), stream.cudaStream);
#else
    verify(false); /* CPU computation is not supported */
#endif /* FRACTAL_USE_CUDA */

    Y.FinishWrite(stream);
}


void Engine::FuncSoftplus(Matrix<FLOAT> &X, Matrix<FLOAT> &Y, PStream &stream)
{
    verify(X.GetEngine() == this);
    verify(Y.GetEngine() == this);
    verify(X.GetNumRows() == Y.GetNumRows());
    verify(X.GetNumCols() == Y.GetNumCols());

    FLOAT *ptrX, *ptrY;

    ptrX = X.GetPtrForReadWrite(stream);
    ptrY = Y.GetPtrForWrite(stream);

#ifdef FRACTAL_USE_CUDA
    SetComputeLoc(stream.loc);
    cudaKernels::FuncSoftplus(ptrX, X.GetLeadingDim(),
            ptrY, Y.GetLeadingDim(),
            Y.GetNumRows(), Y.GetNumCols(), stream.cudaStream);
#else
    verify(false); /* CPU computation is not supported */
#endif /* FRACTAL_USE_CUDA */

    Y.FinishWrite(stream);
}


void Engine::FuncRectLinear(Matrix<FLOAT> &X, Matrix<FLOAT> &Y, PStream &stream)
{
    verify(X.GetEngine() == this);
    verify(Y.GetEngine() == this);
    verify(X.GetNumRows() == Y.GetNumRows());
    verify(X.GetNumCols() == Y.GetNumCols());

    FLOAT *ptrX, *ptrY;

    ptrX = X.GetPtrForReadWrite(stream);
    ptrY = Y.GetPtrForWrite(stream);

#ifdef FRACTAL_USE_CUDA
    SetComputeLoc(stream.loc);
    cudaKernels::FuncRectLinear(ptrX, X.GetLeadingDim(),
            ptrY, Y.GetLeadingDim(),
            Y.GetNumRows(), Y.GetNumCols(), stream.cudaStream);
#else
    verify(false); /* CPU computation is not supported */
#endif /* FRACTAL_USE_CUDA */

    Y.FinishWrite(stream);
}


void Engine::FuncSoftmax(Matrix<FLOAT> &X, Matrix<FLOAT> &Y, PStream &stream)
{
    verify(X.GetEngine() == this);
    verify(Y.GetEngine() == this);
    verify(X.GetNumRows() == Y.GetNumRows());
    verify(X.GetNumCols() == Y.GetNumCols());

    FLOAT *ptrX, *ptrY;

    ptrX = X.GetPtrForReadWrite(stream);
    ptrY = Y.GetPtrForWrite(stream);

#ifdef FRACTAL_USE_CUDA
    SetComputeLoc(stream.loc);
    cudaKernels::FuncSoftmax<FLOAT>(ptrX, X.GetLeadingDim(),
            ptrY, Y.GetLeadingDim(),
            Y.GetNumRows(), Y.GetNumCols(),
            stream.cudaStream);
#else
    verify(false); /* CPU computation is not supported */
#endif /* FRACTAL_USE_CUDA */

    Y.FinishWrite(stream);
}


void Engine::FuncCTCDecode(Matrix<FLOAT> &X, Matrix<FLOAT> &Y, Matrix<INT> &prevIdxMax, Matrix<INT> &idxMax, const unsigned long nStream, PStream &stream)
{
    verify(X.GetEngine() == this);
    verify(Y.GetEngine() == this);
    verify(prevIdxMax.GetEngine() == this);
    verify(idxMax.GetEngine() == this);

    verify(X.GetNumRows() == Y.GetNumRows());
    verify(X.GetNumCols() == Y.GetNumCols());
    verify(idxMax.GetNumRows() == 1);
    verify(idxMax.GetNumCols() == X.GetNumCols());
    verify(prevIdxMax.GetNumRows() == 1);
    verify(prevIdxMax.GetNumCols() == nStream);

    verify(X.GetNumCols() % nStream == 0);

    verify(prevIdxMax.GetNumRows() == prevIdxMax.GetLeadingDim());
    verify(idxMax.GetNumRows() == idxMax.GetLeadingDim());


    FLOAT *ptrX, *ptrY;
    INT *ptrIdxMax, *ptrPrevIdxMax;

    ptrX = X.GetPtrForReadWrite(stream);
    ptrY = Y.GetPtrForWrite(stream);
    ptrPrevIdxMax = prevIdxMax.GetPtrForReadWrite(stream);
    ptrIdxMax = idxMax.GetPtrForWrite(stream);

#ifdef FRACTAL_USE_CUDA
    SetComputeLoc(stream.loc);
    cudaKernels::FuncCTCDecode<INT, FLOAT>(ptrX, X.GetLeadingDim(),
            ptrY, Y.GetLeadingDim(), ptrPrevIdxMax, ptrIdxMax,
            Y.GetNumRows(), Y.GetNumCols() / nStream,
            nStream, stream.cudaStream);
#else
    verify(false); /* CPU computation is not supported */
#endif /* FRACTAL_USE_CUDA */

    Y.FinishWrite(stream);
    idxMax.FinishWrite(stream);
}


void Engine::FuncBoundRange(Matrix<FLOAT> &X, Matrix<FLOAT> &Y, const FLOAT min, const FLOAT max, PStream &stream)
{
    verify(X.GetEngine() == this);
    verify(Y.GetEngine() == this);
    verify(X.GetNumRows() == Y.GetNumRows());
    verify(X.GetNumCols() == Y.GetNumCols());

    FLOAT *ptrX, *ptrY;

    ptrX = X.GetPtrForReadWrite(stream);
    ptrY = Y.GetPtrForWrite(stream);

#ifdef FRACTAL_USE_CUDA
    SetComputeLoc(stream.loc);
    cudaKernels::FuncBoundRange(ptrX, X.GetLeadingDim(),
            ptrY, Y.GetLeadingDim(),
            min, max, Y.GetNumRows(), Y.GetNumCols(),
            stream.cudaStream);
#else
    verify(false); /* CPU computation is not supported */
#endif

    Y.FinishWrite(stream);
}


void Engine::FuncSigmoidDeriv(Matrix<FLOAT> &X, Matrix<FLOAT> &Y, PStream &stream)
{
    verify(X.GetEngine() == this);
    verify(Y.GetEngine() == this);
    verify(X.GetNumRows() == Y.GetNumRows());
    verify(X.GetNumCols() == Y.GetNumCols());

    FLOAT *ptrX, *ptrY;

    ptrX = X.GetPtrForReadWrite(stream);
    ptrY = Y.GetPtrForWrite(stream);

#ifdef FRACTAL_USE_CUDA
    SetComputeLoc(stream.loc);
    cudaKernels::FuncSigmoidDeriv<FLOAT>(ptrX, X.GetLeadingDim(),
            ptrY, Y.GetLeadingDim(),
            Y.GetNumRows(), Y.GetNumCols(),
            stream.cudaStream);
#else
    verify(false); /* CPU computation is not supported */
#endif /* FRACTAL_USE_CUDA */

    Y.FinishWrite(stream);
}


void Engine::FuncTanhDeriv(Matrix<FLOAT> &X, Matrix<FLOAT> &Y, PStream &stream)
{
    verify(X.GetEngine() == this);
    verify(Y.GetEngine() == this);
    verify(X.GetNumRows() == Y.GetNumRows());
    verify(X.GetNumCols() == Y.GetNumCols());

    FLOAT *ptrX, *ptrY;

    ptrX = X.GetPtrForReadWrite(stream);
    ptrY = Y.GetPtrForWrite(stream);

#ifdef FRACTAL_USE_CUDA
    SetComputeLoc(stream.loc);
    cudaKernels::FuncTanhDeriv<FLOAT>(ptrX, X.GetLeadingDim(),
            ptrY, Y.GetLeadingDim(),
            Y.GetNumRows(), Y.GetNumCols(),
            stream.cudaStream);
#else
    verify(false); /* CPU computation is not supported */
#endif /* FRACTAL_USE_CUDA */

    Y.FinishWrite(stream);
}


void Engine::FuncSoftplusDeriv(Matrix<FLOAT> &X, Matrix<FLOAT> &Y, PStream &stream)
{
    verify(X.GetEngine() == this);
    verify(Y.GetEngine() == this);
    verify(X.GetNumRows() == Y.GetNumRows());
    verify(X.GetNumCols() == Y.GetNumCols());

    FLOAT *ptrX, *ptrY;

    ptrX = X.GetPtrForReadWrite(stream);
    ptrY = Y.GetPtrForWrite(stream);

#ifdef FRACTAL_USE_CUDA
    SetComputeLoc(stream.loc);
    cudaKernels::FuncSoftplusDeriv<FLOAT>(ptrX, X.GetLeadingDim(),
            ptrY, Y.GetLeadingDim(),
            Y.GetNumRows(), Y.GetNumCols(),
            stream.cudaStream);
#else
    verify(false); /* CPU computation is not supported */
#endif /* FRACTAL_USE_CUDA */

    Y.FinishWrite(stream);
}


void Engine::FuncRectLinearDeriv(Matrix<FLOAT> &X, Matrix<FLOAT> &Y, PStream &stream)
{
    verify(X.GetEngine() == this);
    verify(Y.GetEngine() == this);
    verify(X.GetNumRows() == Y.GetNumRows());
    verify(X.GetNumCols() == Y.GetNumCols());

    FLOAT *ptrX, *ptrY;

    ptrX = X.GetPtrForReadWrite(stream);
    ptrY = Y.GetPtrForWrite(stream);

#ifdef FRACTAL_USE_CUDA
    SetComputeLoc(stream.loc);
    cudaKernels::FuncRectLinearDeriv<FLOAT>(ptrX, X.GetLeadingDim(),
            ptrY, Y.GetLeadingDim(),
            Y.GetNumRows(), Y.GetNumCols(),
            stream.cudaStream);
#else
    verify(false); /* CPU computation is not supported */
#endif /* FRACTAL_USE_CUDA */

    Y.FinishWrite(stream);
}


void Engine::OneHotEncode(Matrix<INT> &index, Matrix<FLOAT> &vector, PStream &stream)
{
    verify(index.GetEngine() == this);
    verify(vector.GetEngine() == this);
    verify(index.GetNumCols() == vector.GetNumCols());
    verify(index.GetNumRows() == 1);

    verify(index.GetNumRows() == index.GetLeadingDim());
    verify(vector.GetNumRows() == vector.GetLeadingDim());

    INT *ptrIndex;
    FLOAT *ptrVector;

    ptrIndex = index.GetPtrForReadWrite(stream);
    ptrVector = vector.GetPtrForWrite(stream);

#ifdef FRACTAL_USE_CUDA
    SetComputeLoc(stream.loc);
    cudaKernels::OneHotEncode<INT, FLOAT>(ptrIndex, ptrVector,
            vector.GetNumRows(), vector.GetNumCols(),
            stream.cudaStream);
#else
    verify(false); /* CPU computation is not supported */
#endif /* FRACTAL_USE_CUDA */

    vector.FinishWrite(stream);
}


void Engine::GenerateDropoutMask(Matrix<FLOAT> &dropoutMask, const FLOAT dropoutRate, PStream &stream)
{
    verify(dropoutMask.GetEngine() == this);
    verify(dropoutRate >= (FLOAT) 0 && dropoutRate < (FLOAT) 1);

    verify(dropoutMask.GetNumRows() == dropoutMask.GetLeadingDim());

    FLOAT *ptrMask;

    ptrMask = dropoutMask.GetPtrForWrite(stream);

#ifdef FRACTAL_USE_CUDA
    unsigned long n;
    n = dropoutMask.GetNumRows() * dropoutMask.GetNumCols();

    SetComputeLoc(stream.loc);
    verify(curandSetStream(curandGen[GetComputeLocIdx(stream.loc)], stream.cudaStream) == CURAND_STATUS_SUCCESS);
    verify(RANDU(curandGen[GetComputeLocIdx(stream.loc)], ptrMask, n) == CURAND_STATUS_SUCCESS);
    cudaKernels::GenerateDropoutMask<FLOAT>(ptrMask, ptrMask, n, dropoutRate, stream.cudaStream);
#else
    verify(false); /* CPU computation is not supported */
#endif /* FRACTAL_USE_CUDA */

    dropoutMask.FinishWrite(stream);
}


void Engine::Adadelta(Matrix<FLOAT> &deltas, Matrix<FLOAT> &derivs, Matrix<FLOAT> &msDeriv, Matrix<FLOAT> &msDelta, const FLOAT learningRate, const FLOAT decayRate, PStream &stream)
{
    verify(deltas.GetEngine() == this);
    verify(derivs.GetEngine() == this);
    verify(msDeriv.GetEngine() == this);

    verify(derivs.GetNumRows() == deltas.GetNumRows());
    verify(derivs.GetNumCols() == deltas.GetNumCols());
    verify(derivs.GetNumRows() == msDeriv.GetNumRows());
    verify(derivs.GetNumCols() == msDeriv.GetNumCols());
    verify(derivs.GetNumRows() == msDelta.GetNumRows());
    verify(derivs.GetNumCols() == msDelta.GetNumCols());

    verify(deltas.GetNumRows() == deltas.GetLeadingDim());
    verify(derivs.GetNumRows() == derivs.GetLeadingDim());
    verify(msDeriv.GetNumRows() == msDeriv.GetLeadingDim());
    verify(msDelta.GetNumRows() == msDelta.GetLeadingDim());

    FLOAT *ptrDerivs, *ptrDeltas, *ptrMsDeriv, *ptrMsDelta;

    ptrDeltas = deltas.GetPtrForWrite(stream);
    ptrDerivs = derivs.GetPtrForReadWrite(stream);
    ptrMsDeriv = msDeriv.GetPtrForReadWrite(stream);
    ptrMsDelta = msDelta.GetPtrForReadWrite(stream);

#ifdef FRACTAL_USE_CUDA
    SetComputeLoc(stream.loc);
    cudaKernels::Adadelta<FLOAT>(ptrDeltas, ptrDerivs, ptrMsDeriv, ptrMsDelta,
            learningRate, decayRate, derivs.GetNumRows() * derivs.GetNumCols(), stream.cudaStream);
#else
    verify(false); /* CPU computation is not supported */
#endif /* FRACTAL_USE_CUDA */

    deltas.FinishWrite(stream);
    msDeriv.FinishWrite(stream);
    msDelta.FinishWrite(stream);
}


void Engine::Rmsprop(Matrix<FLOAT> &newDerivs, Matrix<FLOAT> &derivs, Matrix<FLOAT> &msDeriv, const FLOAT decayRate, PStream &stream)
{
    verify(newDerivs.GetEngine() == this);
    verify(derivs.GetEngine() == this);
    verify(msDeriv.GetEngine() == this);

    verify(derivs.GetNumRows() == newDerivs.GetNumRows());
    verify(derivs.GetNumCols() == newDerivs.GetNumCols());
    verify(derivs.GetNumRows() == msDeriv.GetNumRows());
    verify(derivs.GetNumCols() == msDeriv.GetNumCols());

    verify(newDerivs.GetNumRows() == newDerivs.GetLeadingDim());
    verify(derivs.GetNumRows() == derivs.GetLeadingDim());
    verify(msDeriv.GetNumRows() == msDeriv.GetLeadingDim());

    FLOAT *ptrDerivs, *ptrNewDerivs, *ptrMsDeriv;

    ptrNewDerivs = derivs.GetPtrForWrite(stream);
    ptrDerivs = derivs.GetPtrForReadWrite(stream);
    ptrMsDeriv = msDeriv.GetPtrForReadWrite(stream);

#ifdef FRACTAL_USE_CUDA
    SetComputeLoc(stream.loc);
    cudaKernels::Rmsprop<FLOAT>(ptrNewDerivs, ptrDerivs, ptrMsDeriv, decayRate,
            derivs.GetNumRows() * derivs.GetNumCols(), stream.cudaStream);
#else
    verify(false); /* CPU computation is not supported */
#endif /* FRACTAL_USE_CUDA */


    newDerivs.FinishWrite(stream);
    msDeriv.FinishWrite(stream);
}


void Engine::EventCreate(PEvent &event, const unsigned long loc)
{
    mtxEvent.lock();

    event.engine = this;
    event.loc = loc;

#ifdef FRACTAL_USE_CUDA
    event.cudaStream = NULL;
    SetComputeLoc(loc);
    verify(cudaEventCreateWithFlags(&event.cudaEvent, cudaEventDisableTiming) == cudaSuccess);
#else
    event.streamId = 0;
#endif /* FRACTAL_USE_CUDA */

    eventCount++;

    mtxEvent.unlock();
}


void Engine::EventDestroy(PEvent &event)
{
    mtxEvent.lock();

    verify(event.engine == this);

#ifdef FRACTAL_USE_CUDA
    verify(cudaEventDestroy(event.cudaEvent) == cudaSuccess);
#endif /* FRACTAL_USE_CUDA */

    eventCount--;
    event.engine = NULL;

    mtxEvent.unlock();
}


void Engine::EventRecord(PEvent &event, PStream &stream)
{
    mtxEvent.lock();
    mtxStream.lock();

    verify(event.engine == this);
    verify(stream.engine == this);
    verify(event.loc == stream.loc);

#ifdef FRACTAL_USE_CUDA
    event.cudaStream = stream.cudaStream;
    verify(cudaEventRecord(event.cudaEvent, stream.cudaStream) == cudaSuccess);
#else
    event.streamId = stream.streamId;
#endif /* FRACTAL_USE_CUDA */

    mtxEvent.unlock();
    mtxStream.unlock();
}


void Engine::EventSynchronize(PEvent &event)
{
    mtxEvent.lock();

    verify(event.engine == this);

#ifdef FRACTAL_USE_CUDA
    verify(cudaEventSynchronize(event.cudaEvent) == cudaSuccess);
#endif /* FRACTAL_USE_CUDA */

    mtxEvent.unlock();
}


void Engine::StreamCreate(PStream &stream, const unsigned long loc)
{
    mtxStream.lock();

    stream.engine = this;
    stream.loc = loc;

#ifdef FRACTAL_USE_CUDA
#ifdef FRACTAL_CUDA_MULTISTREAM
    if(stream.loc == hostLoc) stream.cudaStream = 0;
    else
    {
        SetComputeLoc(loc);
        verify(cudaStreamCreateWithFlags(&stream.cudaStream, cudaStreamNonBlocking) == cudaSuccess);
    }
#else
    stream.cudaStream = 0;
#endif /* FRACTAL_CUDA_MULTISTREAM */
#endif /* FRACTAL_USE_CUDA */

    streamCount++;

    mtxStream.unlock();
}


void Engine::StreamDestroy(PStream &stream)
{
    mtxStream.lock();

    verify(stream.engine == this);

#ifdef FRACTAL_USE_CUDA
#ifdef FRACTAL_CUDA_MULTISTREAM
    if(stream.loc != hostLoc) verify(cudaStreamDestroy(stream.cudaStream) == cudaSuccess);
#endif /* FRACTAL_CUDA_MULTISTREAM */
#endif /* FRACTAL_USE_CUDA */

    streamCount--;
    stream.engine = NULL;

    mtxStream.unlock();
}


void Engine::StreamWaitEvent(PStream &stream, PEvent &event)
{
    mtxEvent.lock();
    mtxStream.lock();

    verify(event.engine == this);
    verify(stream.engine == this);

#ifdef FRACTAL_USE_CUDA
    verify(cudaStreamWaitEvent(stream.cudaStream, event.cudaEvent, 0) == cudaSuccess);
#else

#endif /* FRACTAL_USE_CUDA */

    mtxEvent.unlock();
    mtxStream.unlock();
}


void Engine::StreamSynchronize(PStream &stream)
{
#ifdef FRACTAL_USE_CUDA
    mtxStream.lock();

    verify(stream.engine == this);

    cudaStream_t cudaStreamCopy = stream.cudaStream;

    mtxStream.unlock();

    verify(cudaStreamSynchronize(cudaStreamCopy) == cudaSuccess);
#endif /* FRACTAL_USE_CUDA */
}


void Engine::SetRandomSeed(unsigned long long seed)
{
#ifdef FRACTAL_USE_CUDA
    std::mt19937_64 randGen(seed);
    std::uniform_int_distribution<unsigned long long> uniformDist;

    for(unsigned long i = 0; i < GetNumComputeLocs(); i++)
    {
        SetComputeLoc(GetComputeLoc(i));
        verify(curandSetPseudoRandomGeneratorSeed(curandGen[i], uniformDist(randGen)) == CURAND_STATUS_SUCCESS);
    }
#else
    verify(false);
#endif /* FRACTAL_USE_CUDA */
}


void Engine::SetComputeLoc(unsigned long loc)
{
    verify(loc < numLoc);
    verify(loc != hostLoc);

#ifdef FRACTAL_USE_CUDA
    verify(cudaSetDevice(GetComputeLocIdx(loc)) == cudaSuccess);
#endif /* FRACTAL_USE_CUDA */

    currentLoc = loc;
}


unsigned long Engine::GetComputeLoc(unsigned long index)
{
    return index + 1;
}


unsigned long Engine::GetComputeLocIdx(unsigned long loc)
{
    verify(loc < numLoc);
    verify(loc != hostLoc);
    return loc - 1;
}


unsigned long Engine::GetNumComputeLocs()
{
    return numLoc - 1;
}


}

