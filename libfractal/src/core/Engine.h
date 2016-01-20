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


#ifndef FRACTAL_ENGINE_H_
#define FRACTAL_ENGINE_H_

#define FRACTAL_USE_CUDA /* For now, always use CUDA */

#include <mutex>
#include <vector>

#ifdef FRACTAL_USE_CUDA

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>

#endif /* FRACTAL_USE_CUDA */

#include "Matrix.h"
#include "Mem.h"
#include "FractalCommon.h"




namespace fractal
{


class PEvent
{
public:
    PEvent() : engine(NULL) {}

    unsigned long loc;

#ifdef FRACTAL_USE_CUDA
    cudaEvent_t cudaEvent;
    cudaStream_t cudaStream;
#else
    unsigned long eventId;
    unsigned long streamId;
#endif /* FRACTAL_USE_CUDA */

    Engine *engine;
};


class PStream
{
public:
    PStream() : engine(NULL) {}

    unsigned long loc;

#ifdef FRACTAL_USE_CUDA
    cudaStream_t cudaStream;
#else
    unsigned long streamId;
#endif /* FRACTAL_USE_CUDA */

    Engine *engine;
};


class Engine
{
public:
    Engine();
    virtual ~Engine();

    inline const unsigned long GetNumLocs() { return numLoc; }
    inline const unsigned long GetHostLoc() { return hostLoc; }

    void MemAdd(Mem *mem);
    void MemDel(Mem *mem);

    void MemAlloc(Mem *mem, unsigned long loc);
    void MemDealloc(Mem *mem);

    void MemPull(Mem *mem, const unsigned long loc, PStream &stream);
    void MemCopy(const Mem *memSrc, const size_t offsetSrc,
            Mem *memDst, const size_t offsetDst, const size_t size, PStream &stream);

    void MemCopyFromHost(Mem *memDst, const size_t offsetDst, const void *ptrSrc, const size_t size, PStream &stream);
    void MemCopyToHost(const Mem *memSrc, const size_t offsetSrc, void *ptrDst, const size_t size, PStream &stream);

    /* C = alpha * A * B + beta * C*/
    void MatMult(Matrix<FLOAT> &A, const bool transA, Matrix<FLOAT> &B, const bool transB, Matrix<FLOAT> &C, const FLOAT alpha, const FLOAT beta, PStream &stream);

    /* C = A .* B */
    void MatElemMult(Matrix<FLOAT> &A, Matrix<FLOAT> &B, Matrix<FLOAT> &C, PStream &stream);

    /* B = alpha * A + B */
    void MatAdd(Matrix<FLOAT> &A, Matrix<FLOAT> &B, const FLOAT alpha, PStream &stream);

    /* C = A + B */
    void MatAdd(Matrix<FLOAT> &A, Matrix<FLOAT> &B, Matrix<FLOAT> &C, PStream &stream);

    /* C = A - B */
    void MatSub(Matrix<FLOAT> &A, Matrix<FLOAT> &B, Matrix<FLOAT> &C, PStream &stream);

    void MatAddToDiag(Matrix<FLOAT> &mat, const FLOAT val, const unsigned long shift, PStream &stream);
    
    void MatMakeTri(Matrix<FLOAT> &mat, const bool upper, PStream &stream);

    void MatSet(Matrix<FLOAT> &mat, const FLOAT val, PStream &stream);
    void MatSet(Matrix<INT> &mat, const INT val, PStream &stream);

    void MatRandN(Matrix<FLOAT> &mat, const FLOAT mean, const FLOAT stdev, PStream &stream);
    void MatRandU(Matrix<FLOAT> &mat, const FLOAT a, const FLOAT b, PStream &stream);

    /* B = A */
    void MatCopy(Matrix<FLOAT> &A, Matrix<FLOAT> &B, PStream &stream);

    /* B = tr(A) */
    void MatTranspose(Matrix<FLOAT> &A, Matrix<FLOAT> &B, PStream &stream);

    /* B(:, i) = A(:, srcIdx(i)) */
    void MatShuffle(Matrix<FLOAT> &A, Matrix<FLOAT> &B, Matrix<INT> &srcIdx, PStream &stream);

    /* Y = f(X) */
    void FuncSigmoid(Matrix<FLOAT> &X, Matrix<FLOAT> &Y, PStream &stream);
    void FuncTanh(Matrix<FLOAT> &X, Matrix<FLOAT> &Y, PStream &stream);
    void FuncSoftplus(Matrix<FLOAT> &X, Matrix<FLOAT> &Y, PStream &stream);
    void FuncRectLinear(Matrix<FLOAT> &X, Matrix<FLOAT> &Y, PStream &stream);
    void FuncSoftmax(Matrix<FLOAT> &X, Matrix<FLOAT> &Y, PStream &stream);
    void FuncCTCDecode(Matrix<FLOAT> &X, Matrix<FLOAT> &Y, Matrix<INT> &prevIdxMax, Matrix<INT> &idxMax, const unsigned long nStream, PStream &stream);
    void FuncBoundRange(Matrix<FLOAT> &X, Matrix<FLOAT> &Y, const FLOAT min, const FLOAT max, PStream &stream);

    /* Y = f'(Z) where X = f(Z) */
    void FuncSigmoidDeriv(Matrix<FLOAT> &X, Matrix<FLOAT> &Y, PStream &stream);
    void FuncTanhDeriv(Matrix<FLOAT> &X, Matrix<FLOAT> &Y, PStream &stream);
    void FuncSoftplusDeriv(Matrix<FLOAT> &X, Matrix<FLOAT> &Y, PStream &stream);
    void FuncRectLinearDeriv(Matrix<FLOAT> &X, Matrix<FLOAT> &Y, PStream &stream);

    void OneHotEncode(Matrix<INT> &index, Matrix<FLOAT> &vector, PStream &stream);

    void GenerateDropoutMask(Matrix<FLOAT> &dropoutMask, const FLOAT dropoutRate, PStream &stream);

    /* Rmsprop: newderivs can be identical to derivs */
    void Rmsprop(Matrix<FLOAT> &newDerivs, Matrix<FLOAT> &derivs, Matrix<FLOAT> &meanSquares, const FLOAT decayRate, PStream &stream);

    void Adadelta(Matrix<FLOAT> &deltas, Matrix<FLOAT> &derivs, Matrix<FLOAT> &msDeriv, Matrix<FLOAT> &msDelta, const FLOAT learningRate, const FLOAT decayRate, PStream &stream);

    void EventCreate(PEvent &event, const unsigned long loc);
    void EventDestroy(PEvent &event);
    void EventRecord(PEvent &event, PStream &stream);
    void EventSynchronize(PEvent &event);

    void StreamCreate(PStream &stream, const unsigned long loc);
    void StreamDestroy(PStream &stream);
    void StreamWaitEvent(PStream &stream, PEvent &event);
    void StreamSynchronize(PStream &stream);

    void SetRandomSeed(unsigned long long seed);

    unsigned long GetComputeLoc(unsigned long index);
    unsigned long GetComputeLocIdx(unsigned long loc);
    unsigned long GetNumComputeLocs();


protected:
    void MemCopy(const Mem *memSrc, const size_t offsetSrc, const unsigned long locSrc,
            Mem *memDst, const size_t offsetDst, const unsigned long locDst, const size_t size, PStream &stream);

    void MemCopy(const void *ptrSrc, const unsigned long locSrc,
            void *ptrDst, const unsigned long locDst, const size_t size, PStream &stream);

    void SetComputeLoc(unsigned long loc);

    unsigned long numLoc;
    unsigned long hostLoc;
    unsigned long currentLoc;

    unsigned long memCount;
    unsigned long memAllocCount;

    unsigned long eventCount;
    unsigned long streamCount;

    std::recursive_mutex mtxMem;
    std::mutex mtxStream;
    std::mutex mtxEvent;

#ifdef FRACTAL_USE_CUDA
    std::vector<cublasHandle_t> cublasHandle;
    std::vector<curandGenerator_t> curandGen;
#endif /* FRACTAL_USE_CUDA */
};


}

#endif /* FRACTAL_ENGINE_H_ */


