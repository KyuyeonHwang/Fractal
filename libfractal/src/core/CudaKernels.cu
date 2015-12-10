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


#include "CudaKernels.h"

#define THREAD_PER_BLOCK 512

namespace fractal
{

namespace cudaKernels
{


template<class T>
inline __device__ T _exp(const T x);

template<class T>
inline __device__ T _log(const T x);

template<class T>
inline __device__ T _sqrt(const T x);

template<class T>
static __global__ void MatSetKernel(T *x, const int ldx, const T val,
        const int nRows, const int nCols);

template<class T>
static __global__ void MatElemMultKernel(const T *x, const int ldx,
        const T *y, const int ldy, T *z, const int ldz,
        const int nRows, const int nCols);

template<class T>
static __global__ void MatAddKernel(const T *x, const int ldx,
        const T *y, const int ldy, T *z, const int ldz,
        const int nRows, const int nCols);

template<class T>
static __global__ void MatSubKernel(const T *x, const int ldx,
        const T *y, const int ldy, T *z, const int ldz,
        const int nRows, const int nCols);

template<class T>
static __global__ void MatAddToDiagKernel(T *x, const T val,
        const int shift, const int nRows);

template<class T>
static __global__ void MatMakeTriKernel(T *x, const bool upper, const int nRows);

template<class T>
static __global__ void FuncSigmoidKernel(const T *x, const int ldx,
        T *y, const int ldy, const int nRows, const int nCols);

template<class T>
static __global__ void FuncTanhKernel(const T *x, const int ldx,
        T *y, const int ldy, const int nRows, const int nCols);

template<class T>
static __global__ void FuncSoftplusKernel(const T *x, const int ldx,
        T *y, const int ldy, const int nRows, const int nCols);

template<class T>
static __global__ void FuncRectLinearKernel(const T *x, const int ldx,
        T *y, const int ldy, const int nRows, const int nCols);

template<class T>
static __global__ void FuncSoftmaxKernel(const T *x, const int ldx,
        T *y, const int ldy, const int n);

template<class I, class V>
static __global__ void FuncCTCDecodeKernel(const V *x, const int ldx,
        V *y, const int ldy, const I *prevIdxMax, I *idxMax,
        const int layerSize, const int nStep);

template<class T>
static __global__ void FuncBoundRangeKernel(const T *x, const int ldx,
        T *y, const int ldy, const T _min, const T _max,
        const int nRows, const int nCols);

template<class T>
static __global__ void FuncSigmoidDerivKernel(const T *x, const int ldx,
        T *y, const int ldy, const int nRows, const int nCols);

template<class T>
static __global__ void FuncTanhDerivKernel(const T *x, const int ldx,
        T *y, const int ldy, const int nRows, const int nCols);

template<class T>
static __global__ void FuncSoftplusDerivKernel(const T *x, const int ldx,
        T *y, const int ldy, const int nRows, const int nCols);

template<class T>
static __global__ void FuncRectLinearDerivKernel(const T *x, const int ldx,
        T *y, const int ldy, const int nRows, const int nCols);

template<class I, class V>
static __global__ void OneHotEncodeKernel(const I *index, V *vector, const int n);

template<class T>
static __global__ void GenerateDropoutMaskKernel(T *mask, const T *uniformDist,
        const int n, const T dropoutRate);

template<class T>
static __global__ void RmspropKernel(T *newDerivs, const T *derivs, T *msDeriv,
        const T decayRate, const int n);

template<class T>
static __global__ void AdadeltaKernel(T *deltas, const T *derivs, T *msDeriv, T *msDelta,
        const T learningRate, const T decayRate, const int n);


template<>
inline __device__ float _exp<float>(const float x)
{
    return min(__expf(x), 1e30);
}


template<>
inline __device__ double _exp<double>(const double x)
{
    return min(exp(x), 1e300);
}


template<>
inline __device__ float _log<float>(const float x)
{
    return __logf(x);
}


template<>
inline __device__ double _log<double>(const double x)
{
    return log(x);
}


template<>
inline __device__ float _sqrt<float>(const float x)
{
    return __fsqrt_rn(x);
}


template<>
inline __device__ double _sqrt<double>(const double x)
{
    return __dsqrt_rn(x);
}



template<class T>
static __global__ void MatSetKernel(T *x, const int ldx,
        const T val, const int nRows, const int nCols)
{
    int thdIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int col = thdIdx / nRows;
    int row = thdIdx - col * nRows;
    int xIdx = col * ldx + row;

    if(thdIdx >= nRows * nCols) return;

    x[xIdx] = val;
}


template<class T>
static __global__ void MatElemMultKernel(const T *x, const int ldx,
        const T *y, const int ldy, T *z, const int ldz, const int nRows, const int nCols)
{
    int thdIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int col = thdIdx / nRows;
    int row = thdIdx - col * nRows;
    int xIdx = col * ldx + row;
    int yIdx = col * ldy + row;
    int zIdx = col * ldz + row;

    if(thdIdx >= nRows * nCols) return;

    z[zIdx] = x[xIdx] * y[yIdx];
}


template<class T>
static __global__ void MatAddKernel(const T *x, const int ldx,
        const T *y, const int ldy, T *z, const int ldz, const int nRows, const int nCols)
{
    int thdIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int col = thdIdx / nRows;
    int row = thdIdx - col * nRows;
    int xIdx = col * ldx + row;
    int yIdx = col * ldy + row;
    int zIdx = col * ldz + row;

    if(thdIdx >= nRows * nCols) return;

    z[zIdx] = x[xIdx] + y[yIdx];
}


template<class T>
static __global__ void MatSubKernel(const T *x, const int ldx,
        const T *y, const int ldy, T *z, const int ldz, const int nRows, const int nCols)
{
    int thdIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int col = thdIdx / nRows;
    int row = thdIdx - col * nRows;
    int xIdx = col * ldx + row;
    int yIdx = col * ldy + row;
    int zIdx = col * ldz + row;

    if(thdIdx >= nRows * nCols) return;

    z[zIdx] = x[xIdx] - y[yIdx];
}


template<class T>
static __global__ void MatAddToDiagKernel(T *x, const T val,
        const int shift, const int nRows)
{
    int idx;

    idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= nRows) return;

    idx = (idx * (nRows + 1) + shift) % (nRows * nRows);

    x[idx] += val;
}


template<class T>
static __global__ void MatMakeTriKernel(T *x, const bool upper, const int nRows)
{
    int idx;
    int iRow, iCol;

    idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= nRows * nRows) return;

    /* Column-major order */

    iCol = idx / nRows;
    iRow = idx % nRows;

    if(upper == true) /* Make upper triangular matrix */
    {
        x[idx] *= (T) (iCol >= iRow);
    }
    else
    {
        x[idx] *= (T) (iCol <= iRow);
    }
}


template<class T>
static __global__ void FuncSigmoidKernel(const T *x, const int ldx,
        T *y, const int ldy, const int nRows, const int nCols)
{
    int thdIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int col = thdIdx / nRows;
    int row = thdIdx - col * nRows;
    int xIdx = col * ldx + row;
    int yIdx = col * ldy + row;

    if(thdIdx >= nRows * nCols) return;

    y[yIdx] = (T)1 / ((T)1 + _exp<T>(-x[xIdx]));
}


template<class T>
static __global__ void FuncTanhKernel(const T *x, const int ldx,
        T *y, const int ldy, const int nRows, const int nCols)
{
    int thdIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int col = thdIdx / nRows;
    int row = thdIdx - col * nRows;
    int xIdx = col * ldx + row;
    int yIdx = col * ldy + row;

    if(thdIdx >= nRows * nCols) return;

    T v = _exp<T>((T)(-2) * x[xIdx]);

    y[yIdx] = (T)2 / ((T)1 + v) - (T)1;
}


template<class T>
static __global__ void FuncSoftplusKernel(const T *x, const int ldx,
        T *y, const int ldy, const int nRows, const int nCols)
{
    int thdIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int col = thdIdx / nRows;
    int row = thdIdx - col * nRows;
    int xIdx = col * ldx + row;
    int yIdx = col * ldy + row;

    if(thdIdx >= nRows * nCols) return;

    y[yIdx] = _log<T>((T)1 + _exp<T>(x[xIdx]));
}


template<class T>
static __global__ void FuncRectLinearKernel(const T *x, const int ldx,
        T *y, const int ldy, const int nRows, const int nCols)
{
    int thdIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int col = thdIdx / nRows;
    int row = thdIdx - col * nRows;
    int xIdx = col * ldx + row;
    int yIdx = col * ldy + row;

    if(thdIdx >= nRows * nCols) return;

    //y[yIdx] = max((T)0, x[xIdx]);

    /* Leaky */
    y[yIdx] = max((T)0.01 * x[xIdx], x[xIdx]);
}


template<class T>
static __global__ void FuncSoftmaxKernel(const T *x, const int ldx,
        T *y, const int ldy, const int n)
{
    __shared__ T _v[THREAD_PER_BLOCK];
    T v_tmp, v_max;
    int i;

    x += blockIdx.x * ldx;
    y += blockIdx.x * ldy;


    /* Sequential reduction(max) */
    v_tmp = threadIdx.x < n ? x[threadIdx.x] : (FLOAT) 0;

    #pragma unroll
    for(i = threadIdx.x + blockDim.x; i < n; i += blockDim.x)
    {
        v_tmp = max(v_tmp, x[i]);
    }

    _v[threadIdx.x] = v_tmp;

    __syncthreads();

    /* Parallel reduction(max) */
    i = (blockDim.x >> 1);

    for(; i > 0; i >>= 1)
    {
        if(threadIdx.x < i && threadIdx.x + i < n)
        {
            v_tmp = max(v_tmp, _v[threadIdx.x + i]);
            _v[threadIdx.x] = v_tmp;
        }

        __syncthreads();
    }

    v_max = _v[0];

    __syncthreads();

    /* Sequential reduction(+) */
    v_tmp = (T) 0;

    #pragma unroll
    for(i = threadIdx.x; i < n; i += blockDim.x)
    {
        v_tmp += _exp<T>(x[i] - v_max);
    }

    _v[threadIdx.x] = v_tmp;

    __syncthreads();

    /* Parallel reduction(+) */
    i = (blockDim.x >> 1);
    if(threadIdx.x < i)
        v_tmp = _v[threadIdx.x];

    for(; i > 0; i >>= 1)
    {
        if(threadIdx.x < i)
        {
            v_tmp += _v[threadIdx.x + i];
            _v[threadIdx.x] = v_tmp;
        }

        __syncthreads();
    }


    /* Update */
    v_tmp = _v[0];

    #pragma unroll
    for(i = threadIdx.x; i < n; i += blockDim.x)
    {
        y[i] = _exp<T>(x[i] - v_max) / v_tmp;
    }
}


template<class I, class V>
static __global__ void FuncCTCDecodeKernel(const V *x, const int ldx,
        V *y, const int ldy, const I *prevIdxMax, I *idxMax,
        const int layerSize, const int nStep)
{
    /* One thread block per data stream.
     * Stream index: blockIdx.x
     * Number of streams: gridDim.x
     */

    __shared__ V _v[THREAD_PER_BLOCK];
    __shared__ int _i[THREAD_PER_BLOCK];

    V v_tmp;
    int i_tmp, i_max;
    int i;
    int _prevIdxMax;

    x += blockIdx.x * ldx;
    y += blockIdx.x * ldy;
    idxMax += blockIdx.x;

    if(threadIdx.x == 0)
    {
        _prevIdxMax = prevIdxMax[blockIdx.x];
    }

    __syncthreads();

    for(int curStep = 0; curStep < nStep; curStep++)
    {
        /* Sequential reduction(max) */
        v_tmp = threadIdx.x < layerSize ? x[threadIdx.x] : (FLOAT) 0;
        i_tmp = threadIdx.x < layerSize ? (INT) threadIdx.x : (INT) -1;

        #pragma unroll
        for(i = threadIdx.x + blockDim.x; i < layerSize; i += blockDim.x)
        {
            int comp = (v_tmp > x[i]);

            v_tmp = comp * v_tmp + (1 - comp) * x[i];
            i_tmp = comp * i_tmp + (1 - comp) * i;
        }

        _v[threadIdx.x] = v_tmp;
        _i[threadIdx.x] = i_tmp;

        __syncthreads();

        /* Parallel reduction(max) */
        i = (blockDim.x >> 1);

        for(; i > 0; i >>= 1)
        {
            if(threadIdx.x < i && threadIdx.x + i < layerSize)
            {
                FLOAT v_comp = _v[threadIdx.x + i];
                int i_comp = _i[threadIdx.x + i];

                int comp = (v_tmp > v_comp);

                v_tmp = comp * v_tmp + (1 - comp) * v_comp;
                i_tmp = comp * i_tmp + (1 - comp) * i_comp;

                _v[threadIdx.x] = v_tmp;
                _i[threadIdx.x] = i_tmp;
            }

            __syncthreads();
        }

        //v_max = _v[0];
        i_max = _i[0];

        __syncthreads();


        /* Sequential update */
        #pragma unroll
        for(i = threadIdx.x; i < layerSize - 1; i += blockDim.x)
        {
            y[i] = (i == i_max);
        }

        /* Update clock signal */
        if(threadIdx.x == 0)
        {
            y[layerSize - 1] = (i_max != layerSize - 1) && (i_max != _prevIdxMax);
            *idxMax = i_max;
            _prevIdxMax = i_max;
        }

        __syncthreads();

        /* Next time step */
        x += gridDim.x * ldx;
        y += gridDim.x * ldy;
        idxMax += gridDim.x;
    }
}


template<class T>
static __global__ void FuncBoundRangeKernel(const T *x, const int ldx,
        T *y, const int ldy, const T _min, const T _max,
        const int nRows, const int nCols)
{
    int thdIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int col = thdIdx / nRows;
    int row = thdIdx - col * nRows;
    int xIdx = col * ldx + row;
    int yIdx = col * ldy + row;

    if(thdIdx >= nRows * nCols) return;

    y[yIdx] = min(_max, max(_min, x[xIdx]));
}


template<class T>
static __global__ void FuncSigmoidDerivKernel(const T *x, const int ldx,
        T *y, const int ldy, const int nRows, const int nCols)
{
    int thdIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int col = thdIdx / nRows;
    int row = thdIdx - col * nRows;
    int xIdx = col * ldx + row;
    int yIdx = col * ldy + row;

    if(thdIdx >= nRows * nCols) return;

    T v = x[xIdx];
    y[yIdx] = v * ((T)1 - v);
}


template<class T>
static __global__ void FuncTanhDerivKernel(const T *x, const int ldx,
        T *y, const int ldy, const int nRows, const int nCols)
{
    int thdIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int col = thdIdx / nRows;
    int row = thdIdx - col * nRows;
    int xIdx = col * ldx + row;
    int yIdx = col * ldy + row;

    if(thdIdx >= nRows * nCols) return;

    T v = x[xIdx];
    y[yIdx] = ((T)1 - v) * ((T)1 + v);
}


template<class T>
static __global__ void FuncSoftplusDerivKernel(const T *x, const int ldx,
        T *y, const int ldy, const int nRows, const int nCols)
{
    int thdIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int col = thdIdx / nRows;
    int row = thdIdx - col * nRows;
    int xIdx = col * ldx + row;
    int yIdx = col * ldy + row;

    if(thdIdx >= nRows * nCols) return;

    y[yIdx] = (T)1 - _exp<T>(-x[xIdx]);
}


template<class T>
static __global__ void FuncRectLinearDerivKernel(const T *x, const int ldx,
        T *y, const int ldy, const int nRows, const int nCols)
{
    int thdIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int col = thdIdx / nRows;
    int row = thdIdx - col * nRows;
    int xIdx = col * ldx + row;
    int yIdx = col * ldy + row;

    if(thdIdx >= nRows * nCols) return;

    //y[yIdx] = (T)(x[xIdx] > (T)0);
    /* Leaky */
    y[yIdx] = (T)0.01 + (T)0.99 * (T)(x[xIdx] > (T)0);
}


template<class I, class V>
static __global__ void OneHotEncodeKernel(const I *index, V *vector, const int n)
{
    int elemIdx, batchIdx;
    __shared__ I _index;

    elemIdx = blockIdx.x * blockDim.x + threadIdx.x;
    batchIdx = blockIdx.y;

    if(elemIdx >= n) return;

    if(threadIdx.x == 0)
    {
        _index = index[batchIdx];
    }

    __syncthreads();

    vector[batchIdx * n + elemIdx] = (V)(elemIdx == _index);
}


template<class T>
static __global__ void GenerateDropoutMaskKernel(T *mask, const T *uniformDist,
        const int n, const T dropoutRate)
{
    int idx;

    idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= n) return;

    mask[idx] = (T)(uniformDist[idx] >= dropoutRate) / ((T)1 - dropoutRate);
}


template<class T>
static __global__ void RmspropKernel(T *newDerivs, const T *derivs, T *msDeriv,
        const T decayRate, const int n)
{
    unsigned long idx;
    T ms, rms, deriv;

    idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= n) return;

    ms = msDeriv[idx];
    deriv = derivs[idx];

    T bound = _sqrt<T>((T)1 / ((T)1 - decayRate));

    ms = decayRate * ms + ((T)1 - decayRate) * deriv * deriv;
    rms = _sqrt<T>(ms) + (T)1e-20;

    newDerivs[idx] = min(bound, max(-bound, deriv / rms));
    msDeriv[idx] = ms;
}


template<class T>
static __global__ void AdadeltaKernel(T *deltas, const T *derivs, T *msDeriv, T *msDelta,
        const T learningRate, const T decayRate, const int n)
{
    unsigned int idx;
    T _msDelta, rmsDelta;
    T _msDeriv, rmsDeriv;
    T deriv, delta;

    const T bound = (T)10;

    idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= n) return;

    _msDeriv = msDeriv[idx];
    _msDelta = msDelta[idx];
    deriv = derivs[idx];

    _msDeriv = decayRate * _msDeriv + ((T)1 - decayRate) * deriv * deriv;
    rmsDeriv = _sqrt<T>(_msDeriv) + (T)1e-20;

    rmsDelta = _sqrt<T>(_msDelta + learningRate * learningRate);

    delta = rmsDelta * min(bound, max(-bound, deriv / rmsDeriv));

    _msDelta = decayRate * _msDelta + ((T)1 - decayRate) * delta * delta;

    deltas[idx] = delta;
    msDeriv[idx] = _msDeriv;
    msDelta[idx] = _msDelta;
}


template<class T>
void MatSet(T *_x, const unsigned long ldx, const T val,
        const unsigned long nRows, const unsigned long nCols, const cudaStream_t stream)
{
    dim3 dimGrid((nRows * nCols + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK);
    dim3 dimBlock(THREAD_PER_BLOCK);

    MatSetKernel<T><<<dimGrid, dimBlock, 0, stream>>>(_x, ldx, val, nRows, nCols);
}


template<class T>
void MatElemMult(const T *_x, const unsigned long ldx,
        const T *_y, const unsigned long ldy,
        T *_z, const unsigned long ldz,
        const unsigned long nRows, const unsigned long nCols,
        const cudaStream_t stream)
{
    dim3 dimGrid((nRows * nCols + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK);
    dim3 dimBlock(THREAD_PER_BLOCK);

    MatElemMultKernel<T><<<dimGrid, dimBlock, 0, stream>>>(_x, ldx, _y, ldy, _z, ldz, nRows, nCols);
}


template<class T>
void MatAdd(const T *_x, const unsigned long ldx,
        const T *_y, const unsigned long ldy,
        T *_z, const unsigned long ldz,
        const unsigned long nRows, const unsigned long nCols,
        const cudaStream_t stream)
{
    dim3 dimGrid((nRows * nCols + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK);
    dim3 dimBlock(THREAD_PER_BLOCK);

    MatAddKernel<T><<<dimGrid, dimBlock, 0, stream>>>(_x, ldx, _y, ldy, _z, ldz, nRows, nCols);
}


template<class T>
void MatSub(const T *_x, const unsigned long ldx,
        const T *_y, const unsigned long ldy,
        T *_z, const unsigned long ldz,
        const unsigned long nRows, const unsigned long nCols,
        const cudaStream_t stream)
{
    dim3 dimGrid((nRows * nCols + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK);
    dim3 dimBlock(THREAD_PER_BLOCK);

    MatSubKernel<T><<<dimGrid, dimBlock, 0, stream>>>(_x, ldx, _y, ldy, _z, ldz, nRows, nCols);
}


template<class T>
void MatAddToDiag(T *_x, const T val, const unsigned long shift,
        const unsigned long nRows, const cudaStream_t stream)
{
    dim3 dimGrid((nRows + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK);
    dim3 dimBlock(THREAD_PER_BLOCK);

    MatAddToDiagKernel<T><<<dimGrid, dimBlock, 0, stream>>>(_x, val, shift, nRows);
}


template<class T>
void MatMakeTri(T *_x, const bool upper, const unsigned long nRows, cudaStream_t stream)
{
    dim3 dimGrid((nRows * nRows + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK);
    dim3 dimBlock(THREAD_PER_BLOCK);

    MatMakeTriKernel<T><<<dimGrid, dimBlock, 0, stream>>>(_x, upper, nRows);
}


template<class T>
void FuncSigmoid(const T *_x, const unsigned long ldx,
        T *_y, const unsigned long ldy,
        const unsigned long nRows, const unsigned long nCols,
        const cudaStream_t stream)
{
    dim3 dimGrid((nRows * nCols + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK);
    dim3 dimBlock(THREAD_PER_BLOCK);

    FuncSigmoidKernel<T><<<dimGrid, dimBlock, 0, stream>>>(_x, ldx, _y, ldy, nRows, nCols);
}


template<class T>
void FuncTanh(const T *_x, const unsigned long ldx,
        T *_y, const unsigned long ldy,
        const unsigned long nRows, const unsigned long nCols,
        const cudaStream_t stream)
{
    dim3 dimGrid((nRows * nCols + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK);
    dim3 dimBlock(THREAD_PER_BLOCK);

    FuncTanhKernel<T><<<dimGrid, dimBlock, 0, stream>>>(_x, ldx, _y, ldy, nRows, nCols);
}


template<class T>
void FuncSoftplus(const T *_x, const unsigned long ldx,
        T *_y, const unsigned long ldy,
        const unsigned long nRows, const unsigned long nCols,
        const cudaStream_t stream)
{
    dim3 dimGrid((nRows * nCols + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK);
    dim3 dimBlock(THREAD_PER_BLOCK);

    FuncSoftplusKernel<T><<<dimGrid, dimBlock, 0, stream>>>(_x, ldx, _y, ldy, nRows, nCols);
}


template<class T>
void FuncRectLinear(const T *_x, const unsigned long ldx,
        T *_y, const unsigned long ldy,
        const unsigned long nRows, const unsigned long nCols,
        const cudaStream_t stream)
{
    dim3 dimGrid((nRows * nCols + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK);
    dim3 dimBlock(THREAD_PER_BLOCK);

    FuncRectLinearKernel<T><<<dimGrid, dimBlock, 0, stream>>>(_x, ldx, _y, ldy, nRows, nCols);
}


template<class T>
void FuncSoftmax(const T *_x, const unsigned long ldx,
        T *_y, const unsigned long ldy,
        const unsigned long layerSize, const unsigned long batchSize,
        const cudaStream_t stream)
{
    dim3 dimGrid(batchSize);
    dim3 dimBlock(THREAD_PER_BLOCK);

    FuncSoftmaxKernel<T><<<dimGrid, dimBlock, 0, stream>>>(_x, ldx, _y, ldy, layerSize);
}


template<class I, class V>
void FuncCTCDecode(const V *_x, const unsigned long ldx,
        V *_y, const unsigned long ldy, const I *_prevIdxMax, I *_idxMax,
        const unsigned long layerSize, const unsigned long nStep,
        const unsigned long nStream, const cudaStream_t stream)
{
    dim3 dimGrid(nStream);
    dim3 dimBlock(THREAD_PER_BLOCK);

    FuncCTCDecodeKernel<I, V><<<dimGrid, dimBlock, 0, stream>>>(_x, ldx, _y, ldy, _prevIdxMax, _idxMax, layerSize, nStep);
}


template<class T>
void FuncBoundRange(const T *_x, const unsigned long ldx,
        T *_y, const unsigned long ldy,
        const T min, const T max,
        const unsigned long nRows, const unsigned long nCols,
        const cudaStream_t stream)
{
    dim3 dimGrid((nRows * nCols + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK);
    dim3 dimBlock(THREAD_PER_BLOCK);

    FuncBoundRangeKernel<T><<<dimGrid, dimBlock, 0, stream>>>(_x, ldx, _y, ldy, min, max, nRows, nCols);
}


template<class T>
void FuncSigmoidDeriv(const T *_x, const unsigned long ldx,
        T *_y, const unsigned long ldy,
        const unsigned long nRows, const unsigned long nCols,
        const cudaStream_t stream)
{
    dim3 dimGrid((nRows * nCols + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK);
    dim3 dimBlock(THREAD_PER_BLOCK);

    FuncSigmoidDerivKernel<T><<<dimGrid, dimBlock, 0, stream>>>(_x, ldx, _y, ldy, nRows, nCols);
}


template<class T>
void FuncTanhDeriv(const T *_x, const unsigned long ldx,
        T *_y, const unsigned long ldy,
        const unsigned long nRows, const unsigned long nCols,
        const cudaStream_t stream)
{
    dim3 dimGrid((nRows * nCols + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK);
    dim3 dimBlock(THREAD_PER_BLOCK);

    FuncTanhDerivKernel<T><<<dimGrid, dimBlock, 0, stream>>>(_x, ldx, _y, ldy, nRows, nCols);
}


template<class T>
void FuncSoftplusDeriv(const T *_x, const unsigned long ldx,
        T *_y, const unsigned long ldy,
        const unsigned long nRows, const unsigned long nCols,
        const cudaStream_t stream)
{
    dim3 dimGrid((nRows * nCols + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK);
    dim3 dimBlock(THREAD_PER_BLOCK);

    FuncSoftplusDerivKernel<T><<<dimGrid, dimBlock, 0, stream>>>(_x, ldx, _y, ldy, nRows, nCols);
}


template<class T>
void FuncRectLinearDeriv(const T *_x, const unsigned long ldx,
        T *_y, const unsigned long ldy,
        const unsigned long nRows, const unsigned long nCols,
        const cudaStream_t stream)
{
    dim3 dimGrid((nRows * nCols + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK);
    dim3 dimBlock(THREAD_PER_BLOCK);

    FuncRectLinearDerivKernel<T><<<dimGrid, dimBlock, 0, stream>>>(_x, ldx, _y, ldy, nRows, nCols);
}


template<class I, class V>
void OneHotEncode(const I *_index, V *_vector, const unsigned long n,
        const unsigned long batchSize, const cudaStream_t stream)
{
    dim3 dimGrid((n + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK, batchSize);
    dim3 dimBlock(THREAD_PER_BLOCK);

    OneHotEncodeKernel<I, V><<<dimGrid, dimBlock, 0, stream>>>(_index, _vector, n);
}


template<class T>
void GenerateDropoutMask(T *_mask, const T *_uniformDist, const unsigned long n,
        const T dropoutRate, const cudaStream_t stream)
{
    dim3 dimGrid((n + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK);
    dim3 dimBlock(THREAD_PER_BLOCK);

    GenerateDropoutMaskKernel<T><<<dimGrid, dimBlock, 0, stream>>>(_mask, _uniformDist, n, dropoutRate);
}


template<class T>
void Rmsprop(T *_newDerivs, const T *_derivs, T *_msDeriv, const T decayRate,
        const unsigned long n, const cudaStream_t stream)
{
    dim3 dimGrid((n + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK);
    dim3 dimBlock(THREAD_PER_BLOCK);

    RmspropKernel<T><<<dimGrid, dimBlock, 0, stream>>>(_newDerivs, _derivs, _msDeriv, decayRate, n);
}


template<class T>
void Adadelta(T *_deltas, const T *_derivs, T *_msDeriv, T *_msDelta,
        const T learningRate, const T decayRate, const unsigned long n,
        const cudaStream_t stream)
{
    dim3 dimGrid((n + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK);
    dim3 dimBlock(THREAD_PER_BLOCK);

    AdadeltaKernel<T><<<dimGrid, dimBlock, 0, stream>>>(_deltas, _derivs, _msDeriv, _msDelta, learningRate, decayRate, n);
}


template void MatSet<float>(float *_x, const unsigned long ldx, const float val,
        const unsigned long nRows, const unsigned long nCols, const cudaStream_t stream);
template void MatSet<double>(double *_x, const unsigned long ldx, const double val,
        const unsigned long nRows, const unsigned long nCols, const cudaStream_t stream);
template void MatSet<int>(int *_x, const unsigned long ldx, const int val,
        const unsigned long nRows, const unsigned long nCols, const cudaStream_t stream);
template void MatSet<long>(long *_x, const unsigned long ldx, const long val,
        const unsigned long nRows, const unsigned long nCols, const cudaStream_t stream);
template void MatSet<long long>(long long *_x, const unsigned long ldx, const long long val,
        const unsigned long nRows, const unsigned long nCols, const cudaStream_t stream);

template void MatElemMult<float>(const float *_x, const unsigned long ldx,
        const float *_y, const unsigned long ldy,
        float *_z, const unsigned long ldz,
        const unsigned long nRows, const unsigned long nCols,
        const cudaStream_t stream);
template void MatElemMult<double>(const double *_x, const unsigned long ldx,
        const double *_y, const unsigned long ldy,
        double *_z, const unsigned long ldz,
        const unsigned long nRows, const unsigned long nCols,
        const cudaStream_t stream);

template void MatAdd<float>(const float *_x, const unsigned long ldx,
        const float *_y, const unsigned long ldy,
        float *_z, const unsigned long ldz,
        const unsigned long nRows, const unsigned long nCols,
        const cudaStream_t stream);
template void MatAdd<double>(const double *_x, const unsigned long ldx,
        const double *_y, const unsigned long ldy,
        double *_z, const unsigned long ldz,
        const unsigned long nRows, const unsigned long nCols,
        const cudaStream_t stream);

template void MatSub<float>(const float *_x, const unsigned long ldx,
        const float *_y, const unsigned long ldy,
        float *_z, const unsigned long ldz,
        const unsigned long nRows, const unsigned long nCols,
        const cudaStream_t stream);
template void MatSub<double>(const double *_x, const unsigned long ldx,
        const double *_y, const unsigned long ldy,
        double *_z, const unsigned long ldz,
        const unsigned long nRows, const unsigned long nCols,
        const cudaStream_t stream);

template void MatAddToDiag<float>(float *_x, const float val, const unsigned long shift,
        const unsigned long nRows, const cudaStream_t stream);
template void MatAddToDiag<double>(double *_x, const double val, const unsigned long shift,
        const unsigned long nRows, const cudaStream_t stream);

template void MatMakeTri<float>(float *_x, const bool upper,
        const unsigned long nRows, cudaStream_t stream);
template void MatMakeTri<double>(double *_x, const bool upper,
        const unsigned long nRows, cudaStream_t stream);

template void FuncSigmoid<float>(const float *_x, const unsigned long ldx,
        float *_y, const unsigned long ldy,
        const unsigned long nRows, const unsigned long nCols,
        const cudaStream_t stream);
template void FuncSigmoid<double>(const double *_x, const unsigned long ldx,
        double *_y, const unsigned long ldy,
        const unsigned long nRows, const unsigned long nCols,
        const cudaStream_t stream);

template void FuncTanh<float>(const float *_x, const unsigned long ldx,
        float *_y, const unsigned long ldy,
        const unsigned long nRows, const unsigned long nCols,
        const cudaStream_t stream);
template void FuncTanh<double>(const double *_x, const unsigned long ldx,
        double *_y, const unsigned long ldy,
        const unsigned long nRows, const unsigned long nCols,
        const cudaStream_t stream);

template void FuncSoftplus<float>(const float *_x, const unsigned long ldx,
        float *_y, const unsigned long ldy,
        const unsigned long nRows, const unsigned long nCols,
        const cudaStream_t stream);
template void FuncSoftplus<double>(const double *_x, const unsigned long ldx,
        double *_y, const unsigned long ldy,
        const unsigned long nRows, const unsigned long nCols,
        const cudaStream_t stream);

template void FuncRectLinear<float>(const float *_x, const unsigned long ldx,
        float *_y, const unsigned long ldy,
        const unsigned long nRows, const unsigned long nCols,
        const cudaStream_t stream);
template void FuncRectLinear<double>(const double *_x, const unsigned long ldx,
        double *_y, const unsigned long ldy,
        const unsigned long nRows, const unsigned long nCols,
        const cudaStream_t stream);

template void FuncSoftmax<float>(const float *_x, const unsigned long ldx,
        float *_y, const unsigned long ldy,
        const unsigned long layerSize, const unsigned long batchSize,
        const cudaStream_t stream);
template void FuncSoftmax<double>(const double *_x, const unsigned long ldx,
        double *_y, const unsigned long ldy,
        const unsigned long layerSize, const unsigned long batchSize,
        const cudaStream_t stream);

template void FuncCTCDecode<int, float>(const float *_x, const unsigned long ldx,
        float *_y, const unsigned long ldy, const int *_prevIdxMax, int *_idxMax,
        const unsigned long layerSize, const unsigned long nStep,
        const unsigned long nStream, const cudaStream_t stream);
template void FuncCTCDecode<long, float>(const float *_x, const unsigned long ldx,
        float *_y, const unsigned long ldy, const long *_prevIdxMax, long *_idxMax,
        const unsigned long layerSize, const unsigned long nStep,
        const unsigned long nStream, const cudaStream_t stream);
template void FuncCTCDecode<long long, float>(const float *_x, const unsigned long ldx,
        float *_y, const unsigned long ldy, const long long *_prevIdxMax, long long *_idxMax,
        const unsigned long layerSize, const unsigned long nStep,
        const unsigned long nStream, const cudaStream_t stream);
template void FuncCTCDecode<int, double>(const double *_x, const unsigned long ldx,
        double *_y, const unsigned long ldy, const int *_prevIdxMax, int *_idxMax,
        const unsigned long layerSize, const unsigned long nStep,
        const unsigned long nStream, const cudaStream_t stream);
template void FuncCTCDecode<long, double>(const double *_x, const unsigned long ldx,
        double *_y, const unsigned long ldy, const long *_prevIdxMax, long *_idxMax,
        const unsigned long layerSize, const unsigned long nStep,
        const unsigned long nStream, const cudaStream_t stream);
template void FuncCTCDecode<long long, double>(const double *_x, const unsigned long ldx,
        double *_y, const unsigned long ldy, const long long *_prevIdxMax, long long *_idxMax,
        const unsigned long layerSize, const unsigned long nStep,
        const unsigned long nStream, const cudaStream_t stream);

template void FuncBoundRange<float>(const float *_x, const unsigned long ldx,
        float *_y, const unsigned long ldy,
        const float min, const float max,
        const unsigned long nRows, const unsigned long nCols,
        const cudaStream_t stream);
template void FuncBoundRange<double>(const double *_x, const unsigned long ldx,
        double *_y, const unsigned long ldy,
        const double min, const double max,
        const unsigned long nRows, const unsigned long nCols,
        const cudaStream_t stream);

template void FuncSigmoidDeriv<float>(const float *_x, const unsigned long ldx,
        float *_y, const unsigned long ldy,
        const unsigned long nRows, const unsigned long nCols,
        const cudaStream_t stream);
template void FuncSigmoidDeriv<double>(const double *_x, const unsigned long ldx,
        double *_y, const unsigned long ldy,
        const unsigned long nRows, const unsigned long nCols,
        const cudaStream_t stream);

template void FuncTanhDeriv<float>(const float *_x, const unsigned long ldx,
        float *_y, const unsigned long ldy,
        const unsigned long nRows, const unsigned long nCols,
        const cudaStream_t stream);
template void FuncTanhDeriv<double>(const double *_x, const unsigned long ldx,
        double *_y, const unsigned long ldy,
        const unsigned long nRows, const unsigned long nCols,
        const cudaStream_t stream);

template void FuncSoftplusDeriv<float>(const float *_x, const unsigned long ldx,
        float *_y, const unsigned long ldy,
        const unsigned long nRows, const unsigned long nCols,
        const cudaStream_t stream);
template void FuncSoftplusDeriv<double>(const double *_x, const unsigned long ldx,
        double *_y, const unsigned long ldy,
        const unsigned long nRows, const unsigned long nCols,
        const cudaStream_t stream);

template void FuncRectLinearDeriv<float>(const float *_x, const unsigned long ldx,
        float *_y, const unsigned long ldy,
        const unsigned long nRows, const unsigned long nCols,
        const cudaStream_t stream);
template void FuncRectLinearDeriv<double>(const double *_x, const unsigned long ldx,
        double *_y, const unsigned long ldy,
        const unsigned long nRows, const unsigned long nCols,
        const cudaStream_t stream);

template void OneHotEncode<int, float>(const int *_index, float *_vector, const unsigned long n,
        const unsigned long batchSize, const cudaStream_t stream);
template void OneHotEncode<long, float>(const long *_index, float *_vector, const unsigned long n,
        const unsigned long batchSize, const cudaStream_t stream);
template void OneHotEncode<long long, float>(const long long *_index, float *_vector, const unsigned long n,
        const unsigned long batchSize, const cudaStream_t stream);
template void OneHotEncode<int, double>(const int *_index, double *_vector, const unsigned long n,
        const unsigned long batchSize, const cudaStream_t stream);
template void OneHotEncode<long, double>(const long *_index, double *_vector, const unsigned long n,
        const unsigned long batchSize, const cudaStream_t stream);
template void OneHotEncode<long long, double>(const long long *_index, double *_vector, const unsigned long n,
        const unsigned long batchSize, const cudaStream_t stream);

template void GenerateDropoutMask<float>(float *_mask, const float *_uniformDist, const unsigned long n,
        const float dropoutRate, const cudaStream_t stream);
template void GenerateDropoutMask<double>(double *_mask, const double *_uniformDist, const unsigned long n,
        const double dropoutRate, const cudaStream_t stream);

template void Rmsprop<float>(float *_newDerivs, const float *_derivs, float *_msDeriv, const float decayRate,
        const unsigned long n, const cudaStream_t stream);
template void Rmsprop<double>(double *_newDerivs, const double *_derivs, double *_msDeriv, const double decayRate,
        const unsigned long n, const cudaStream_t stream);

template void Adadelta<float>(float *_deltas, const float *_derivs, float *_msDeriv, float *_msDelta,
        const float learningRate, const float decayRate, const unsigned long n,
        const cudaStream_t stream);
template void Adadelta<double>(double *_deltas, const double *_derivs, double *_msDeriv, double *_msDelta,
        const double learningRate, const double decayRate, const unsigned long n,
        const cudaStream_t stream);

}

}

