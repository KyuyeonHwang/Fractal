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


#ifndef FRACTAL_CUDAKERNELS_H_
#define FRACTAL_CUDAKERNELS_H_


#include <cuda_runtime.h>
#include "FractalCommon.h"

namespace fractal
{

namespace cudaKernels
{
    template<class T>
    void MatSet(T *_x, const unsigned long ldx, const T val,
            const unsigned long nRows, const unsigned long nCols, const cudaStream_t stream);

    /* _z = _x .* _y */
    template<class T>
    void MatElemMult(const T *_x, const unsigned long ldx,
            const T *_y, const unsigned long ldy,
            T *_z, const unsigned long ldz,
            const unsigned long nRows, const unsigned long nCols,
            const cudaStream_t stream);

    /* _y = a * _x + b */
    template<class T>
    void MatScale(const T *_x, const unsigned long ldx,
            T *_y, const unsigned long ldy,
            const T a, const T b,
            const unsigned long nRows, const unsigned long nCols,
            const cudaStream_t stream);

    /* _z = _x + _y */
    template<class T>
    void MatAdd(const T *_x, const unsigned long ldx,
            const T *_y, const unsigned long ldy,
            T *_z, const unsigned long ldz,
            const unsigned long nRows, const unsigned long nCols,
            const cudaStream_t stream);

    /* _z = _x - _y */
    template<class T>
    void MatSub(const T *_x, const unsigned long ldx,
            const T *_y, const unsigned long ldy,
            T *_z, const unsigned long ldz,
            const unsigned long nRows, const unsigned long nCols,
            const cudaStream_t stream);

    template<class T>
    void MatAddToDiag(T *_x, const T val, const unsigned long shift,
            const unsigned long nRows, const cudaStream_t stream);

    template<class T>
    void MatMakeTri(T *_x, const bool upper, const unsigned long nRows, cudaStream_t stream);

    template<class T>
    void FuncSigmoid(const T *_x, const unsigned long ldx,
            T *_y, const unsigned long ldy,
            const unsigned long nRows, const unsigned long nCols,
            const cudaStream_t stream);

    template<class T>
    void FuncTanh(const T *_x, const unsigned long ldx,
            T *_y, const unsigned long ldy,
            const unsigned long nRows, const unsigned long nCols,
            const cudaStream_t stream);

    template<class T>
    void FuncSoftplus(const T *_x, const unsigned long ldx,
            T *_y, const unsigned long ldy,
            const unsigned long nRows, const unsigned long nCols,
            const cudaStream_t stream);

    template<class T>
    void FuncRectLinear(const T *_x, const unsigned long ldx,
            T *_y, const unsigned long ldy,
            const unsigned long nRows, const unsigned long nCols,
            const cudaStream_t stream);

    template<class T>
    void FuncSoftmax(const T *_x, const unsigned long ldx,
            T *_y, const unsigned long ldy,
            const unsigned long layerSize, const unsigned long batchSize,
            const cudaStream_t stream);

    template<class I, class V>
    void FuncCTCDecode(const V *_x, const unsigned long ldx,
            V *_y, const unsigned long ldy, const I *_prevIdxMax, I *_idxMax,
            const unsigned long layerSize, const unsigned long nStep,
            const unsigned long nStream, const cudaStream_t stream);

    template<class T>
    void FuncBoundRange(const T *_x, const unsigned long ldx,
            T *_y, const unsigned long ldy,
            const T min, const T max,
            const unsigned long nRows, const unsigned long nCols,
            const cudaStream_t stream);

    template<class T>
    void FuncSigmoidDeriv(const T *_x, const unsigned long ldx,
            T *_y, const unsigned long ldy,
            const unsigned long nRows, const unsigned long nCols,
            const cudaStream_t stream);

    template<class T>
    void FuncTanhDeriv(const T *_x, const unsigned long ldx,
            T *_y, const unsigned long ldy,
            const unsigned long nRows, const unsigned long nCols,
            const cudaStream_t stream);

    template<class T>
    void FuncSoftplusDeriv(const T *_x, const unsigned long ldx,
            T *_y, const unsigned long ldy,
            const unsigned long nRows, const unsigned long nCols,
            const cudaStream_t stream);

    template<class T>
    void FuncRectLinearDeriv(const T *_x, const unsigned long ldx,
            T *_y, const unsigned long ldy,
            const unsigned long nRows, const unsigned long nCols,
            const cudaStream_t stream);

    template<class I, class V>
    void OneHotEncode(const I *_index, V *_vector, const unsigned long n,
            const unsigned long batchSize, const cudaStream_t stream);

    template<class T>
    void GenerateDropoutMask(T *_mask, const T *_uniformDist, const unsigned long n,
            const T dropoutRate, const cudaStream_t stream);

    template<class T>
    void Rmsprop(T *_newDerivs, const T *_derivs, T *_msDeriv, const T decayRate,
            const unsigned long n, const cudaStream_t stream);

    template<class T>
    void Adadelta(T *_deltas, const T *_derivs, T *_msDeriv, T *_msDelta,
            const T learningRate, const T decayRate, const unsigned long n,
            const cudaStream_t stream);
}

}

#endif /* FRACTAL_CUDAKERNELS_H_ */

