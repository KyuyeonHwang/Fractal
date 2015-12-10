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


#ifndef FRACTAL_MATRIX_H_
#define FRACTAL_MATRIX_H_

#include <mutex>
#include <vector>
#include <string>

#include "FractalCommon.h"

namespace fractal
{

class PStream;
class Engine;
class Mem;

/* Column-major order */

template<class T>
class Matrix
{
public:
    Matrix(const unsigned long nRows = 0, const unsigned long nCols = 1);

    Matrix(Matrix<T> &A, const unsigned long c1, const unsigned long c2);

    Matrix(Matrix<T> &A, const unsigned long r1, const unsigned long r2,
            const unsigned long c1, const unsigned long c2);

    virtual ~Matrix();

    void SetEngine(Engine *engine);

    inline const unsigned long GetOffset() const { return offset; }
    inline const unsigned long GetLeadingDim() const { return leadingDim; }

    inline const unsigned long GetNumRows() const { return nRows; }
    inline const unsigned long GetNumCols() const { return nCols; }

    inline Engine *GetEngine() const { return engine; }
    inline Mem *GetMem() { return mem; }

    T *GetPtrForReadWrite(PStream &stream);
    T *GetPtrForWrite(PStream &stream);
    void FinishWrite(PStream &stream);

    T *const GetHostData();
    void HostPush();
    void HostPull(PStream &stream);

    void Resize(const unsigned long nRows, const unsigned long nCols);
    void Link(Matrix<T> &src);
    void Unlink();

    void Import(const std::vector<T> &vec, PStream &stream);
    void Import(const Matrix<T> &mat, PStream &stream);
    void Export(std::vector<T> &vec, PStream &stream) const;
    void Export(Matrix<T> &mat, PStream &stream) const;

    void Swap(Matrix<T> &target);

    inline void Lock() { mtx.lock(); }
    inline void Unlock() { mtx.unlock(); }

    void Save(const std::string &filename);
    void Load(const std::string &filename);


protected:
    Matrix(const Matrix<T> &);

    void Malloc();
    void Clear();

    Mem *mem;
    unsigned long offset;
    unsigned long nRows, nCols;
    unsigned long leadingDim;
    bool isSub;

    std::recursive_mutex mtx;
    Engine *engine;
};

}

#endif /* FRACTAL_MATRIX_H_ */

