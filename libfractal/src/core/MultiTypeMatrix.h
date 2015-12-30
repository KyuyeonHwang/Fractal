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


#ifndef FRACTAL_MULTITYPEMATRIX_H_
#define FRACTAL_MULTITYPEMATRIX_H_


#include "Matrix.h"
#include "Engine.h"
#include "FractalCommon.h"


namespace fractal
{

class MultiTypeMatrix
{
public:

    enum DataType {DATATYPE_UNDEFINED, DATATYPE_FLOAT, DATATYPE_INT};

    MultiTypeMatrix(const MultiTypeMatrix &) = delete;
    MultiTypeMatrix(MultiTypeMatrix &&) = delete;

    MultiTypeMatrix();
    MultiTypeMatrix(MultiTypeMatrix &src, const unsigned long a1, const unsigned long a2);
    virtual ~MultiTypeMatrix();

    void *GetMatrix();

    void SetDataType(const DataType dataType);
    inline const DataType GetDataType() { return dataType; }
    void SetEngine(Engine *engine);
    void Resize(const unsigned long nRows, const unsigned long nCols);

    void Swap(MultiTypeMatrix& target);

    void Clear();


protected:

    DataType dataType;

    void *matrix;
};

}


#endif /* FRACTAL_MULTITYPEMATRIX_H_ */

