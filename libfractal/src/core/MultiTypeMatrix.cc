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


#include "MultiTypeMatrix.h"


namespace fractal
{


MultiTypeMatrix::MultiTypeMatrix()
{
    dataType = DATATYPE_UNDEFINED;
    matrix = NULL;
}


MultiTypeMatrix::MultiTypeMatrix(MultiTypeMatrix &src, const unsigned long a1, const unsigned long a2)
{
    void *srcMat;

    dataType = src.GetDataType();

    srcMat = src.GetMatrix();

    switch(dataType)
    {
        case DATATYPE_UNDEFINED:
            verify(dataType != DATATYPE_UNDEFINED);
            break;

        case DATATYPE_FLOAT:
            matrix = new Matrix<FLOAT>(*reinterpret_cast<Matrix<FLOAT> *>(srcMat), a1, a2);
            break;

        case DATATYPE_INT:
            matrix = new Matrix<INT>(*reinterpret_cast<Matrix<INT> *>(srcMat), a1, a2);
            break;
    }
}


MultiTypeMatrix::~MultiTypeMatrix()
{
    Clear();
}


void *MultiTypeMatrix::GetMatrix()
{
    verify(matrix != NULL);

    return matrix;
}


void MultiTypeMatrix::SetDataType(const DataType dataType)
{
    if(dataType == this->dataType) return;

    Engine *engine;
    unsigned long nRows, nCols;

    engine = NULL;
    nRows = 0;
    nCols = 0;

    switch(this->dataType)
    {
        case DATATYPE_UNDEFINED:
            break;

        case DATATYPE_FLOAT:
            engine = reinterpret_cast<Matrix<FLOAT> *>(matrix)->GetEngine();
            nRows = reinterpret_cast<Matrix<FLOAT> *>(matrix)->GetNumRows();
            nCols = reinterpret_cast<Matrix<FLOAT> *>(matrix)->GetNumCols();
            break;

        case DATATYPE_INT:
            engine = reinterpret_cast<Matrix<INT> *>(matrix)->GetEngine();
            nRows = reinterpret_cast<Matrix<INT> *>(matrix)->GetNumRows();
            nCols = reinterpret_cast<Matrix<INT> *>(matrix)->GetNumCols();
            break;
    }

    switch(dataType)
    {
        case DATATYPE_UNDEFINED:
            Clear();
            break;

        case DATATYPE_FLOAT:
            matrix = new Matrix<FLOAT>();
            break;

        case DATATYPE_INT:
            matrix = new Matrix<INT>();
            break;
    }

    if(this->dataType != DATATYPE_UNDEFINED && dataType != DATATYPE_UNDEFINED)
    {
        SetEngine(engine);
        Resize(nRows, nCols);
    }

    this->dataType = dataType;
}


void MultiTypeMatrix::SetEngine(Engine *engine)
{
    switch(dataType)
    {
        case DATATYPE_UNDEFINED:
            verify(dataType != DATATYPE_UNDEFINED);
            break;

        case DATATYPE_FLOAT:
            reinterpret_cast<Matrix<FLOAT> *>(matrix)->SetEngine(engine);
            break;

        case DATATYPE_INT:
            reinterpret_cast<Matrix<INT> *>(matrix)->SetEngine(engine);
            break;
    }
}


void MultiTypeMatrix::Resize(const unsigned long nRows, const unsigned long nCols)
{
    switch(dataType)
    {
        case DATATYPE_UNDEFINED:
            verify(dataType != DATATYPE_UNDEFINED);
            break;

        case DATATYPE_FLOAT:
            reinterpret_cast<Matrix<FLOAT> *>(matrix)->Resize(nRows, nCols);
            break;

        case DATATYPE_INT:
            reinterpret_cast<Matrix<INT> *>(matrix)->Resize(nRows, nCols);
            break;
    }
}


void MultiTypeMatrix::Swap(MultiTypeMatrix& target)
{
    DataType tmpDataType;
    void *tmpMatrix;

    tmpDataType = this->dataType;
    this->dataType = target.dataType;
    target.dataType = tmpDataType;

    tmpMatrix = this->matrix;
    this->matrix = target.matrix;
    target.matrix = tmpMatrix;
}


void MultiTypeMatrix::Clear()
{
    if(matrix == NULL) return;

    switch(dataType)
    {
        case DATATYPE_UNDEFINED:
            verify(dataType != DATATYPE_UNDEFINED);
            break;

        case DATATYPE_FLOAT:
            delete reinterpret_cast<Matrix<FLOAT> *>(matrix);
            break;

        case DATATYPE_INT:
            delete reinterpret_cast<Matrix<INT> *>(matrix);
            break;
    }

    dataType = DATATYPE_UNDEFINED;
    matrix = NULL;
}


}

