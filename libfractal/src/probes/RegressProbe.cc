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


#include "RegressProbe.h"

#include "../core/Layer.h"


namespace fractal
{


RegressProbe::RegressProbe() : TrainableProbe(false)
{
    ResetStatistics();
}


void RegressProbe::SetTarget(MultiTypeMatrix &mat, const unsigned long idxFrom, const unsigned long idxTo)
{
    unsigned long nRows, nCols;

    verify(engine != NULL);

    nRows = linkedLayer->GetSize();
    nCols = nStream * nUnroll;

    target.Resize(nRows, nCols);


    Matrix<FLOAT> targetSub(target, idxFrom * nStream, idxTo * nStream + nStream - 1);

    switch(mat.GetDataType())
    {
        case MultiTypeMatrix::DATATYPE_FLOAT:
            {
                Matrix<FLOAT> *ptrMat = reinterpret_cast<Matrix<FLOAT> *>(mat.GetMatrix());

                verify(ptrMat->GetNumRows() == nRows);
                verify(ptrMat->GetNumCols() == nStream * (idxTo - idxFrom + 1));

                engine->MatCopy(*ptrMat, targetSub, stream);
            }
            break;

        case MultiTypeMatrix::DATATYPE_INT:
            {
                Matrix<INT> *ptrMat = reinterpret_cast<Matrix<INT> *>(mat.GetMatrix());

                verify(ptrMat->GetNumRows() == 1);
                verify(ptrMat->GetNumCols() == nStream * (idxTo - idxFrom + 1));

                engine->OneHotEncode(*ptrMat, targetSub, stream);
            }
            break;

        default:
            verify(false);
    }
}


void RegressProbe::ComputeErr(const unsigned long idxFrom, const unsigned long idxTo)
{
    verify(engine != NULL);

    Matrix<FLOAT> targetSub(target, idxFrom * nStream, idxTo * nStream + nStream - 1);
    Matrix<FLOAT> errSub(err, idxFrom * nStream, idxTo * nStream + nStream - 1);
    Matrix<FLOAT> actSub(GetActivation(), idxFrom * nStream, idxTo * nStream + nStream - 1);


    /* err = target - act */
    engine->MatCopy(targetSub, errSub, stream);
    engine->MatAdd(actSub, errSub, (FLOAT) -1, stream);
}


void RegressProbe::EvaluateOnHost(MultiTypeMatrix &target, Matrix<FLOAT> &output)
{
    double sePartialSum;
    unsigned long n, dim, nFrame;

    dim = output.GetNumRows();
    nFrame = output.GetNumCols();

    sePartialSum = 0.0;

    nSample += nFrame;

    switch(target.GetDataType())
    {
        case MultiTypeMatrix::DATATYPE_FLOAT:
            {
                Matrix<FLOAT> *targetMat = reinterpret_cast<Matrix<FLOAT> *>(target.GetMatrix());
                FLOAT *t, *o;

                verify(dim == targetMat->GetNumRows());
                verify(nFrame == targetMat->GetNumCols());

                PStream hostStream;

                engine->StreamCreate(hostStream, engine->GetHostLoc());

                targetMat->HostPull(hostStream);
                output.HostPull(hostStream);

                engine->StreamSynchronize(hostStream);
                engine->StreamDestroy(hostStream);

                t = targetMat->GetHostData();
                o = output.GetHostData();

                n = dim * nFrame;


                #ifdef FRACTAL_USE_OMP
                #pragma omp parallel for reduction(+:sePartialSum)
                #endif
                for(unsigned long i = 0; i < n; i++)
                {
                    FLOAT err = o[i] - t[i];
                    sePartialSum += err * err;
                }
            }
            break;

        case MultiTypeMatrix::DATATYPE_INT:
            {
                Matrix<INT> *targetMat = reinterpret_cast<Matrix<INT> *>(target.GetMatrix());
                FLOAT *o;
                INT *t;

                verify(targetMat->GetNumRows() == 1);
                verify(nFrame == targetMat->GetNumCols());

                PStream hostStream;

                engine->StreamCreate(hostStream, engine->GetHostLoc());

                targetMat->HostPull(hostStream);
                output.HostPull(hostStream);

                engine->StreamSynchronize(hostStream);
                engine->StreamDestroy(hostStream);

                t = targetMat->GetHostData();
                o = output.GetHostData();


                #ifdef FRACTAL_USE_OMP
                #pragma omp parallel for reduction(+:sePartialSum)
                #endif
                for(unsigned long i = 0; i < nFrame; i++)
                {
                    for(unsigned long j = 0; j < dim; j++)
                    {
                        unsigned long idx = i * dim + j;

                        FLOAT err = o[idx] - (FLOAT)((INT) j == t[i]);
                        sePartialSum += err * err;
                    }
                }
            }
            break;

        default:
            verify(false);
    }

    seSum += sePartialSum;
}


void RegressProbe::ResetStatistics()
{
    nSample = 0;
    seSum = 0.0;
}


const double RegressProbe::GetLoss()
{
    return GetMeanSquaredError();
}


void RegressProbe::PrintStatistics(std::ostream &outStream)
{
    outStream << "MSE: " << GetMeanSquaredError();
}


const double RegressProbe::GetMeanSquaredError()
{
    return seSum / nSample;
}


void RegressProbe::SetEngine(Engine *engine)
{
    TrainableProbe::SetEngine(engine);

    target.SetEngine(engine);
}


}

