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


#include "MultiClassifProbe.h"

#include <cmath>

#include "../core/Layer.h"


namespace fractal
{


MultiClassifProbe::MultiClassifProbe() : TrainableProbe(true)
{
    ResetStatistics();
}


void MultiClassifProbe::SetTarget(MultiTypeMatrix &mat, const unsigned long idxFrom, const unsigned long idxTo)
{
    unsigned long nRows, nCols;

    verify(engine != NULL);
    verify(linkedLayer->GetSize() > 1);

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


void MultiClassifProbe::ComputeErr(const unsigned long idxFrom, const unsigned long idxTo)
{
    verify(engine != NULL);
    //verify(linkedLayer->actType == ACT_SOFTMAX);

    Matrix<FLOAT> targetSub(target, idxFrom * nStream, idxTo * nStream + nStream - 1);
    Matrix<FLOAT> errSub(err, idxFrom * nStream, idxTo * nStream + nStream - 1);
    Matrix<FLOAT> actSub(GetActivation(), idxFrom * nStream, idxTo * nStream + nStream - 1);


    /* err = target - act */
    engine->MatCopy(targetSub, errSub, stream);
    engine->MatAdd(actSub, errSub, (FLOAT) -1, stream);
}


void MultiClassifProbe::EvaluateOnHost(MultiTypeMatrix &target, Matrix<FLOAT> &output)
{
    unsigned long nPartialError;
    double sePartialSum, cePartialSum;
    unsigned long dim, nFrame;

    dim = output.GetNumRows();
    nFrame = output.GetNumCols();

    sePartialSum = 0.0;
    cePartialSum = 0.0;
    nPartialError = 0;

    nSample += nFrame;

    switch(target.GetDataType())
    {
        case MultiTypeMatrix::DATATYPE_FLOAT:
            {
                FLOAT *t, *o;

                Matrix<FLOAT> *targetMat = reinterpret_cast<Matrix<FLOAT> *>(target.GetMatrix());


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



                #ifdef FRACTAL_USE_OMP
                #pragma omp parallel for reduction(+:sePartialSum, cePartialSum, nPartialError)
                #endif
                for(unsigned long i = 0; i < nFrame; i++)
                {
                    unsigned long tMaxIdx = 0;
                    unsigned long oMaxIdx = 0;
                    FLOAT tMax = t[i * dim];
                    FLOAT oMax = o[i * dim];

                    for(unsigned long j = 0; j < dim; j++)
                    {
                        unsigned long idx = i * dim + j;

                        FLOAT tCur = t[idx];
                        FLOAT oCur = o[idx];

                        FLOAT err = oCur - tCur;
                        sePartialSum += err * err;

                        if(tCur > (FLOAT) 0)
                            cePartialSum -= tCur * std::log(oCur + (double) 1e-300);

                        if(oCur > oMax)
                        {
                            oMax = oCur;
                            oMaxIdx = j;
                        }

                        if(tCur > tMax)
                        {
                            tMax = tCur;
                            tMaxIdx = j;
                        }
                    }

                    if(tMaxIdx != oMaxIdx) nPartialError++;
                }
            }
            break;

        case MultiTypeMatrix::DATATYPE_INT:
            {
                FLOAT *o;
                INT *t;
                Matrix<INT> *targetMat = reinterpret_cast<Matrix<INT> *>(target.GetMatrix());


                verify(targetMat->GetNumRows() == 1);
                verify(nFrame == targetMat->GetNumCols());

                PStream hostStream;

                engine->StreamCreate(hostStream, engine->GetHostLoc());

                targetMat->HostPull(hostStream);
                output.HostPull(hostStream);

                engine->StreamSynchronize(hostStream);
                engine->StreamDestroy(hostStream);

                stream.engine->StreamSynchronize(stream);

                t = targetMat->GetHostData();
                o = output.GetHostData();



                #ifdef FRACTAL_USE_OMP
                #pragma omp parallel for reduction(+:sePartialSum, cePartialSum, nPartialError)
                #endif
                for(unsigned long i = 0; i < nFrame; i++)
                {
                    verify(t[i] >= (INT) 0 && t[i] < (INT) dim);

                    unsigned long oMaxIdx = 0;
                    FLOAT oMax = o[i * dim];

                    for(unsigned long j = 0; j < dim; j++)
                    {
                        unsigned long idx = i * dim + j;

                        FLOAT oCur = o[idx];

                        FLOAT err = oCur - (FLOAT)((INT) j == t[i]);
                        sePartialSum += err * err;

                        if(oCur > oMax)
                        {
                            oMax = oCur;
                            oMaxIdx = j;
                        }
                    }

                    if(t[i] != (INT) oMaxIdx) nPartialError++;

                    cePartialSum -= std::log(o[i * dim + t[i]] + (double) 1e-300);
                }
            }
            break;

        default:
            verify(false);
    }


    seSum += sePartialSum;
    ceSum += cePartialSum;
    nError += nPartialError;
}


void MultiClassifProbe::ResetStatistics()
{
    nSample = 0;
    nError = 0;
    seSum = 0.0;
    ceSum = 0.0;
}


const double MultiClassifProbe::GetLoss()
{
    return GetAverageCrossEntropy();
}


void MultiClassifProbe::PrintStatistics(std::ostream &outStream)
{
    outStream << "MSE: " << GetMeanSquaredError()
        << "  ACE: " << GetAverageCrossEntropy()
        << "  FER: " << GetFrameErrorRate();
}


const double MultiClassifProbe::GetMeanSquaredError()
{
    return seSum / nSample;
}


const double MultiClassifProbe::GetAverageCrossEntropy()
{
    return ceSum / nSample;
}


const double MultiClassifProbe::GetFrameErrorRate()
{
    return (double) nError / nSample;
}


void MultiClassifProbe::SetEngine(Engine *engine)
{
    TrainableProbe::SetEngine(engine);

    target.SetEngine(engine);
}


}

