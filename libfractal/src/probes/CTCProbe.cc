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


#include "CTCProbe.h"

/* TODO: remove this? */
#include <iostream>

#include "../core/Layer.h"


namespace fractal
{

static const INT STATE_LABEL = 0x01;
static const INT STATE_LABEL_BEGIN = 0x02;
static const INT STATE_LABEL_END = 0x04;
static const INT STATE_COVERAGE = 0x10;
static const INT STATE_COVERAGE_BEGIN = 0x20;
static const INT STATE_COVERAGE_END = 0x40;


CTCProbe::CTCProbe() : TrainableProbe(true)
{
    maxTargetLen = 4096;

    forceBlankFirst = true;
    residualTraining = false;

    ResetStatistics();
}


void CTCProbe::SetQueueDepth(const unsigned long depth)
{
    verify(depth > 32);

    maxTargetLen = depth;
}


void CTCProbe::SetLabelGroup(std::vector<INT> &group)
{
    labelGroup = group;
}


void CTCProbe::SetWordDelimiter(std::vector<bool> &delimiter)
{
    wordDelimiter = delimiter;
}


void CTCProbe::SetTarget(MultiTypeMatrix &mat, const unsigned long idxFrom, const unsigned long idxTo)
{
    verify(engine != NULL);

    verify(mat.GetDataType() == MultiTypeMatrix::DATATYPE_INT);

    Matrix<INT> *ptrMat = reinterpret_cast<Matrix<INT> *>(mat.GetMatrix());

    verify(ptrMat->GetNumRows() == 2);
    verify(ptrMat->GetNumCols() == nStream * (idxTo - idxFrom + 1));

    verify(nStream == linkedLayer->GetNumStreams());
    verify(nUnroll == linkedLayer->GetNumUnrollSteps());

    ptrMat->HostPull(stream);

    engine->StreamSynchronize(stream);

    INT *ptr = ptrMat->GetHostData();
    INT *ptrTargetBuf = targetBuf.GetHostData();
    INT *ptrTargetHead = targetHead.GetHostData();
    INT *ptrTargetTail = targetTail.GetHostData();
    INT *ptrTargetState = targetState.GetHostData();
    INT *ptrTargetStartIdx = targetStartIdx.GetHostData();
    INT *ptrTargetCount = targetCount.GetHostData();


    /* Generate the target matrix with ring buffer structure */
    #ifdef FRACTAL_USE_OMP
    #pragma omp parallel for
    #endif
    for(long streamIdx = 0; streamIdx < (long) nStream; streamIdx++)
    {
        for(long frameIdx = (long) idxFrom; frameIdx <= (long) idxTo; frameIdx++)
        {
            INT *framePtr = ptr + ((frameIdx - idxFrom) * nStream + streamIdx) * 2;

            INT seq = framePtr[0];
            INT sig = framePtr[1];


            INT *ptrCurTargetState = ptrTargetState + nStream * frameIdx + streamIdx;
            INT *ptrPrevTargetState = ptrTargetState + nStream * ((frameIdx - 1 + nUnroll) % nUnroll) + streamIdx;

            *ptrCurTargetState = *ptrPrevTargetState;

            *ptrCurTargetState &= ~STATE_LABEL_BEGIN & ~STATE_COVERAGE_BEGIN;

            if((*ptrPrevTargetState & STATE_LABEL_END) != (INT) 0)
            {
                *ptrCurTargetState &= ~STATE_LABEL_END & ~STATE_LABEL;
            }

            if((*ptrPrevTargetState & STATE_COVERAGE_END) != (INT) 0)
            {
                *ptrCurTargetState &= ~STATE_COVERAGE_END & ~STATE_COVERAGE;
            }

            if((sig & (INT) 0x01) != (INT) 0) /* Label & label coverage begin */
            {
                *ptrCurTargetState = STATE_LABEL | STATE_LABEL_BEGIN | STATE_COVERAGE | STATE_COVERAGE_BEGIN;
            }
            if((sig & (INT) 0x02) != (INT) 0) /* Label end */
            {
                //verify((*ptrCurTargetState & STATE_LABEL) != (INT) 0);
                if((*ptrCurTargetState & STATE_LABEL) != (INT) 0)
                {
                    *ptrCurTargetState |= STATE_LABEL_END;
                }
            }
            if((sig & (INT) 0x04) != (INT) 0) /* Coverage end */
            {
                //verify((*ptrCurTargetState & STATE_COVERAGE) != (INT) 0);
                if((*ptrCurTargetState & STATE_COVERAGE) != (INT) 0)
                {
                    *ptrCurTargetState |= STATE_COVERAGE_END;
                }
            }


            INT headIdx = ptrTargetHead[streamIdx];


            if((*ptrCurTargetState & STATE_LABEL) != (INT) 0)
            {
                if(headIdx == ptrTargetTail[streamIdx]) /* Queue full */
                {
                    /* TODO */
                    std::cerr << std::endl << "CTC queue overflow" << std::endl;
                    verify(false);
                }

                ptrTargetBuf[headIdx * nStream + streamIdx] = seq;

                ptrTargetHead[streamIdx] = (headIdx + 1) % maxTargetLen;
                if(ptrTargetTail[streamIdx] == -1)
                {
                    ptrTargetTail[streamIdx] = headIdx;
                }
            }

            if((*ptrCurTargetState & STATE_LABEL_BEGIN) != (INT) 0) /* Label & label coverage begin */
            {
                ptrTargetStartIdx[nStream * frameIdx + streamIdx] = headIdx;
                ptrTargetCount[nStream * frameIdx + streamIdx] = 1;
            }
            else if((*ptrCurTargetState & STATE_COVERAGE) != (INT) 0)
            {
                ptrTargetStartIdx[nStream * frameIdx + streamIdx] =
                    ptrTargetStartIdx[nStream * ((frameIdx - 1 + nUnroll) % nUnroll) + streamIdx];
                ptrTargetCount[nStream * frameIdx + streamIdx] =
                    ptrTargetCount[nStream * ((frameIdx - 1 + nUnroll) % nUnroll) + streamIdx];

                if((*ptrCurTargetState & STATE_LABEL) != (INT) 0)
                {
                    ptrTargetCount[nStream * frameIdx + streamIdx]++;
                }
            }
        }
    }


    targetBuf.HostPush();
    targetHead.HostPush();
    targetTail.HostPush();
    targetState.HostPush();
    targetStartIdx.HostPush();
    targetCount.HostPush();
}


void CTCProbe::ComputeErr(const unsigned long idxFrom, const unsigned long idxTo)
{
    verify(engine != NULL);

    verify(nStream == linkedLayer->GetNumStreams());
    verify(nUnroll == linkedLayer->GetNumUnrollSteps());

    GetActivation().HostPull(stream);
    GetState().HostPull(stream);
    engine->StreamSynchronize(stream);

    FLOAT *ptrAct = GetActivation().GetHostData(); /* Unnormalized log softmax activation */
    FLOAT *ptrState = GetState().GetHostData(); /* Unnormalized log softmax activation */
    FLOAT *ptrErr = err.GetHostData();

    INT *ptrTargetBuf = targetBuf.GetHostData();
    INT *ptrTargetState = targetState.GetHostData();
    INT *ptrTargetStartIdx = targetStartIdx.GetHostData();
    INT *ptrTargetCount = targetCount.GetHostData();

    FLOAT *ptrForward = forward.GetHostData();
    FLOAT *ptrBackward = backward.GetHostData();

    unsigned long layerSize = GetLayerSize();

    const FLOAT inf = ((FLOAT) (+1.0))/((FLOAT) (+0.0));

    /* Forward */
    #ifdef FRACTAL_USE_OMP
    #pragma omp parallel for
    #endif
    for(long streamIdx = 0; streamIdx < (long) nStream; streamIdx++)
    {
        for(long frameIdx = (long) idxFrom; frameIdx <= (long) idxTo; frameIdx++)
        {
            FLOAT *ptrCurState = ptrState + (nStream * frameIdx + streamIdx) * layerSize;

            FLOAT *ptrCurForward = ptrForward + (nStream * frameIdx + streamIdx) * (2 * maxTargetLen + 1);
            FLOAT *ptrPrevForward = ptrForward + (nStream * ((frameIdx - 1 + nUnroll) % nUnroll) + streamIdx) * (2 * maxTargetLen + 1);

            INT curTargetStartIdx = ptrTargetStartIdx[nStream * frameIdx + streamIdx];
            INT curTargetState = ptrTargetState[nStream * frameIdx + streamIdx];
            INT curTargetCount = ptrTargetCount[nStream * frameIdx + streamIdx];

            FLOAT curStateSum = LogSumExpN(ptrCurState, layerSize);

            if((curTargetState & STATE_LABEL_BEGIN) != (INT) 0) /* First frame */
            {
                ptrCurForward[0] = ptrCurState[layerSize - 1] - curStateSum; /* Blank */
                if(forceBlankFirst == true)
                {
                    ptrCurForward[1] = -inf;
                }
                else
                {
                    ptrCurForward[1] = ptrCurState[ptrTargetBuf[curTargetStartIdx * nStream + streamIdx]] - curStateSum; /* First label */
                }
                ptrCurForward[2] = -inf;
                ptrCurForward[3] = -inf;
                ptrCurForward[4] = -inf;
            }
            else if((curTargetState & STATE_COVERAGE) != (INT) 0)
            {
                FLOAT prevForward_im1 = -inf;
                FLOAT prevForward_im2 = -inf;

                for(long i = 0; i < 2 * curTargetCount + 1; i++)
                {
                    FLOAT prevSum;
                    FLOAT prevForward_im0 = ptrPrevForward[i];

                    INT actIdx = (i % 2 == 0) ? (layerSize - 1) : (ptrTargetBuf[((curTargetStartIdx + i / 2) % maxTargetLen) * nStream + streamIdx]);
                    INT actIdx_im2 = ((i % 2 == 0) || (i - 2 < 0)) ? (layerSize - 1) : (ptrTargetBuf[((curTargetStartIdx + (i - 2) / 2 + maxTargetLen) % maxTargetLen) * nStream + streamIdx]);

                    if(actIdx == actIdx_im2)
                    {
                        prevSum = LogSumExp2(prevForward_im0, prevForward_im1);
                    }
                    else
                    {
                        prevSum = LogSumExp3(prevForward_im0, prevForward_im1, prevForward_im2);
                        //prevSum = LogSumExp2(prevForward_im1, prevForward_im2);
                    }

                    prevForward_im2 = prevForward_im1;
                    prevForward_im1 = prevForward_im0;

                    ptrCurForward[i] = prevSum + ptrCurState[actIdx] - curStateSum;
                }
                
                if(curTargetCount < (long) maxTargetLen)
                {
                    ptrCurForward[2 * curTargetCount + 1] = -inf;
                    ptrCurForward[2 * curTargetCount + 2] = -inf;
                }
            }
        }
    }


    /* Backward */
    #ifdef FRACTAL_USE_OMP
    #pragma omp parallel for
    #endif
    for(long streamIdx = 0; streamIdx < (long) nStream; streamIdx++)
    {
        bool computeBackward = false;
        INT maxTargetCount = (INT) 0;
        FLOAT logLikelihood = (FLOAT) 0;

        for(long frameOffset = 0; frameOffset < (long) nUnroll; frameOffset++)
        {
            long frameIdx = (nUnroll + idxTo - frameOffset) % nUnroll;

            FLOAT *ptrCurAct = ptrAct + (nStream * frameIdx + streamIdx) * layerSize;
            FLOAT *ptrPrevState = ptrState + (nStream * ((frameIdx + 1) % nUnroll) + streamIdx) * layerSize;

            FLOAT *ptrCurErr = ptrErr + (nStream * frameIdx + streamIdx) * layerSize;

            FLOAT *ptrCurForward = ptrForward + (nStream * frameIdx + streamIdx) * (2 * maxTargetLen + 1);
            FLOAT *ptrCurBackward = ptrBackward + (nStream * frameIdx + streamIdx) * (2 * maxTargetLen + 1);
            FLOAT *ptrPrevBackward = ptrBackward + (nStream * ((frameIdx + 1) % nUnroll) + streamIdx) * (2 * maxTargetLen + 1);

            INT curTargetStartIdx = ptrTargetStartIdx[nStream * frameIdx + streamIdx];
            INT curTargetState = ptrTargetState[nStream * frameIdx + streamIdx];
            INT curTargetCount = ptrTargetCount[nStream * frameIdx + streamIdx];


            if((curTargetState & STATE_COVERAGE_END) != (INT) 0) /* Last frame */
            {
                if((frameIdx >= (long) idxFrom) && (frameIdx <= (long) idxTo))
                {
                    computeBackward = true;

                    verify(curTargetCount > 0);

                    ptrCurBackward[2 * curTargetCount - 0] = (FLOAT) 0; /* Blank */
                    ptrCurBackward[2 * curTargetCount - 1] = (FLOAT) 0; /* Last label */
                    //ptrCurBackward[2 * curTargetCount - 1] = -inf; /* Last label */

                    for(long i = 0; i < 2 * curTargetCount - 1; i++)
                    {
                        ptrCurBackward[i] = -inf;
                    }

                    maxTargetCount = curTargetCount;
                }
                else
                {
                    computeBackward = false;
                }
            }
            else if((curTargetState & STATE_COVERAGE) != (INT) 0)
            {
                FLOAT prevStateSum = LogSumExpN(ptrPrevState, layerSize);

                if(computeBackward == true)
                {
                    FLOAT prevBackward_ip1 = -inf;
                    FLOAT prevBackward_ip2 = -inf;

                    for(long i = 2 * maxTargetCount; i >= 0; i--)
                    {
                        FLOAT prevSum;

                        INT actIdx = (i % 2 == 0) ? (layerSize - 1) : (ptrTargetBuf[((curTargetStartIdx + i / 2) % maxTargetLen) * nStream + streamIdx]);
                        INT actIdx_ip2 = ((i % 2 == 0) || (i + 2 > 2 * maxTargetCount)) ? (layerSize - 1) : (ptrTargetBuf[((curTargetStartIdx + (i + 2) / 2) % maxTargetLen) * nStream + streamIdx]);

                        FLOAT prevBackward_ip0 = ptrPrevBackward[i] + ptrPrevState[actIdx] - prevStateSum;

                        if(actIdx == actIdx_ip2) /* Blank or same label */
                        {
                            prevSum = LogSumExp2(prevBackward_ip0, prevBackward_ip1);
                        }
                        else
                        {
                            prevSum = LogSumExp3(prevBackward_ip0, prevBackward_ip1, prevBackward_ip2);
                            //prevSum = LogSumExp2(prevBackward_ip1, prevBackward_ip2);
                        }

                        prevBackward_ip2 = prevBackward_ip1;
                        prevBackward_ip1 = prevBackward_ip0;

                        ptrCurBackward[i] = prevSum;
                    }
                }
            }
            else
            {
                computeBackward = false;
            }

            for(unsigned long i = 0; i < layerSize; i++)
            {
                ptrCurErr[i] = (FLOAT) 0;
            }

            if(computeBackward == true)
            {
                if((curTargetState & STATE_COVERAGE_END) != (INT) 0) /* Last frame */
                {
                    logLikelihood = LogSumExp2(ptrCurForward[2 * curTargetCount], ptrCurForward[2 * curTargetCount - 1]);
                }

                for(long i = 0; i < 2 * curTargetCount + 1; i++)
                {
                    INT actIdx = (i % 2 == 0) ? layerSize - 1 : ptrTargetBuf[((curTargetStartIdx + i / 2) % maxTargetLen) * nStream + streamIdx];

                    ptrCurErr[actIdx] += exp(ptrCurForward[i] + ptrCurBackward[i] - logLikelihood);
                }

                FLOAT errSum = (FLOAT) 0;

                for(unsigned long i = 0; i < layerSize; i++)
                {
                    errSum += ptrCurErr[i];
                }

                for(unsigned long i = 0; i < layerSize; i++)
                {
                    ptrCurErr[i] = ptrCurErr[i] / errSum - ptrCurAct[i];
                }

            }
        }
    }

   
    /* Backward - residual */
    if(residualTraining == true)
    {
        #ifdef FRACTAL_USE_OMP
        #pragma omp parallel for
        #endif
        for(long streamIdx = 0; streamIdx < (long) nStream; streamIdx++)
        {
            /* Assume that (idxTo - idxFrom + 1) is constant during training */
            //long nextIdxFrom = (idxTo + 1) % nUnroll;
            //long nextIdxTo = (idxTo + (idxTo - idxFrom + 1)) % nUnroll;
            //long nFrameBackwardOnly = (idxTo - nextIdxTo) & nUnroll;
            long nFrameBackwardOnly = nUnroll - (idxTo - idxFrom + 1);

            INT maxTargetCount = (INT) 0;
            FLOAT logLikelihood = (FLOAT) 0;
            bool computeBackward = true;

            /* First check whether the backward process is needed or not */
            for(long frameOffset = 0; frameOffset < nFrameBackwardOnly; frameOffset++)
            {
                long frameIdx = (nUnroll + idxTo - frameOffset) % nUnroll;

                INT curTargetState = ptrTargetState[nStream * frameIdx + streamIdx];

                if((curTargetState & STATE_COVERAGE) == (INT) 0 /* Not in the coverage */
                        || (curTargetState & STATE_COVERAGE_END) != (INT) 0 /* Last frame (boundary detected) */
                        || (curTargetState & STATE_COVERAGE_BEGIN) != (INT) 0) /* First frame (boundary detected) */
                {
                    computeBackward = false;
                    break;
                }
            }

            if(computeBackward == false) continue;

            for(long frameOffset = 0; frameOffset < (long) nUnroll; frameOffset++)
            {
                long frameIdx = (nUnroll + idxTo - frameOffset) % nUnroll;

                FLOAT *ptrCurAct = ptrAct + (nStream * frameIdx + streamIdx) * layerSize;
                FLOAT *ptrPrevState = ptrState + (nStream * ((frameIdx + 1) % nUnroll) + streamIdx) * layerSize;

                FLOAT *ptrCurErr = ptrErr + (nStream * frameIdx + streamIdx) * layerSize;

                FLOAT *ptrCurForward = ptrForward + (nStream * frameIdx + streamIdx) * (2 * maxTargetLen + 1);
                FLOAT *ptrCurBackward = ptrBackward + (nStream * frameIdx + streamIdx) * (2 * maxTargetLen + 1);
                FLOAT *ptrPrevBackward = ptrBackward + (nStream * ((frameIdx + 1) % nUnroll) + streamIdx) * (2 * maxTargetLen + 1);

                INT curTargetStartIdx = ptrTargetStartIdx[nStream * frameIdx + streamIdx];
                INT curTargetState = ptrTargetState[nStream * frameIdx + streamIdx];
                INT curTargetCount = ptrTargetCount[nStream * frameIdx + streamIdx];

                if((curTargetState & STATE_COVERAGE) == (INT) 0 /* Not in the coverage */
                        || (curTargetState & STATE_COVERAGE_END) != (INT) 0) /* Last frame */
                {
                    break; /* No more error computation */
                }


                /* Backward */

                if(frameOffset == 0)
                {
                    verify(curTargetCount > 0);
                    maxTargetCount = curTargetCount;
#if 1
                    for(long i = 0; i < 2 * curTargetCount + 1; i++)
                    {
                        ptrCurBackward[i] = (FLOAT) 0;
                    }

                    logLikelihood = LogSumExpN(ptrCurForward, 2 * curTargetCount + 1);
#else
                    for(long i = 0; i < 2 * curTargetCount + 1; i++)
                    {
                        ptrCurBackward[i] = 2 * ptrCurForward[i];
                    }
                    logLikelihood = LogSumExpN(ptrCurBackward, 2 * curTargetCount + 1);
                    for(long i = 0; i < 2 * curTargetCount + 1; i++)
                    {
                        //ptrCurBackward[i] = (FLOAT) 0;
                        ptrCurBackward[i] = ptrCurForward[i];
                    }
#endif
                }
                else
                {
                    FLOAT prevStateSum = LogSumExpN(ptrPrevState, layerSize);

                    FLOAT prevBackward_ip1 = -inf;
                    FLOAT prevBackward_ip2 = -inf;

                    for(long i = 2 * maxTargetCount; i >= 0; i--)
                    {
                        FLOAT prevSum;

                        INT actIdx = (i % 2 == 0) ? (layerSize - 1) : (ptrTargetBuf[((curTargetStartIdx + i / 2) % maxTargetLen) * nStream + streamIdx]);
                        INT actIdx_ip2 = ((i % 2 == 0) || (i + 2 > 2 * maxTargetCount)) ? (layerSize - 1) : (ptrTargetBuf[((curTargetStartIdx + (i + 2) / 2) % maxTargetLen) * nStream + streamIdx]);

                        FLOAT prevBackward_ip0 = ptrPrevBackward[i] + ptrPrevState[actIdx] - prevStateSum;

                        if(actIdx == actIdx_ip2) /* Blank or same label */
                        {
                            prevSum = LogSumExp2(prevBackward_ip0, prevBackward_ip1);
                        }
                        else
                        {
                            prevSum = LogSumExp3(prevBackward_ip0, prevBackward_ip1, prevBackward_ip2);
                        }

                        prevBackward_ip2 = prevBackward_ip1;
                        prevBackward_ip1 = prevBackward_ip0;

                        ptrCurBackward[i] = prevSum;
                    }
                }


                if(frameOffset < nFrameBackwardOnly)
                {
                    continue;
                }

                /* Error computation */
                for(unsigned long i = 0; i < layerSize; i++)
                {
                    ptrCurErr[i] = (FLOAT) 0;
                }

                for(long i = 0; i < 2 * curTargetCount + 1; i++)
                {
                    INT actIdx = (i % 2 == 0) ? layerSize - 1 : ptrTargetBuf[((curTargetStartIdx + i / 2) % maxTargetLen) * nStream + streamIdx];

                    ptrCurErr[actIdx] += exp(ptrCurForward[i] + ptrCurBackward[i] - logLikelihood);
                }

                FLOAT errSum = (FLOAT) 0;

                for(unsigned long i = 0; i < layerSize; i++)
                {
                    errSum += ptrCurErr[i];
                }

                for(unsigned long i = 0; i < layerSize; i++)
                {
                    ptrCurErr[i] = ptrCurErr[i] / errSum - ptrCurAct[i];
                }
            }
        }
    }

    err.HostPush();
    forward.HostPush();
    //backward.HostPush();

    Dequeue(idxFrom, idxTo);
}


void CTCProbe::EvaluateOnHost(MultiTypeMatrix &target, Matrix<FLOAT> &output)
{
    verify(engine != NULL);

    verify(target.GetDataType() == MultiTypeMatrix::DATATYPE_INT);

    Matrix<INT> *ptrTargetMat = reinterpret_cast<Matrix<INT> *>(target.GetMatrix());

    verify(ptrTargetMat->GetNumRows() == 2);

    unsigned long nFrames = ptrTargetMat->GetNumCols();
    unsigned long layerSize = GetLayerSize();

    verify(layerSize == output.GetNumRows());
    verify(nFrames % nStream == 0);
    verify(nFrames == output.GetNumCols());

    verify(nStream == linkedLayer->GetNumStreams());
    verify(nUnroll == linkedLayer->GetNumUnrollSteps());

    PStream hostStream;

    engine->StreamCreate(hostStream, engine->GetHostLoc());

    ptrTargetMat->HostPull(hostStream);
    output.HostPull(hostStream);

    engine->StreamSynchronize(hostStream);
    engine->StreamDestroy(hostStream);

    INT *ptrTarget = ptrTargetMat->GetHostData();
    INT *ptrTargetBuf = targetBuf.GetHostData();
    INT *ptrTargetHead = targetHead.GetHostData();
    INT *ptrTargetTail = targetTail.GetHostData();
    INT *ptrTargetState = targetState.GetHostData();
    INT *ptrTargetCount = targetCount.GetHostData();

    FLOAT *ptrOutput = output.GetHostData();
    INT *ptrOutputBuf = outputBuf.GetHostData();
    INT *ptrOutputHead = outputHead.GetHostData();
    INT *ptrOutputTail = outputTail.GetHostData();

    FLOAT *ptrForward = forward.GetHostData();

    INT *ptrPrevMaxIdx = prevMaxIdx.GetHostData();

    const FLOAT inf = ((FLOAT) (+1.0))/((FLOAT) (+0.0));

    unsigned long nNewSeq = 0;

    double partialLossSum = 0.0;

    unsigned long nNewLabels = 0;
    unsigned long nNewLabelSubstitutions = 0;
    unsigned long nNewLabelDeletions = 0;
    unsigned long nNewLabelInsertions = 0;

    unsigned long nNewWords = 0;
    unsigned long nNewWordSubstitutions = 0;
    unsigned long nNewWordDeletions = 0;
    unsigned long nNewWordInsertions = 0;

    /* Generate the target and output matrices with ring buffer structure.
     * Perform Wagner–Fischer algorithm to compute edit distance.
     * CTC forward propagation is needed to compute the loss value.
     */
    #ifdef FRACTAL_USE_OMP
    #pragma omp parallel for \
        reduction(+:nNewSeq, partialLossSum, \
                nNewLabels, nNewLabelSubstitutions, nNewLabelDeletions, nNewLabelInsertions, \
                nNewWords, nNewWordSubstitutions, nNewWordDeletions, nNewWordInsertions)
    #endif
    for(long streamIdx = 0; streamIdx < (long) nStream; streamIdx++)
    {
        for(long frameIdx = 0; frameIdx < (long) (nFrames / nStream); frameIdx++)
        {
            FLOAT *ptrCurForward = ptrForward + streamIdx * (2 * maxTargetLen + 1);

            INT *ptrCurTarget = ptrTarget + (frameIdx * nStream + streamIdx) * 2;
            FLOAT *ptrCurOutput = ptrOutput + (frameIdx * nStream + streamIdx) * layerSize;

            INT *ptrCurTargetState = ptrTargetState + streamIdx;
            INT *ptrCurTargetCount = ptrTargetCount + streamIdx;

            INT seq = ptrCurTarget[0];
            INT sig = ptrCurTarget[1];


            *ptrCurTargetState &= ~STATE_LABEL_BEGIN & ~STATE_COVERAGE_BEGIN;

            if((*ptrCurTargetState & STATE_LABEL_END) != (INT) 0)
            {
                *ptrCurTargetState &= ~STATE_LABEL_END & ~STATE_LABEL;
            }

            if((*ptrCurTargetState & STATE_COVERAGE_END) != (INT) 0)
            {
                *ptrCurTargetState &= ~STATE_COVERAGE_END & ~STATE_COVERAGE;
            }

            if((sig & (INT) 0x01) != (INT) 0) /* Label & label coverage begin */
            {
                *ptrCurTargetState = STATE_LABEL | STATE_LABEL_BEGIN | STATE_COVERAGE | STATE_COVERAGE_BEGIN;
            }
            if((sig & (INT) 0x02) != (INT) 0) /* Label end */
            {
                //verify((*ptrCurTargetState & STATE_LABEL) != (INT) 0);
                if((*ptrCurTargetState & STATE_LABEL) != (INT) 0)
                {
                    *ptrCurTargetState |= STATE_LABEL_END;
                }
            }
            if((sig & (INT) 0x04) != (INT) 0) /* Coverage end */
            {
                //verify((*ptrCurTargetState & STATE_COVERAGE) != (INT) 0);
                if((*ptrCurTargetState & STATE_COVERAGE) != (INT) 0)
                {
                    *ptrCurTargetState |= STATE_COVERAGE_END;
                }
            }



            if((*ptrCurTargetState & STATE_LABEL) != (INT) 0)
            {
                INT headIdx = ptrTargetHead[streamIdx];

                if(headIdx == ptrTargetTail[streamIdx]) /* Queue full */
                {
                    /* TODO */
                    std::cerr << std::endl << "CTC queue overflow" << std::endl;
                    verify(false);
                }

                ptrTargetBuf[headIdx * nStream + streamIdx] = seq;

                ptrTargetHead[streamIdx] = (headIdx + 1) % maxTargetLen;
                if(ptrTargetTail[streamIdx] == -1)
                {
                    ptrTargetTail[streamIdx] = headIdx;
                }
            }

            if((*ptrCurTargetState & STATE_LABEL_BEGIN) != (INT) 0) /* Label & label coverage begin */
            {
                *ptrCurTargetCount = 1;
            }
            else if((*ptrCurTargetState & STATE_LABEL) != (INT) 0)
            {
                (*ptrCurTargetCount)++;
            }


            if((*ptrCurTargetState & STATE_COVERAGE) == (INT) 0) continue;


            /* Forward */
            if((*ptrCurTargetState & STATE_LABEL_BEGIN) != (INT) 0) /* First frame */
            {
                ptrCurForward[0] = log(ptrCurOutput[layerSize - 1]); /* Blank */
                if(forceBlankFirst == true)
                {
                    ptrCurForward[1] = -inf;
                }
                else
                {
                    ptrCurForward[1] = log(ptrCurOutput[ptrTargetBuf[ptrTargetHead[streamIdx] * nStream + streamIdx]]); /* First label */
                }
                ptrCurForward[2] = -inf;
                ptrCurForward[3] = -inf;
                ptrCurForward[4] = -inf;
            }
            else if((*ptrCurTargetState & STATE_COVERAGE) != (INT) 0)
            {
                FLOAT prevForward_im1 = -inf;
                FLOAT prevForward_im2 = -inf;

                for(long i = 0; i < 2 * *ptrCurTargetCount + 1; i++)
                {
                    FLOAT prevSum;
                    FLOAT prevForward_im0 = ptrCurForward[i];

                    INT actIdx = (i % 2 == 0) ? (layerSize - 1) : (ptrTargetBuf[((ptrTargetTail[streamIdx] + i / 2) % maxTargetLen) * nStream + streamIdx]);
                    INT actIdx_im2 = ((i % 2 == 0) || (i - 2 < 0)) ? (layerSize - 1) : (ptrTargetBuf[((ptrTargetTail[streamIdx] + (i - 2) / 2 + maxTargetLen) % maxTargetLen) * nStream + streamIdx]);

                    if(actIdx == actIdx_im2)
                    {
                        prevSum = LogSumExp2(prevForward_im0, prevForward_im1);
                    }
                    else
                    {
                        prevSum = LogSumExp3(prevForward_im0, prevForward_im1, prevForward_im2);
                    }

                    prevForward_im2 = prevForward_im1;
                    prevForward_im1 = prevForward_im0;

                    ptrCurForward[i] = prevSum + log(ptrCurOutput[actIdx]);
                }
                
                if(*ptrCurTargetCount < (long) maxTargetLen)
                {
                    ptrCurForward[2 * *ptrCurTargetCount + 1] = -inf;
                    ptrCurForward[2 * *ptrCurTargetCount + 2] = -inf;
                }
            }

            if((*ptrCurTargetState & STATE_COVERAGE_END) != (INT) 0) /* Last frame */
            {
                FLOAT logLikelihood = LogSumExp2(ptrCurForward[2 * *ptrCurTargetCount], ptrCurForward[2 * *ptrCurTargetCount - 1]);
                partialLossSum += -logLikelihood;
            }

            /* Find the maximum output index */
            INT maxIdx = 0;
            FLOAT maxVal = ptrCurOutput[0];

            for(unsigned long i = 1; i < layerSize; i++)
            {
                if(ptrCurOutput[i] > maxVal)
                {
                    maxIdx = i;
                    maxVal = ptrCurOutput[i];
                }
            }


            if(((*ptrCurTargetState & STATE_COVERAGE_BEGIN) != (INT) 0)
                    || (ptrPrevMaxIdx[streamIdx] != maxIdx))
            {
                if(maxIdx != (INT) (layerSize - 1)) /* Not blank */
                {
                    /* New output label */

                    INT headIdx = ptrOutputHead[streamIdx];

                    if(headIdx == ptrOutputTail[streamIdx]) /* Queue full */
                    {
                        /* TODO */
                        std::cerr << std::endl << "CTC queue overflow" << std::endl;
                        verify(false);
                    }

                    ptrOutputBuf[headIdx * nStream + streamIdx] = maxIdx;

                    ptrOutputHead[streamIdx] = (headIdx + 1) % maxTargetLen;

                    if(ptrOutputTail[streamIdx] == -1)
                    {
                        ptrOutputTail[streamIdx] = headIdx;
                    }
                }
            }

            if((*ptrCurTargetState & STATE_COVERAGE_END) != (INT) 0)
            {
                nNewSeq++;


                /* Compute label edit distance */
                EditDistance distance = ComputeEditDistance(streamIdx);

                nNewLabels += distance.nSymbols;
                nNewLabelSubstitutions += distance.nSubstitutions;
                nNewLabelDeletions += distance.nDeletions;
                nNewLabelInsertions += distance.nInsertions;

                /* Compute label edit distance */
                if(wordDelimiter.size() > 0)
                {
                    distance = ComputeWordEditDistance(streamIdx);

                    nNewWords += distance.nSymbols;
                    nNewWordSubstitutions += distance.nSubstitutions;
                    nNewWordDeletions += distance.nDeletions;
                    nNewWordInsertions += distance.nInsertions;
                }

                /* Dequeue */
                ptrTargetTail[streamIdx] = -1;
                ptrOutputTail[streamIdx] = -1;
            }

            ptrPrevMaxIdx[streamIdx] = maxIdx;
        }
    }

    nSeq += nNewSeq;
    lossSum += partialLossSum;
    
    labelEditDistance.nSymbols += nNewLabels;
    labelEditDistance.nSubstitutions += nNewLabelSubstitutions;
    labelEditDistance.nDeletions += nNewLabelDeletions;
    labelEditDistance.nInsertions += nNewLabelInsertions;

    wordEditDistance.nSymbols += nNewWords;
    wordEditDistance.nSubstitutions += nNewWordSubstitutions;
    wordEditDistance.nDeletions += nNewWordDeletions;
    wordEditDistance.nInsertions += nNewWordInsertions;

    targetBuf.HostPush();
    targetHead.HostPush();
    targetTail.HostPush();
    targetState.HostPush();
    targetCount.HostPush();

    outputBuf.HostPush();
    outputHead.HostPush();
    outputTail.HostPush();

    forward.HostPush();

    prevMaxIdx.HostPush();
}


void CTCProbe::ResetStatistics()
{
    lossSum = 0.0;
    nSeq = 0;

    labelEditDistance = EditDistance();
    wordEditDistance = EditDistance();
}


const double CTCProbe::GetLoss()
{
    return lossSum / nSeq;
}


const double CTCProbe::GetLabelErrorRate()
{
    return ((double) (labelEditDistance.nSubstitutions + labelEditDistance.nDeletions + labelEditDistance.nInsertions)) / labelEditDistance.nSymbols;
}


const double CTCProbe::GetLabelSubstitutionRate()
{
    return ((double) labelEditDistance.nSubstitutions) / labelEditDistance.nSymbols;
}


const double CTCProbe::GetLabelDeletionRate()
{
    return ((double) labelEditDistance.nDeletions) / labelEditDistance.nSymbols;
}


const double CTCProbe::GetLabelInsertionRate()
{
    return ((double) labelEditDistance.nInsertions) / labelEditDistance.nSymbols;
}


const double CTCProbe::GetWordErrorRate()
{
    return ((double) (wordEditDistance.nSubstitutions + wordEditDistance.nDeletions + wordEditDistance.nInsertions)) / wordEditDistance.nSymbols;
}


const double CTCProbe::GetWordSubstitutionRate()
{
    return ((double) wordEditDistance.nSubstitutions) / wordEditDistance.nSymbols;
}


const double CTCProbe::GetWordDeletionRate()
{
    return ((double) wordEditDistance.nDeletions) / wordEditDistance.nSymbols;
}


const double CTCProbe::GetWordInsertionRate()
{
    return ((double) wordEditDistance.nInsertions) / wordEditDistance.nSymbols;
}


void CTCProbe::PrintStatistics(std::ostream &outStream)
{
    outStream << "Loss: " << GetLoss() << "  LER: " << GetLabelErrorRate() * 100 << "%"
        << " (S: " << GetLabelSubstitutionRate() * 100 << "%"
        << ", D: " << GetLabelDeletionRate() * 100 << "%"
        << ", I: " << GetLabelInsertionRate() * 100 << "%" << ")";

    if(wordDelimiter.size() > 0)
    {
        outStream << "  WER: " << GetWordErrorRate() * 100 << "%"
            << " (S: " << GetWordSubstitutionRate() * 100 << "%"
            << ", D: " << GetWordDeletionRate() * 100 << "%"
            << ", I: " << GetWordInsertionRate() * 100 << "%" << ")";
    }
}


void CTCProbe::SetForceBlankFirst(const bool val)
{
    forceBlankFirst = val;
}


void CTCProbe::SetResidualTraining(const bool val)
{
    residualTraining = val;
}


void CTCProbe::SetEngine(Engine *engine)
{
    TrainableProbe::SetEngine(engine);

    targetBuf.SetEngine(engine);
    targetHead.SetEngine(engine);
    targetTail.SetEngine(engine);
    targetState.SetEngine(engine);
    targetStartIdx.SetEngine(engine);
    targetCount.SetEngine(engine);

    outputBuf.SetEngine(engine);
    outputHead.SetEngine(engine);
    outputTail.SetEngine(engine);

    forward.SetEngine(engine);
    backward.SetEngine(engine);

    prevMaxIdx.SetEngine(engine);

    distanceMat.SetEngine(engine);
    substitutionMat.SetEngine(engine);
    deletionMat.SetEngine(engine);
    insertionMat.SetEngine(engine);
}


void CTCProbe::InitTraining(const unsigned long nStream, const unsigned long nUnroll)
{
    TrainableProbe::InitTraining(nStream, nUnroll);

    /* Target ring buffer */
    targetBuf.Resize(nStream, maxTargetLen);
    targetHead.Resize(nStream, 1);
    targetTail.Resize(nStream, 1);
    targetState.Resize(nStream, nUnroll);
    targetStartIdx.Resize(nStream, nUnroll);
    targetCount.Resize(nStream, nUnroll);

    INT *ptrTargetHead = targetHead.GetHostData();
    INT *ptrTargetTail = targetTail.GetHostData();
    INT *ptrTargetState = targetState.GetHostData();

    for(unsigned long i = 0; i < nStream; i++)
    {
        ptrTargetHead[i] = 0;
        ptrTargetTail[i] = -1; /* -1: empty queue */
        for(unsigned long frameIdx = 0; frameIdx < nUnroll; frameIdx++)
        {
            ptrTargetState[nStream * frameIdx + i] = (INT) 0;
        }
    }

    targetHead.HostPush();
    targetTail.HostPush();
    targetState.HostPush();

    /* Forward and backward variables */
    forward.Resize(2 * maxTargetLen + 1, nStream * nUnroll);
    backward.Resize(2 * maxTargetLen + 1, nStream * nUnroll);
}


void CTCProbe::InitEvaluation(const unsigned long nStream, const unsigned long nUnroll)
{
    TrainableProbe::InitEvaluation(nStream, nUnroll);

    /* Target ring buffer */
    targetBuf.Resize(nStream, maxTargetLen);
    targetHead.Resize(nStream, 1);
    targetTail.Resize(nStream, 1);
    targetState.Resize(nStream, 1);
    targetCount.Resize(nStream, 1);

    INT *ptrTargetHead = targetHead.GetHostData();
    INT *ptrTargetTail = targetTail.GetHostData();
    INT *ptrTargetState = targetState.GetHostData();

    for(unsigned long i = 0; i < nStream; i++)
    {
        ptrTargetHead[i] = 0;
        ptrTargetTail[i] = -1; /* -1: empty queue */
        ptrTargetState[i] = (INT) 0;
    }

    targetHead.HostPush();
    targetTail.HostPush();
    targetState.HostPush();

    /* Output ring buffer */
    outputBuf.Resize(nStream, maxTargetLen);
    outputHead.Resize(nStream, 1);
    outputTail.Resize(nStream, 1);

    INT *ptrOutputHead = outputHead.GetHostData();
    INT *ptrOutputTail = outputTail.GetHostData();

    for(unsigned long i = 0; i < nStream; i++)
    {
        ptrOutputHead[i] = 0;
        ptrOutputTail[i] = -1; /* -1: empty queue */
    }

    outputHead.HostPush();
    outputTail.HostPush();

    /* Forward variable is needed to compute likelihood */
    forward.Resize(2 * maxTargetLen + 1, nStream);

    /* For best path decoding */
    prevMaxIdx.Resize(nStream, 1);

    /* For computing edit distance */
    distanceMat.Resize(maxTargetLen + 1, nStream);
    substitutionMat.Resize(maxTargetLen + 1, nStream);
    deletionMat.Resize(maxTargetLen + 1, nStream);
    insertionMat.Resize(maxTargetLen + 1, nStream);
}


void CTCProbe::Dequeue(const unsigned long idxFrom, const unsigned long idxTo)
{
    INT *ptrTargetHead = targetHead.GetHostData();
    INT *ptrTargetTail = targetTail.GetHostData();
    INT *ptrTargetState = targetState.GetHostData();
    INT *ptrTargetStartIdx = targetStartIdx.GetHostData();
    INT *ptrTargetCount = targetCount.GetHostData();

    /* Dequeue */
    #ifdef FRACTAL_USE_OMP
    #pragma omp parallel for
    #endif
    for(long streamIdx = 0; streamIdx < (long) nStream; streamIdx++)
    {
        for(long frameIdx = (long) idxFrom; frameIdx < (long) idxTo; frameIdx++)
        {
            if((ptrTargetState[nStream * frameIdx + streamIdx] & STATE_COVERAGE_END) != (INT) 0) /* Coverage end */
            {
                ptrTargetTail[streamIdx] = (ptrTargetStartIdx[nStream * frameIdx + streamIdx] + ptrTargetCount[nStream * frameIdx + streamIdx]) % maxTargetLen;

                if(ptrTargetTail[streamIdx] == ptrTargetHead[streamIdx]) /* Empty */
                    ptrTargetTail[streamIdx] = -1;
            }
        }
    }

    targetTail.HostPush();
}


const FLOAT CTCProbe::LogSumExpN(const FLOAT *x, unsigned long n)
{
    verify(n >= 1);

    if(n == 1) return x[0];

    FLOAT _max = std::max(x[0], x[1]);
    FLOAT sum = (FLOAT) 0;

    for(unsigned long i = 2; i < n; i++)
    {
        _max = std::max(_max, x[i]);
    }

    if(isinf(_max)) return _max;

    for(unsigned long i = 0; i < n; i++)
    {
        sum += exp(x[i] - _max);
    }

    return log(sum) + _max;
}


CTCProbe::EditDistance CTCProbe::ComputeEditDistance(const unsigned long streamIdx)
{
    /* Wagner-Fischer algorithm */

    EditDistance distance;

    Matrix<INT> curDistMat(distanceMat, streamIdx, streamIdx);
    Matrix<INT> curSubMat(substitutionMat, streamIdx, streamIdx);
    Matrix<INT> curDelMat(deletionMat, streamIdx, streamIdx);
    Matrix<INT> curInsMat(insertionMat, streamIdx, streamIdx);

    INT *dist = curDistMat.GetHostData();
    INT *sub = curSubMat.GetHostData();
    INT *del = curDelMat.GetHostData();
    INT *ins = curInsMat.GetHostData();

    INT curTargetHead = targetHead.GetHostData()[streamIdx];
    INT curTargetTail = targetTail.GetHostData()[streamIdx];
    INT curOutputHead = outputHead.GetHostData()[streamIdx];
    INT curOutputTail = outputTail.GetHostData()[streamIdx];

    INT *ptrTargetBuf = targetBuf.GetHostData();
    INT *ptrOutputBuf = outputBuf.GetHostData();

    unsigned long nTargetLabel;
    unsigned long nOutputLabel;
    unsigned long nTargetSkip = 0;

    if(curTargetTail == (INT) -1)
    {
        nTargetLabel = 0;
    }
    else
    {
        nTargetLabel = ((curTargetHead - curTargetTail + maxTargetLen - 1) % maxTargetLen) + 1;
    }

    if(curOutputTail == (INT) -1)
    {
        nOutputLabel = 0;
    }
    else
    {
        nOutputLabel = ((curOutputHead - curOutputTail + maxTargetLen - 1) % maxTargetLen) + 1;
    }

#ifdef CTC_EDIT_DISTANCE_DEBUG
    for(unsigned long i = 1; i <= nTargetLabel; i++)
    {
        INT curTargetLabel = ptrTargetBuf[((curTargetTail + i - 1) % maxTargetLen) * nStream + streamIdx];
        if(streamIdx == 0) std::cout << curTargetLabel << " ";
    }
    if(streamIdx == 0) std::cout << std::endl;

    for(unsigned long j = 1; j <= nOutputLabel; j++)
    {
        INT curOutputLabel = ptrOutputBuf[((curOutputTail + j - 1) % maxTargetLen) * nStream + streamIdx];
        if(streamIdx == 0) std::cout << curOutputLabel << " ";
    }
    if(streamIdx == 0) std::cout << std::endl;
#endif


    dist[0] = 0;
    sub[0] = 0;
    ins[0] = 0;
    del[0] = 0;


    for(unsigned long i = 1; i <= nTargetLabel; i++)
    {
        INT curTargetLabel = ptrTargetBuf[((curTargetTail + i - 1) % maxTargetLen) * nStream + streamIdx];

        if(labelGroup.size() > 0)
        {
            verify(curTargetLabel < (INT) labelGroup.size());

            curTargetLabel = labelGroup[curTargetLabel];
        }

        dist[i] = dist[i - 1] + (curTargetLabel != -1);
        sub[i] = 0;
        ins[i] = 0;
        del[i] = dist[i];
    }

    for(unsigned long j = 1; j <= nOutputLabel; j++)
    {
        INT curOutputLabel = ptrOutputBuf[((curOutputTail + j - 1) % maxTargetLen) * nStream + streamIdx];

        if(labelGroup.size() > 0)
        {
            verify(curOutputLabel < (INT) labelGroup.size());

            curOutputLabel = labelGroup[curOutputLabel];
        }

        INT dist_im1_jm1 = dist[0];
        INT sub_im1_jm1 = 0;
        INT del_im1_jm1 = 0;
        INT ins_im1_jm1 = ins[0];
    
        dist[0] = dist[0] + (curOutputLabel != -1);
        sub[0] = 0;
        del[0] = 0;
        ins[0] = dist[0];

        for(unsigned long i = 1; i <= nTargetLabel; i++)
        {
            INT curTargetLabel = ptrTargetBuf[((curTargetTail + i - 1) % maxTargetLen) * nStream + streamIdx];

            if(labelGroup.size() > 0)
            {
                verify(curTargetLabel < (INT) labelGroup.size());

                curTargetLabel = labelGroup[curTargetLabel];
            }

            INT dist_im1_j = dist[i - 1];
            INT sub_im1_j = sub[i - 1];
            INT del_im1_j = del[i - 1];
            INT ins_im1_j = ins[i - 1];

            INT dist_i_jm1 = dist[i];
            INT sub_i_jm1 = sub[i];
            INT del_i_jm1 = del[i];
            INT ins_i_jm1 = ins[i];


            if(curTargetLabel == -1)
            {
                /* Skip (deletion without cost) */
                dist[i] = dist_im1_j;
                sub[i] = sub_im1_j;
                del[i] = del_im1_j;
                ins[i] = ins_im1_j;
            }
            else if(curOutputLabel == -1)
            {
                /* Skip (insertion without cost) */
                dist[i] = dist_i_jm1;
                sub[i] = sub_i_jm1;
                del[i] = del_i_jm1;
                ins[i] = ins_i_jm1;
            }
            else if(curTargetLabel == curOutputLabel)
            {
                /* Correct */
                dist[i] = dist_im1_jm1;
                sub[i] = sub_im1_jm1;
                del[i] = del_im1_jm1;
                ins[i] = ins_im1_jm1;
            }
            else if((dist_im1_jm1 < dist_im1_j) && (dist_im1_jm1 < dist_i_jm1))
            {
                /* Substitution */
                dist[i] = dist_im1_jm1 + 1;
                sub[i] = sub_im1_jm1 + 1;
                del[i] = del_im1_jm1;
                ins[i] = ins_im1_jm1;
            }
            else if(dist_im1_j < dist_i_jm1)
            {
                /* Deletion */
                dist[i] = dist_im1_j + 1;
                sub[i] = sub_im1_j;
                del[i] = del_im1_j + 1;
                ins[i] = ins_im1_j;
            }
            else
            {
                /* Insertion */
                dist[i] = dist_i_jm1 + 1;
                sub[i] = sub_i_jm1;
                del[i] = del_i_jm1;
                ins[i] = ins_i_jm1 + 1;
            }

            dist_im1_jm1 = dist_i_jm1;
            sub_im1_jm1 = sub_i_jm1;
            del_im1_jm1 = del_i_jm1;
            ins_im1_jm1 = ins_i_jm1;
        }
    }

    for(unsigned long i = 1; i <= nTargetLabel; i++)
    {
        INT curTargetLabel = ptrTargetBuf[((curTargetTail + i - 1) % maxTargetLen) * nStream + streamIdx];

        if(labelGroup.size() > 0)
        {
            verify(curTargetLabel < (INT) labelGroup.size());

            curTargetLabel = labelGroup[curTargetLabel];
        }

        if(curTargetLabel == -1) nTargetSkip++;
    }

    distance.nSubstitutions = sub[nTargetLabel];
    distance.nDeletions = del[nTargetLabel];
    distance.nInsertions = ins[nTargetLabel];
    distance.nSymbols = nTargetLabel - nTargetSkip;

#ifdef CTC_EDIT_DISTANCE_DEBUG
    if(streamIdx == 0) std::cout << "S:" << distance.nSubstitutions << " ";
    if(streamIdx == 0) std::cout << "D:" << distance.nDeletions << " ";
    if(streamIdx == 0) std::cout << "I:" << distance.nInsertions << " ";
    if(streamIdx == 0) std::cout << "N:" << distance.nSymbols << std::endl;
#endif

    return distance;
}


CTCProbe::EditDistance CTCProbe::ComputeWordEditDistance(const unsigned long streamIdx)
{
    /* Wagner-Fischer algorithm for computing word error rates */

    EditDistance distance;

    Matrix<INT> curDistMat(distanceMat, streamIdx, streamIdx);
    Matrix<INT> curSubMat(substitutionMat, streamIdx, streamIdx);
    Matrix<INT> curDelMat(deletionMat, streamIdx, streamIdx);
    Matrix<INT> curInsMat(insertionMat, streamIdx, streamIdx);

    INT *dist = curDistMat.GetHostData();
    INT *sub = curSubMat.GetHostData();
    INT *del = curDelMat.GetHostData();
    INT *ins = curInsMat.GetHostData();

    INT curTargetHead = targetHead.GetHostData()[streamIdx];
    INT curTargetTail = targetTail.GetHostData()[streamIdx];
    INT curOutputHead = outputHead.GetHostData()[streamIdx];
    INT curOutputTail = outputTail.GetHostData()[streamIdx];

    INT *ptrTargetBuf = targetBuf.GetHostData();
    INT *ptrOutputBuf = outputBuf.GetHostData();

    unsigned long nTargetLabel;
    unsigned long nOutputLabel;

    if(curTargetTail == (INT) -1)
    {
        nTargetLabel = 0;
    }
    else
    {
        nTargetLabel = ((curTargetHead - curTargetTail + maxTargetLen - 1) % maxTargetLen) + 1;
    }

    if(curOutputTail == (INT) -1)
    {
        nOutputLabel = 0;
    }
    else
    {
        nOutputLabel = ((curOutputHead - curOutputTail + maxTargetLen - 1) % maxTargetLen) + 1;
    }

#ifdef CTC_WORD_EDIT_DISTANCE_DEBUG
    for(unsigned long i = 1; i <= nTargetLabel; i++)
    {
        INT curTargetLabel = ptrTargetBuf[((curTargetTail + i - 1) % maxTargetLen) * nStream + streamIdx];
        if(streamIdx == 0) std::cout << curTargetLabel << " ";
    }
    if(streamIdx == 0) std::cout << std::endl;

    for(unsigned long j = 1; j <= nOutputLabel; j++)
    {
        INT curOutputLabel = ptrOutputBuf[((curOutputTail + j - 1) % maxTargetLen) * nStream + streamIdx];
        if(streamIdx == 0) std::cout << curOutputLabel << " ";
    }
    if(streamIdx == 0) std::cout << std::endl;
#endif


    unsigned long outputWordLen = 0;
    unsigned long outputWordCount = 0;
    unsigned long targetWordCount;

    for(unsigned long j = 0; j <= nOutputLabel; j++)
    {
        /* Even though j == 0, there is no problem */
        INT curOutputLabel = ptrOutputBuf[((curOutputTail + j - 1) % maxTargetLen) * nStream + streamIdx];
        INT nextOutputLabel = ptrOutputBuf[((curOutputTail + j) % maxTargetLen) * nStream + streamIdx];

        verify(curOutputLabel >= 0);
        verify(curOutputLabel < (INT) wordDelimiter.size());

        if(j == 0 || wordDelimiter[curOutputLabel] == true)
        {
            outputWordLen = 0;
        }
        else
        {
            if(labelGroup.size() > 0)
            {
                verify(curOutputLabel < (INT) labelGroup.size());

                curOutputLabel = labelGroup[curOutputLabel];
            }

            outputWordLen += (curOutputLabel != -1);
        }

        bool outputWordEnd = (outputWordLen > 0) && (j == nOutputLabel || wordDelimiter[nextOutputLabel] == true);

        if(outputWordEnd == true)
        {
            outputWordCount++;
        }

        unsigned long targetWordLen = 0;
        INT dist_i_jm1 = 0;
        INT sub_wi_wjm1 = 0;
        INT del_wi_wjm1 = 0;
        INT ins_wi_wjm1 = 0;

        targetWordCount = 0;

        for(unsigned long i = 0; i <= nTargetLabel; i++)
        {
            INT dist_im1_jm1 = dist_i_jm1;
            dist_i_jm1 = dist[i];

            /* Even though i == 0, there is no problem */
            INT curTargetLabel = ptrTargetBuf[((curTargetTail + i - 1) % maxTargetLen) * nStream + streamIdx];
            INT nextTargetLabel = ptrTargetBuf[((curTargetTail + i) % maxTargetLen) * nStream + streamIdx];

            verify(curTargetLabel >= 0);
            verify(curTargetLabel < (INT) wordDelimiter.size());

            if(i == 0 || wordDelimiter[curTargetLabel] == true)
            {
                targetWordLen = 0;
            }
            else
            {
                if(labelGroup.size() > 0)
                {
                    verify(curTargetLabel < (INT) labelGroup.size());

                    curTargetLabel = labelGroup[curTargetLabel];
                }

                targetWordLen += (curTargetLabel != -1);
            }

            /* Label-level dynamic programming */
            if(outputWordLen == 0)
            {
                dist[i] = targetWordLen;
            }
            else if(targetWordLen == 0)
            {
                dist[i] = outputWordLen;
            }
            else
            {
                INT dist_im1_j = dist[i - 1];

                if(curTargetLabel == -1)
                {
                    /* Skip (deletion without cost) */
                    dist[i] = dist_im1_j;
                }
                else if(curOutputLabel == -1)
                {
                    /* Skip (insertion without cost) */
                    dist[i] = dist_i_jm1;
                }
                else if(curTargetLabel == curOutputLabel)
                {
                    /* Correct */
                    dist[i] = dist_im1_jm1;
                }
                else if((dist_im1_jm1 <= dist_im1_j) && (dist_im1_jm1 <= dist_i_jm1))
                {
                    /* Substitution */
                    dist[i] = dist_im1_jm1 + 1;
                }
                else if(dist_im1_j < dist_i_jm1)
                {
                    /* Deletion */
                    dist[i] = dist_im1_j + 1;
                }
                else
                {
                    /* Insertion */
                    dist[i] = dist_i_jm1 + 1;
                }
            }

            bool targetWordEnd = (targetWordLen > 0) && (i == nTargetLabel || wordDelimiter[nextTargetLabel] == true);

            if(targetWordEnd == true)
            {
                targetWordCount++;
            }

            /* Word-level dynamic programming */
            if(j == 0 && i == 0)
            {
                sub[0] = 0;
                del[0] = 0;
                ins[0] = 0;
            }
            else if(j == 0 && targetWordEnd)
            {
                sub[targetWordCount] = 0;
                del[targetWordCount] = targetWordCount;
                ins[targetWordCount] = 0;
            }
            else if(i == 0 && outputWordEnd)
            {
                sub_wi_wjm1 = sub[0];
                del_wi_wjm1 = del[0];
                ins_wi_wjm1 = ins[0];

                sub[0] = 0;
                del[0] = 0;
                ins[0] = outputWordCount;
            }
            else if(targetWordEnd && outputWordEnd)
            {
                INT sub_wim1_wjm1 = sub_wi_wjm1;
                INT del_wim1_wjm1 = del_wi_wjm1;
                INT ins_wim1_wjm1 = ins_wi_wjm1;

                sub_wi_wjm1 = sub[targetWordCount];
                del_wi_wjm1 = del[targetWordCount];
                ins_wi_wjm1 = ins[targetWordCount];

                INT sub_wim1_wj = sub[targetWordCount - 1];
                INT del_wim1_wj = del[targetWordCount - 1];
                INT ins_wim1_wj = ins[targetWordCount - 1];

                INT dist_wim1_wjm1 = sub_wim1_wjm1 + del_wim1_wjm1 + ins_wim1_wjm1;
                INT dist_wi_wjm1 = sub_wi_wjm1 + del_wi_wjm1 + ins_wi_wjm1;
                INT dist_wim1_wj = sub_wim1_wj + del_wim1_wj + ins_wim1_wj;

                if(dist[i] == 0)
                {
                    /* Correct */
                    sub[targetWordCount] = sub_wim1_wjm1;
                    del[targetWordCount] = del_wim1_wjm1;
                    ins[targetWordCount] = ins_wim1_wjm1;
                }
                else if((dist_wim1_wjm1 <= dist_wim1_wj) && (dist_wim1_wjm1 <= dist_wi_wjm1))
                {
                    /* Substitution */
                    sub[targetWordCount] = sub_wim1_wjm1 + 1;
                    del[targetWordCount] = del_wim1_wjm1;
                    ins[targetWordCount] = ins_wim1_wjm1;
                }
                else if(dist_wim1_wj < dist_wi_wjm1)
                {
                    /* Deletion */
                    sub[targetWordCount] = sub_wim1_wj;
                    del[targetWordCount] = del_wim1_wj + 1;
                    ins[targetWordCount] = ins_wim1_wj;
                }
                else
                {
                    /* Insertion */
                    sub[targetWordCount] = sub_wi_wjm1;
                    del[targetWordCount] = del_wi_wjm1;
                    ins[targetWordCount] = ins_wi_wjm1 + 1;
                }
            }
        }
    }

    distance.nSubstitutions = sub[targetWordCount];
    distance.nDeletions = del[targetWordCount];
    distance.nInsertions = ins[targetWordCount];
    distance.nSymbols = targetWordCount;

#ifdef CTC_WORD_EDIT_DISTANCE_DEBUG
    if(streamIdx == 0) std::cout << "WS:" << distance.nSubstitutions << " ";
    if(streamIdx == 0) std::cout << "WD:" << distance.nDeletions << " ";
    if(streamIdx == 0) std::cout << "WI:" << distance.nInsertions << " ";
    if(streamIdx == 0) std::cout << "WN:" << distance.nSymbols << std::endl;
#endif

    return distance;
}


}

