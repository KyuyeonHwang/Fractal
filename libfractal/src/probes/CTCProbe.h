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


#ifndef FRACTAL_CTCPROBE_H_
#define FRACTAL_CTCPROBE_H_

#include <vector>

#include "../core/TrainableProbe.h"
#include "../core/FractalCommon.h"


namespace fractal
{

class CTCProbe : public TrainableProbe
{
public:
    CTCProbe();

    void InitTraining(const unsigned long nStream, const unsigned long nUnroll);
    void InitEvaluation(const unsigned long nStream, const unsigned long nUnroll);
    void SetTarget(MultiTypeMatrix &mat, const unsigned long idxFrom, const unsigned long idxTo);
    void SetQueueDepth(const unsigned long depth);
    void SetLabelGroup(std::vector<INT> &group);
    void SetWordDelimiter(std::vector<bool> &delimiter);
    void ComputeErr(const unsigned long idxFrom, const unsigned long idxTo);
    void EvaluateOnHost(MultiTypeMatrix &target, Matrix<FLOAT> &output);
    void ResetStatistics();

    const double GetLoss();

    const double GetLabelErrorRate();
    const double GetLabelSubstitutionRate();
    const double GetLabelDeletionRate();
    const double GetLabelInsertionRate();

    const double GetWordErrorRate();
    const double GetWordSubstitutionRate();
    const double GetWordDeletionRate();
    const double GetWordInsertionRate();

    void PrintStatistics(std::ostream &outStream);

    void SetForceBlankFirst(const bool val);
    void SetResidualTraining(const bool val);

protected:
    typedef struct _EditDistance
    {
        _EditDistance() : nSubstitutions(0), nDeletions(0), nInsertions(0), nSymbols(0) {};

        unsigned long nSubstitutions;
        unsigned long nDeletions;
        unsigned long nInsertions;
        unsigned long nSymbols;
    } EditDistance;

    virtual void SetEngine(Engine *engine);
    
    void Dequeue(const unsigned long idxFrom, const unsigned long idxTo);

    const FLOAT LogSumExpN(const FLOAT *x, unsigned long n);

    inline const FLOAT LogSumExp2(const FLOAT x, const FLOAT y)
    {
        FLOAT _max = std::max(x, y);
        return isinf(_max) ? _max : log(exp(x - _max) + exp(y - _max)) + _max;
    }

    inline const FLOAT LogSumExp3(const FLOAT x, const FLOAT y, const FLOAT z)
    {
        FLOAT _max = std::max(std::max(x, y), z);
        return isinf(_max) ? _max : log(exp(x - _max) + exp(y - _max) + exp(z - _max)) + _max;
    }

    EditDistance ComputeEditDistance(const unsigned long streamIdx);
    EditDistance ComputeWordEditDistance(const unsigned long streamIdx);

    /* Ring buffers */
    Matrix<INT> targetBuf;
    Matrix<INT> targetHead;
    Matrix<INT> targetTail;
    Matrix<INT> targetState;
    Matrix<INT> targetStartIdx;
    Matrix<INT> targetCount;

    Matrix<INT> outputBuf;
    Matrix<INT> outputHead;
    Matrix<INT> outputTail;

    /* Forward and backward variables */
    Matrix<FLOAT> forward;
    Matrix<FLOAT> backward;

    /* For best path decoding */
    Matrix<INT> prevMaxIdx;

    /* For computing edit distance */
    Matrix<INT> distanceMat;
    Matrix<INT> substitutionMat;
    Matrix<INT> deletionMat;
    Matrix<INT> insertionMat;

    unsigned long maxTargetLen;

    EditDistance labelEditDistance;
    EditDistance wordEditDistance;

    unsigned long nSeq;
    double lossSum;

    bool forceBlankFirst;
    bool residualTraining;

    std::vector<INT> labelGroup;
    std::vector<bool> wordDelimiter;
};

}

#endif /* FRACTAL_CTCPROBE_H_ */

