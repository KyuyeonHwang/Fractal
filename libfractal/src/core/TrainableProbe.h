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


#ifndef FRACTAL_TRAINABLEPROBE_H_
#define FRACTAL_TRAINABLEPROBE_H_

#include <ostream>

#include "MultiTypeMatrix.h"
#include "Probe.h"
#include "FractalCommon.h"


namespace fractal
{

class TrainableProbe : public OutputProbe
{
public:
    TrainableProbe(const bool injectToSrcErr) : OutputProbe(), injectToSrcErr(injectToSrcErr), nStream(0), nUnroll(0) {}

    virtual void InitTraining(const unsigned long nStream, const unsigned long nUnroll);
    virtual void InitEvaluation(const unsigned long nStream, const unsigned long nUnroll);

    virtual void SetTarget(MultiTypeMatrix &mat, const unsigned long idxFrom, const unsigned long idxTo) = 0;
    virtual void ComputeErr(const unsigned long idxFrom, const unsigned long idxTo) = 0;
    virtual void EvaluateOnHost(MultiTypeMatrix &target, Matrix<FLOAT> &output) = 0;
    virtual void ResetStatistics() = 0;
    virtual const double GetLoss() = 0;
    virtual void PrintStatistics(std::ostream &outStream) = 0;

    virtual void InitErr(const unsigned long idxFrom, const unsigned long idxTo);
    inline const bool InjectToSrcErr() const { return injectToSrcErr; }
    inline Matrix<FLOAT> &GetErr() { return err; }

protected:
    virtual void SetEngine(Engine *engine);

    Matrix<FLOAT> err;
    const bool injectToSrcErr;
    unsigned long nStream;
    unsigned long nUnroll;
};

}

#endif /* FRACTAL_TRAINABLEPROBE_H_ */

