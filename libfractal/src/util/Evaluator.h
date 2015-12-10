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


#ifndef FRACTAL_EVALUATOR_H_
#define FRACTAL_EVALUATOR_H_

#include "Pipe.h"
#include "PortMap.h"
#include "Stream.h"
#include "../core/TrainableProbe.h"
#include "../core/MultiTypeMatrix.h"
#include "../core/Rnn.h"
#include "../core/FractalCommon.h"


namespace fractal
{

class EvaluateArgs
{
public:
    Rnn *rnn;
    Stream *stream;

    unsigned long numFrame;
    unsigned long nStream;
    unsigned long frameStep;
    unsigned long nInput;
    unsigned long nOutput;

    InputProbe **inputProbe;
    TrainableProbe **outputProbe;

    unsigned long *inputChannel;
    unsigned long *outputChannel;

    MultiTypeMatrix *input, *inputBuf;
    Matrix<FLOAT> *output, *outputBuf;
    MultiTypeMatrix *target;
    MultiTypeMatrix *targetPipe1, *targetPipe2, *targetPipe3, *targetPipe4, *targetPipe5;
};


class Evaluator
{
public:
    Evaluator();

    void Evaluate(Rnn &rnn, Stream &stream, const PortMapList &inputPorts, const PortMapList &outputPorts,
            const unsigned long numFrame, const unsigned long stepSize);

    inline void SetInitFrameSkip(const unsigned long val) { initFrameSkip = val; }
    inline const unsigned long GetInitFrameSkip() const { return initFrameSkip; }


protected:
    static void EvaluatePipe0(Evaluator *evaluator, EvaluateArgs &args);
    static void EvaluatePipe1(Evaluator *evaluator, EvaluateArgs &args);
    static void EvaluatePipe2(Evaluator *evaluator, EvaluateArgs &args);
    static void EvaluatePipe3(Evaluator *evaluator, EvaluateArgs &args);
    static void EvaluatePipe4(Evaluator *evaluator, EvaluateArgs &args);
    static void EvaluatePipe5(Evaluator *evaluator, EvaluateArgs &args);
    static void EvaluatePipe6(Evaluator *evaluator, EvaluateArgs &args);

    Pipe pipe[7];
    PStream pStreamDataTransferToBuf;
    PStream pStreamDataTransferToRnn;
    PStream pStreamDataTransferFromRnn;
    PStream pStreamDataTransferFromBuf;
    PEvent pEventDataTransferToBuf;
    PEvent pEventDataTransferToRnn;
    PEvent pEventDataTransferFromRnn;
    PEvent pEventDataTransferFromBuf;

    unsigned long initFrameSkip;
};

}

#endif /* FRACTAL_EVALUATOR_H_ */

