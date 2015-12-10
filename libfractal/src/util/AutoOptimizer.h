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


#ifndef FRACTAL_AUTOOPTIMIZER_H_
#define FRACTAL_AUTOOPTIMIZER_H_


#include <string>

#include "Optimizer.h"
#include "Evaluator.h"
#include "../core/TrainableProbe.h"
#include "../core/FractalCommon.h"


namespace fractal
{

class AutoOptimizer
{
public:
    AutoOptimizer();
    virtual ~AutoOptimizer() {}

    void Optimize(Rnn &rnn, Stream &trainStream, Stream &evalStream,
            const PortMapList &inputPorts, const PortMapList &outputPorts,
            const unsigned long nTrainFramePerEpoch, const unsigned long nEvalFramePerEpoch,
            const unsigned long windowSize, const unsigned long stepSize);

    void SetWorkspacePath(const std::string &path) { workspacePath = path; }
    void SetInitLearningRate(const FLOAT val) { initLearningRate = val; }
    void SetMinLearningRate(const FLOAT val) { minLearningRate = val; }
    void SetMomentum(const FLOAT val) { momentum = val; }
    void SetWeightNoise(const FLOAT val) { weightNoise = val; }
    void SetRmsprop(const bool val) { rmsprop = val; }
    void SetAdadelta(const bool val) { adadelta = val; }
    void SetRmsDecayRate(const FLOAT val) { rmsDecayRate = val; }
    void SetMaxRetryCount(const unsigned long val) { maxRetryCount = val; }
    void SetLearningRateDecayRate(const FLOAT val) { learningRateDecayRate = val; }
    void SetInitFrameSkip(const unsigned long val) { initFrameSkip = val; }

    void SetLambdaLoss(std::function<double (std::vector<TrainableProbe *> &)> lambda) { lambdaLoss = lambda; }
    void SetLambdaPostEval(std::function<void (std::vector<TrainableProbe *> &)> lambda) {lambdaPostEval = lambda; }

    const std::string &GetWorkspacePath() const { return workspacePath; }
    const FLOAT GetInitLearningRate() const { return initLearningRate; }
    const FLOAT GetMinLearningRate() const { return minLearningRate; }
    const FLOAT GetMomentum() const { return momentum; }
    const FLOAT GetWeightNoise() const { return weightNoise; }
    const bool GetRmsprop() const { return rmsprop; }
    const bool GetAdadelta() const { return adadelta; }
    const FLOAT GetRmsDecayRate() const { return rmsDecayRate; }
    const unsigned long GetMaxRetryCount() const { return maxRetryCount; }
    const FLOAT GetLearningRateDecayRate() const { return learningRateDecayRate; }
    const unsigned long GetInitFrameSkip() const { return initFrameSkip; }


protected:
    std::function<double (std::vector<TrainableProbe *> &probe)> lambdaLoss;
    std::function<void (std::vector<TrainableProbe *> &probe)> lambdaPostEval;

    std::string workspacePath;

    FLOAT initLearningRate;
    FLOAT minLearningRate;
    FLOAT momentum;

    FLOAT weightNoise;

    bool rmsprop;
    bool adadelta;
    FLOAT rmsDecayRate;

    FLOAT learningRateDecayRate;

    unsigned long maxRetryCount;
    unsigned long initFrameSkip;
};

}

#endif /* FRACTAL_AUTOOPTIMIZER_H_ */

