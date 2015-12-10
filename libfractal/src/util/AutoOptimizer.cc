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


#include "AutoOptimizer.h"


#include <chrono>
#include <iostream>
#include <cmath>


namespace fractal
{

AutoOptimizer::AutoOptimizer()
{
    workspacePath = "workspace";
    initLearningRate = 1e-5;
    minLearningRate = 1e-7;
    momentum = 0.9;
    weightNoise = 0.0;

    adadelta = false;
    rmsprop = false;
    rmsDecayRate = 0.95;

    maxRetryCount = 4;

    learningRateDecayRate = 0.5;

    initFrameSkip = 0;

    lambdaLoss = [] (std::vector<TrainableProbe *> &probe) -> double
    {
        double loss = 0.0;

        for(unsigned long i = 0; i < probe.size(); i++)
        {
            loss += probe[i]->GetLoss();
        }

        return loss;
    };

    lambdaPostEval = [] (std::vector<TrainableProbe *> &probe) -> void
    {
        for(unsigned long i = 0; i < probe.size(); i++)
        {
            std::cout << "  [" << probe[i]->GetLayerName() << "]  ";
            probe[i]->PrintStatistics(std::cout);
            std::cout << std::endl;
        }
    };
}


void AutoOptimizer::Optimize(Rnn &rnn, Stream &trainStream, Stream &evalStream,
        const PortMapList &inputPorts, const PortMapList &outputPorts,
        const unsigned long nTrainFramePerEpoch, const unsigned long nEvalFramePerEpoch,
        const unsigned long windowSize, const unsigned long stepSize)
{
    std::string pivotPath = workspacePath + "/net/0/";
    std::string prevPath = workspacePath + "/net/1/";
    std::string bestPath = workspacePath + "/net/best/";

    FLOAT learningRate;
    unsigned long evalStepSize;
    unsigned long retryCount;
    unsigned long totalTrainedFrame, totalTrainedFrameAtPivot, totalTrainedFrameAtBest, totalDiscardedFrame;
    unsigned long _nTrainFramePerEpoch, _nEvalFramePerEpoch;
    double prevLoss, pivotLoss, bestLoss;

    std::vector<TrainableProbe *> outputProbe(outputPorts.size());

    Optimizer optimizer;
    Evaluator evaluator;

    {
        unsigned long i = 0;

        for(auto &port : outputPorts)
        {
            outputProbe[i] = dynamic_cast<TrainableProbe *>(std::get<0>(port));
            verify(outputProbe[i] != NULL);
            i++;
        }
    }

    optimizer.SetLearningRate(initLearningRate);
    optimizer.SetMomentum(momentum);
    optimizer.SetWeightNoise(weightNoise);
    optimizer.SetRmsprop(rmsprop);
    optimizer.SetAdadelta(adadelta);
    optimizer.SetInitFrameSkip(initFrameSkip);

    evaluator.SetInitFrameSkip(initFrameSkip);

    rnn.SetBatchSize(0, 0);

    rnn.InitNesterov();

    {
        unsigned long chunkSize;

        chunkSize = trainStream.GetNumStream() * stepSize;
        _nTrainFramePerEpoch = (nTrainFramePerEpoch / chunkSize) * chunkSize;

        chunkSize = evalStream.GetNumStream();
        _nEvalFramePerEpoch = (nEvalFramePerEpoch / chunkSize) * chunkSize;
    }

    if(adadelta == true)
    {
        rnn.InitAdadelta(rmsDecayRate, true);

        verify(rmsprop == false);
    }
    else
    {
        if(rmsprop == true)
            rnn.InitRmsprop(rmsDecayRate);
    }


    rnn.Ready();

    evalStepSize = windowSize * trainStream.GetNumStream() / evalStream.GetNumStream();
    learningRate = initLearningRate;
    totalTrainedFrame = 0;
    totalTrainedFrameAtBest = 0;
    totalTrainedFrameAtPivot = 0;
    totalDiscardedFrame = 0;
    bestLoss = 0.0;
    pivotLoss = 0.0;
    prevLoss = 0.0;
    retryCount = 0;

    rnn.SaveState(prevPath);
    rnn.SaveState(bestPath);

    std::cout << "======================================================================" << std::endl;
    std::cout << "                            Auto Optimizer                            " << std::endl;
    std::cout << "======================================================================" << std::endl;
    std::cout << std::endl;
    std::cout << "-------------------------- General Settings --------------------------" << std::endl;
    std::cout << "Workspace path: " << workspacePath << std::endl;
    std::cout << "Maximum retry count: " << maxRetryCount << std::endl;
    std::cout << "Decay rate of the learning rate: " << learningRateDecayRate << std::endl;
    std::cout << "Number of initial frames to skip: " << initFrameSkip << std::endl;
    std::cout << "----------------------------------------------------------------------" << std::endl;
    std::cout << std::endl;
    std::cout << "------------------------- Training Parameters ------------------------" << std::endl;
    std::cout << "Initial learning rate: " << initLearningRate << std::endl;
    std::cout << "Minimum learning rate: " << minLearningRate << std::endl;
    std::cout << "Momentum: " << momentum << std::endl;
    std::cout << "Weight noise: " << weightNoise << std::endl;

    std::cout << "AdaDelta: " << (adadelta == true ? "enabled" : "disabled") << std::endl;
    std::cout << "RMSProp: " << (rmsprop == true ? "enabled" : "disabled") << std::endl;
    if(rmsprop == true || adadelta == true) std::cout << "RMS decay rate: " << rmsDecayRate << std::endl;

    std::cout << "Number of frames per epoch: " << _nTrainFramePerEpoch << std::endl;
    std::cout << "Number of streams: " << trainStream.GetNumStream() << std::endl;
    std::cout << "Forward step size: " << stepSize << std::endl;
    std::cout << "Backward window size: " << windowSize << std::endl;
    std::cout << "----------------------------------------------------------------------" << std::endl;
    std::cout << std::endl;
    std::cout << "------------------------ Evaluation Parameters -----------------------" << std::endl;
    std::cout << "Number of frames per epoch: " << _nEvalFramePerEpoch << std::endl;
    std::cout << "Number of streams: " << evalStream.GetNumStream() << std::endl;
    std::cout << "Forward step size: " << evalStepSize << std::endl;
    std::cout << "----------------------------------------------------------------------" << std::endl;
    std::cout << std::endl;

    while(true)
    {
        std::cout << "----------------------------------------------------------------------" << std::endl;

        /* Training */
        std::cout << "Training ...  ";
        std::cout.flush();
        {
            trainStream.Reset();

            auto t1 = std::chrono::steady_clock::now();
            optimizer.Backprop(rnn, trainStream, inputPorts, outputPorts, _nTrainFramePerEpoch, windowSize, stepSize);
            auto t2 = std::chrono::steady_clock::now();

            std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

            std::cout << "(" << time_span.count() << " sec)" << std::endl;
        }

        totalTrainedFrame += _nTrainFramePerEpoch;


        /* Evaluation */
        std::cout << "Evaluating ...  ";
        std::cout.flush();
        {
            for(auto &probe : outputProbe)
            {
                probe->ResetStatistics();
            }

            evalStream.Reset();

            auto t1 = std::chrono::steady_clock::now();
            evaluator.Evaluate(rnn, evalStream, inputPorts, outputPorts, _nEvalFramePerEpoch, evalStepSize);
            auto t2 = std::chrono::steady_clock::now();

            std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
            std::cout << "(" << time_span.count() << " sec)" << std::endl;
        }

        std::cout << "Total trained frames: " << totalTrainedFrame << std::endl;
        std::cout << "Total discarded frames: " << totalDiscardedFrame << std::endl;

        double curLoss = lambdaLoss(outputProbe);

        std::cout << "Loss: " << curLoss;


        if(isnan(curLoss))
        {
            curLoss = (double)1.0 / (double)+0.0;
        }

        if(totalTrainedFrame == _nTrainFramePerEpoch || curLoss < bestLoss)
        {
            std::cout << "  (best)";

            bestLoss = curLoss;
            totalTrainedFrameAtBest = totalTrainedFrame;
            rnn.SaveState(bestPath);
        }

        std::cout << std::endl;

        lambdaPostEval(outputProbe);

        if(totalTrainedFrame > _nTrainFramePerEpoch && prevLoss < curLoss)
        {
            std::cout << "----------------------------------------------------------------------" << std::endl;

            retryCount++;
            if(retryCount > maxRetryCount)
            {
                retryCount = 0;

                learningRate *= learningRateDecayRate;
                optimizer.SetLearningRate(learningRate);
                //optimizer.SetMetaLearningRate(learningRate);

                if(learningRate < minLearningRate) break;

                rnn.LoadState(pivotPath);
                rnn.SaveState(prevPath);

                std::cout << "Discard recently trained " << totalTrainedFrame - totalTrainedFrameAtPivot << " frames" << std::endl;
                std::cout << "New learning rate: " << learningRate << std::endl;

                /* Init accumulator variables */
                rnn.InitNesterov();

                if(adadelta == true)
                {
                    rnn.InitAdadelta(rmsDecayRate, false);
                }

                totalDiscardedFrame += totalTrainedFrame - totalTrainedFrameAtPivot;
                totalTrainedFrame = totalTrainedFrameAtPivot;
                prevLoss = pivotLoss;
            }
            else
            {
                std::cout << "Retry count: " << retryCount << " / " << maxRetryCount << std::endl;
            }
        }
        else
        {
            retryCount = 0;

            pivotLoss = prevLoss;
            prevLoss = curLoss;
            pivotPath.swap(prevPath);
            rnn.SaveState(prevPath);

            totalTrainedFrameAtPivot = totalTrainedFrame - _nTrainFramePerEpoch;
        }
    }

    rnn.LoadState(bestPath);

    totalDiscardedFrame += totalTrainedFrame - totalTrainedFrameAtBest;
    totalTrainedFrame = totalTrainedFrameAtBest;

    std::cout << "----------------------------------------------------------------------" << std::endl;
    std::cout << std::endl;

    std::cout << "Done." << std::endl;
    std::cout << "Total trained frames: " << totalTrainedFrame << std::endl;
    std::cout << "Total discarded frames: " << totalDiscardedFrame << std::endl;

}

}

