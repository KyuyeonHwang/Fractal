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


#include <cstdio>
#include <string>
#include <sstream>
#include <iostream>

#include <fractal/fractal.h>

#include "TextDataSet.h"



using namespace fractal;


void printUsage(const char *argv0)
{
    std::cout << "Usage: " << argv0 << " [ -t <corpus path> | -g <starting sentence> ] <workspace path>" << std::endl;
}


int main(int argc, char *argv[])
{
    Rnn rnn;
    Engine engine;
    InitWeightParam initWeightParam;
    unsigned long long randomSeed;
    bool train = false;

    unsigned long inputChannel, outputChannel, inputDim, outputDim;

    if(argc != 4)
    {
        printUsage(argv[0]);
        exit(1);
    }


    if(std::string(argv[1]) == "-t")
    {
        train = true;
    }
    else if(std::string(argv[1]) == "-g")
    {
        train = false;
    }
    else
    {
        printUsage(argv[0]);
        exit(1);
    }

    std::string workspacePath = argv[3];

    initWeightParam.stdev = 1e-2;
    initWeightParam.isValid = true;

    randomSeed = 0;
    std::cout << "Random seed: " << randomSeed << std::endl;

    std::string corpusFile = argv[2];

    TextDataSet textTrainData, textDevData, textTestData;
    DataStream textTrainDataStream, textDevDataStream, textTestDataStream;

    if(train == true)
    {
        textTrainData.ReadTextData(corpusFile);

        textTrainData.Split(textDevData, 0.05);
        textTrainData.Split(textTestData, 0.05);


        textTrainDataStream.LinkDataSet(&textTrainData);
        textDevDataStream.LinkDataSet(&textDevData);
        textTestDataStream.LinkDataSet(&textTestData);
    }

    inputChannel = TextDataSet::CHANNEL_TEXT1;
    outputChannel = TextDataSet::CHANNEL_TEXT2;

    inputDim = textTrainData.GetChannelInfo(inputChannel).frameDim;
    outputDim = textTrainData.GetChannelInfo(outputChannel).frameDim;

    printf("Train: %ld sequences\n", textTrainData.GetNumSeq());
    printf("  Dev: %ld sequences\n", textDevData.GetNumSeq());
    printf(" Test: %ld sequences\n", textTestData.GetNumSeq());

    printf("\n");

    printf(" Input dim: %ld\n", inputDim);
    printf("Output dim: %ld\n", outputDim);

    printf("\n");

    /* Setting random seeds */
    engine.SetRandomSeed(randomSeed);
    textTrainDataStream.SetRandomSeed(randomSeed);
    textDevDataStream.SetRandomSeed(randomSeed);
    textTestDataStream.SetRandomSeed(randomSeed);

    rnn.SetEngine(&engine);


    /* Set compute locations */
    rnn.SetComputeLocs({1}); /* Single GPU */
    //rnn.SetComputeLocs({1, 2, 3, 4}); /* Four GPUs */
    //rnn.SetComputeLocs({1, 3}); /* Just first and third of them */


    long N = 256;

    rnn.AddLayer("BIAS", ACT_BIAS, AGG_DONTCARE, 1);
    rnn.AddLayer("INPUT", ACT_LINEAR, AGG_DONTCARE, inputDim);
    rnn.AddLayer("OUTPUT", ACT_SOFTMAX, AGG_SUM, outputDim);

    basicLayers::AddFastLstmLayer(rnn, "LSTM[0]", "BIAS", 1, N, true, initWeightParam);
    basicLayers::AddFastLstmLayer(rnn, "LSTM[1]", "BIAS", 1, N, true, initWeightParam);

    rnn.AddConnection("INPUT", "LSTM[0].INPUT", initWeightParam);

    rnn.AddConnection("LSTM[0].OUTPUT", "LSTM[1].INPUT", initWeightParam);

    rnn.AddConnection("LSTM[1].OUTPUT", "OUTPUT", initWeightParam);
    rnn.AddConnection("BIAS", "OUTPUT", initWeightParam);



    rnn.Ready();

    printf("Number of weights: %ld\n\n", rnn.GetNumWeights());

    if(train == true)
    {
        AutoOptimizer autoOptimizer;


        textTrainDataStream.SetDelay(inputChannel, 1);
        textDevDataStream.SetDelay(inputChannel, 1);
        textTestDataStream.SetDelay(inputChannel, 1);

        textTrainDataStream.SetNumStream(64);
        textDevDataStream.SetNumStream(128);

        /* Set ports */
        InputProbe inputProbe;
        MultiClassifProbe outputProbe;

        rnn.LinkProbe(inputProbe, "INPUT");
        rnn.LinkProbe(outputProbe, "OUTPUT");

        PortMapList inputPorts, outputPorts;

        inputPorts.push_back(PortMap(&inputProbe, inputChannel));
        outputPorts.push_back(PortMap(&outputProbe, outputChannel));

        /* Training */
        {
            autoOptimizer.SetWorkspacePath(workspacePath);
            autoOptimizer.SetInitLearningRate(1e-5);
            autoOptimizer.SetMinLearningRate(1e-6);
            autoOptimizer.SetLearningRateDecayRate(0.1);
            autoOptimizer.SetMaxRetryCount(3);
            autoOptimizer.SetMomentum(0.9);
            autoOptimizer.SetWeightNoise(0.0);
            autoOptimizer.SetAdadelta(true);
            //autoOptimizer.SetRmsprop(true);
            autoOptimizer.SetRmsDecayRate(0.99);

            autoOptimizer.Optimize(rnn,
                    textTrainDataStream, textDevDataStream,
                    inputPorts, outputPorts,
                    2 * 1024 * 1024, 1 * 1024 * 1024, 256, 128);
        }

        /* Evaluate the best network */

        textTrainDataStream.SetNumStream(128);
        textDevDataStream.SetNumStream(128);
        textTestDataStream.SetNumStream(128);

        textTrainDataStream.Reset();
        textDevDataStream.Reset();
        textTestDataStream.Reset();

        std::cout << "Best network:" << std::endl;

        Evaluator evaluator;

        outputProbe.ResetStatistics();
        evaluator.Evaluate(rnn, textTrainDataStream, inputPorts, outputPorts, 4 * 1024 * 1024, 32);
        printf("Train :  MSE: %f  ACE: %f  FER: %f  BPC: %f\n", outputProbe.GetMeanSquaredError(),
                outputProbe.GetAverageCrossEntropy(),
                outputProbe.GetFrameErrorRate(),
                outputProbe.GetAverageCrossEntropy() / log(2.0));

        outputProbe.ResetStatistics();
        evaluator.Evaluate(rnn, textDevDataStream, inputPorts, outputPorts, 4 * 1024 * 1024, 32);
        printf("  Dev :  MSE: %f  ACE: %f  FER: %f  BPC: %f\n", outputProbe.GetMeanSquaredError(),
                outputProbe.GetAverageCrossEntropy(),
                outputProbe.GetFrameErrorRate(),
                outputProbe.GetAverageCrossEntropy() / log(2.0));

        outputProbe.ResetStatistics();
        evaluator.Evaluate(rnn, textTestDataStream, inputPorts, outputPorts, 4 * 1024 * 1024, 32);
        printf(" Test :  MSE: %f  ACE: %f  FER: %f  BPC: %f\n", outputProbe.GetMeanSquaredError(),
                outputProbe.GetAverageCrossEntropy(),
                outputProbe.GetFrameErrorRate(),
                outputProbe.GetAverageCrossEntropy() / log(2.0));
    }
    else
    {
        rnn.LoadState(workspacePath + "/net/best/");

        /* ================================================================ */
        /*                  Get data from the RNN output                    */
        /* ================================================================ */

        unsigned long nUnroll = 32, nStream = 1;
        Matrix<FLOAT> matInput(inputDim, nStream);
        Matrix<FLOAT> matOutput(outputDim, nStream);
        InputProbe inputProbe;
        OutputProbe outputProbe;
        const std::string startString = argv[2];
        unsigned long szStartString;
        unsigned long maxOutputIdx = 0;



        /* Set engine */
        matInput.SetEngine(&engine);
        matOutput.SetEngine(&engine);

        std::cout << startString;

        /* Link I/O probes */
        rnn.LinkProbe(inputProbe, "INPUT");
        rnn.LinkProbe(outputProbe, "OUTPUT");


        /* Unroll the network nUnroll times and replicate it nStream times */
        rnn.SetBatchSize(nStream, nUnroll);


        /* Initialize the forward activations */
        rnn.InitForward(0, nUnroll - 1);

        szStartString = startString.size();

        for(unsigned long j = 0; j < 4000; j++)
        {
            if(j < szStartString)
            {
                unsigned long idx = static_cast<unsigned long>(startString.c_str()[j]);

                for(unsigned long i = 0; i < inputDim; i++)
                {
                    matInput.GetHostData()[i] = (FLOAT) (idx == i);
                }
            }
            else
            {
                for(unsigned long i = 0; i < inputDim; i++)
                {
                    matInput.GetHostData()[i] = (FLOAT) (maxOutputIdx == i);
                }
            }

            /* Copy the input sequence to the RNN (asynchronous) */
            Matrix<FLOAT> stateSub(inputProbe.GetState(), j % nUnroll, j % nUnroll);

            matInput.HostPush();
            engine.MatCopy(matInput, stateSub, inputProbe.GetPStream());
            inputProbe.EventRecord();


            /* Forward computation */
            /* Automatically scheduled to be executed after the copy event through the input probe */
            rnn.Forward(j % nUnroll, j % nUnroll);


            /* Copy the output sequence from the RNN to the GPU memory of matOutput */
            /* Automatically scheduled to be executed after finishing the forward activation */
            Matrix<FLOAT> actSub(outputProbe.GetActivation(), j % nUnroll, j % nUnroll);

            outputProbe.Wait();
            engine.MatCopy(actSub, matOutput, outputProbe.GetPStream());


            /* Copy the output matrix from the device (GPU) to the host (CPU) */
            matOutput.HostPull(outputProbe.GetPStream());
            outputProbe.EventRecord();


            /* Since the above operations are asynchronous, synchronization is required */
            outputProbe.EventSynchronize();

            if(j >= szStartString - 1)
            {
#if 1
                /* Find the maximum output */
                FLOAT maxVal = (FLOAT) 0;
                maxOutputIdx = 0;

                for(unsigned long i = 0; i < outputDim; i++)
                {
                    if(matOutput.GetHostData()[i] > maxVal)
                    {
                        maxVal = matOutput.GetHostData()[i];
                        maxOutputIdx = i;
                    }
                }
#endif
#if 1
                /* Random pick */
                FLOAT randVal = (FLOAT) rand() / RAND_MAX;
                for(unsigned long i = 0; i < outputDim; i++)
                {
                    FLOAT prob = matOutput.GetHostData()[i];

                    if(prob >= randVal)
                    {
                        maxOutputIdx = i;
                        break;
                    }
                    randVal -= prob;
                }
#endif
                const char outChar = static_cast<char>(maxOutputIdx);
                std::cout << outChar;
                if(outChar == '\n') std::cout << std::endl;
                std::cout.flush();
            }
        }

        std::cout << std::endl;


        /* Unlink the probes */
        inputProbe.UnlinkLayer();
        outputProbe.UnlinkLayer();
    }


    rnn.SetEngine(NULL);

    return 0;
}

