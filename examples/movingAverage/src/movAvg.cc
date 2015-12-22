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


#include <iostream>
#include <fractal/fractal.h>

#include "MovAvgStream.h"


using namespace fractal;


int main(int argc, char *argv[])
{
    if(argc != 2)
    {
        std::cout << "usage: " << argv[0] << " <workspace path>" << std::endl;
        return 1;
    }

    Engine engine;
    Rnn rnn;
    MovAvgStream movAvgStreamTrain, movAvgStreamDev;


    std::string workspacePath = argv[1];

    unsigned long inputChannel = MovAvgStream::CHANNEL_INPUT;
    unsigned long outputChannel = MovAvgStream::CHANNEL_TARGET;

    unsigned long inputDim = movAvgStreamTrain.GetChannelInfo(inputChannel).frameDim;
    unsigned long outputDim = movAvgStreamTrain.GetChannelInfo(outputChannel).frameDim;

    std::cout << " Input dim: " << inputDim << std::endl;
    std::cout << "Output dim: " << outputDim << std::endl;


    /* Set random seeds */
    engine.SetRandomSeed(0);


    /* Set engine */
    rnn.SetEngine(&engine);


    /* Set compute locations */
    rnn.SetComputeLocs({1}); /* Single GPU */
    //rnn.SetComputeLocs({1, 2, 3, 4}); /* Four GPUs */
    //rnn.SetComputeLocs({1, 3}); /* Just first and third of them */


    /* Set moving average window size */
    unsigned long movAvgWindowSize = 10;
    movAvgStreamTrain.SetWindowSize(movAvgWindowSize);
    movAvgStreamDev.SetWindowSize(movAvgWindowSize);

    std::cout << "Compute the moving average of the recent " << movAvgWindowSize << " inputs." << std::endl;
    std::cout << std::endl;

#if 0
    /* Elman network */
    long hiddenSize = 256;

    rnn.AddLayer("BIAS", ACT_BIAS, AGG_DONTCARE, 1);
    rnn.AddLayer("INPUT", ACT_LINEAR, AGG_DONTCARE, inputDim);
    rnn.AddLayer("HIDDEN", ACT_TANH, AGG_SUM, hiddenSize);
    rnn.AddLayer("OUTPUT", ACT_LINEAR, AGG_SUM, outputDim);

    rnn.AddConnection("INPUT", "HIDDEN");
    rnn.AddConnection("HIDDEN", "HIDDEN", 1);
    rnn.AddConnection("BIAS", "HIDDEN");

    rnn.AddConnection("HIDDEN", "OUTPUT");
    rnn.AddConnection("BIAS", "OUTPUT");
#endif

#if 0
    /* LSTM network */
    long hiddenSize = 256;

    rnn.AddLayer("BIAS", ACT_BIAS, AGG_DONTCARE, 1);
    rnn.AddLayer("INPUT", ACT_LINEAR, AGG_DONTCARE, inputDim);
    rnn.AddLayer("OUTPUT", ACT_LINEAR, AGG_SUM, outputDim);

    basicLayers::AddLstmLayer(rnn, "LSTM[0]", "BIAS", 1, hiddenSize, true);

    rnn.AddConnection("INPUT", "LSTM[0].INPUT");
    rnn.AddConnection("INPUT", "LSTM[0].INPUT_GATE");
    rnn.AddConnection("INPUT", "LSTM[0].FORGET_GATE");
    rnn.AddConnection("INPUT", "LSTM[0].OUTPUT_GATE");

    rnn.AddConnection("LSTM[0].OUTPUT", "OUTPUT");
    rnn.AddConnection("BIAS", "OUTPUT");
#endif

#if 0
    /* Fast LSTM network */
    long hiddenSize = 256;

    rnn.AddLayer("BIAS", ACT_BIAS, AGG_DONTCARE, 1);
    rnn.AddLayer("INPUT", ACT_LINEAR, AGG_DONTCARE, inputDim);
    rnn.AddLayer("OUTPUT", ACT_LINEAR, AGG_SUM, outputDim);

    basicLayers::AddFastLstmLayer(rnn, "LSTM[0]", "BIAS", 1, hiddenSize, true);

    rnn.AddConnection("INPUT", "LSTM[0].INPUT");
    rnn.AddConnection("LSTM[0].OUTPUT", "OUTPUT");
    rnn.AddConnection("BIAS", "OUTPUT");
#endif

#if 0
    /* GRU network */
    long hiddenSize = 256;

    rnn.AddLayer("BIAS", ACT_BIAS, AGG_DONTCARE, 1);
    rnn.AddLayer("INPUT", ACT_LINEAR, AGG_DONTCARE, inputDim);
    rnn.AddLayer("OUTPUT", ACT_LINEAR, AGG_SUM, outputDim);

    basicLayers::AddGruLayer(rnn, "GRU[0]", "BIAS", 1, hiddenSize);

    rnn.AddConnection("INPUT", "GRU[0].INPUT");
    rnn.AddConnection("INPUT", "GRU[0].RESET_GATE");
    rnn.AddConnection("INPUT", "GRU[0].UPDATE_GATE");

    rnn.AddConnection("GRU[0].OUTPUT", "OUTPUT");
    rnn.AddConnection("BIAS", "OUTPUT");
#endif

#if 1
    /* Fast GRU network */
    long hiddenSize = 256;

    rnn.AddLayer("BIAS", ACT_BIAS, AGG_DONTCARE, 1);
    rnn.AddLayer("INPUT", ACT_LINEAR, AGG_DONTCARE, inputDim);
    rnn.AddLayer("OUTPUT", ACT_LINEAR, AGG_SUM, outputDim);

    basicLayers::AddFastGruLayer(rnn, "GRU[0]", "BIAS", 1, hiddenSize);
    rnn.AddLayer("FC", ACT_RECTLINEAR, AGG_SUM, hiddenSize);

    rnn.AddConnection("INPUT", "GRU[0].INPUT");

    rnn.AddConnection("GRU[0].OUTPUT", "FC");
    rnn.AddConnection("BIAS", "FC");

    rnn.AddConnection("FC", "OUTPUT");
    rnn.AddConnection("BIAS", "OUTPUT");
#endif

    /* Initialize weights */
    rnn.InitWeights(1e-2);


    /* Set ports */
    InputProbe inputProbe;
    RegressProbe outputProbe;

    rnn.LinkProbe(inputProbe, "INPUT");
    rnn.LinkProbe(outputProbe, "OUTPUT");

    PortMapList inputPorts, outputPorts;

    inputPorts.push_back(PortMap(&inputProbe, inputChannel));
    outputPorts.push_back(PortMap(&outputProbe, outputChannel));


    /* Set the number of parallel streams */
    movAvgStreamTrain.SetNumStream(64);
    movAvgStreamDev.SetNumStream(64);


    /* Set optimizer */
    AutoOptimizer optimizer;

    optimizer.SetWorkspacePath(workspacePath);
    optimizer.SetInitLearningRate(1e-4);
    optimizer.SetMinLearningRate(1e-7);
    optimizer.SetMaxRetryCount(2);
    optimizer.SetMomentum(0.9);
    optimizer.SetLearningRateDecayRate(0.1);
    optimizer.SetWeightNoise(0.0);
    optimizer.SetAdadelta(true);
    //optimizer.SetRmsprop(true);
    optimizer.SetRmsDecayRate(0.99);


    /* Train the network */
    optimizer.Optimize(rnn,
            movAvgStreamTrain, movAvgStreamDev,
            inputPorts, outputPorts,
            128 * 1024, 64 * 1024, 64, 32);


    /* ================================================================ */
    /*                  Get data from the RNN output                    */
    /* ================================================================ */

    unsigned long nUnroll = 10, streamIdx = 0, nStream = 1;
    Matrix<FLOAT> matInput(inputDim, nStream * nUnroll);
    Matrix<FLOAT> matTarget(outputDim, nStream * nUnroll);
    Matrix<FLOAT> matOutput(outputDim, nStream * nUnroll);


    movAvgStreamDev.Reset();
    movAvgStreamDev.SetNumStream(nStream);


    /* Set engine */
    matInput.SetEngine(&engine);
    matTarget.SetEngine(&engine);
    matOutput.SetEngine(&engine);


    /* Unroll the network nUnroll times and replicate it nStream times */
    rnn.SetBatchSize(nStream, nUnroll);


    /* Initialize the forward activations */
    rnn.InitForward(0, nUnroll * nStream - 1);


    for(unsigned long j = 0; j < 4; j++)
    {
        /* Generate sequences from the dev stream */
        for(unsigned long i = 0; i < nUnroll; i++)
        {
            unsigned long frameSize;

            frameSize = movAvgStreamDev.GetChannelInfo(inputChannel).frameSize;
            movAvgStreamDev.GenerateFrame(streamIdx, inputChannel,
                    matInput.GetHostData() + (i * nStream + streamIdx) * frameSize);

            frameSize = movAvgStreamDev.GetChannelInfo(outputChannel).frameSize;
            movAvgStreamDev.GenerateFrame(streamIdx, outputChannel,
                    matTarget.GetHostData() + (i * nStream + streamIdx) * frameSize);

            movAvgStreamDev.Next(streamIdx);
        }


        /* Copy the input sequence to the RNN (asynchronous) */
        Matrix<FLOAT> stateSub(inputProbe.GetState(), 0, nUnroll * nStream - 1);

        matInput.HostPush();
        engine.MatCopy(matInput, stateSub, inputProbe.GetPStream());
        inputProbe.EventRecord();


        /* Forward computation */
        /* Automatically scheduled to be executed after the copy event through the input probe */
        rnn.Forward(0, nUnroll - 1);


        /* Copy the output sequence from the RNN to the GPU memory of matOutput */
        /* Automatically scheduled to be executed after finishing the forward activation */
        Matrix<FLOAT> actSub(outputProbe.GetActivation(), 0, nUnroll * nStream - 1);

        outputProbe.Wait();
        engine.MatCopy(actSub, matOutput, outputProbe.GetPStream());


        /* Copy the output matrix from the device (GPU) to the host (CPU) */
        matOutput.HostPull(outputProbe.GetPStream());
        outputProbe.EventRecord();


        /* Since the above operations are asynchronous, synchronization is required */
        outputProbe.EventSynchronize();


        /* Print out the result */
        std::cout << std::endl;
        std::cout << "Iteration " << j << std::endl;
        std::cout << "Input\tTarget\tOutput" << std::endl;

        for(unsigned long i = 0; i < nUnroll; i++)
        {
            std::cout << matInput.GetHostData()[(i * nStream + streamIdx) * inputDim + 0] << "\t"
                << matTarget.GetHostData()[(i * nStream + streamIdx) * inputDim + 0] << "\t"
                << matOutput.GetHostData()[(i * nStream + streamIdx) * inputDim + 0] << std::endl;
        }
    }


    /* Unlink the probes */
    inputProbe.UnlinkLayer();
    outputProbe.UnlinkLayer();


    /* Dealloc all memory */
    rnn.SetEngine(NULL);

    return 0;
}

