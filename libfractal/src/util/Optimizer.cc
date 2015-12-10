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


#include "Optimizer.h"

#include <thread>
#include <vector>


#ifdef FRACTAL_PIPELINE
#define PIPELINE_WIDTH 256
#endif /* FRACTAL_PIPELINE */


#define LOC 1


namespace fractal
{


Optimizer::Optimizer()
{
    learningRate = (FLOAT) 1e-5;
    momentum = (FLOAT) 0.9;

    weightNoise = (FLOAT) 0.0;
    adadelta = false;

    rmsprop = false;

    initFrameSkip = 0;
}


Optimizer::~Optimizer()
{
}


void Optimizer::Backprop(Rnn &rnn, Stream &stream, const PortMapList &inputPorts, const PortMapList &outputPorts,
        const unsigned long numFrame, const unsigned long windowSize, const unsigned long stepSize)
{
    verify(numFrame > 0 && stepSize > 0);

    BackpropArgs args;
    Engine *engine;
    unsigned long i;

    PortMapList::const_iterator portIter, portIter_end;

    args.nInput = inputPorts.size();
    args.nOutput = outputPorts.size();

    std::vector<InputProbe *> inputProbe(args.nInput);
    std::vector<TrainableProbe *> outputProbe(args.nOutput);

    std::vector<MultiTypeMatrix> input(args.nInput);
    std::vector<MultiTypeMatrix> inputBuf(args.nInput);
    std::vector<MultiTypeMatrix> target(args.nOutput);
    std::vector<MultiTypeMatrix> targetBuf(args.nOutput);
    std::vector<unsigned long> inputChannel(args.nInput);
    std::vector<unsigned long> outputChannel(args.nOutput);

    engine = rnn.GetEngine();
    verify(engine != NULL);

    args.rnn = &rnn;
    args.stream = &stream;
    args.numFrame = numFrame;
    args.nStream = stream.GetNumStream();
    args.batchSize = windowSize * args.nStream;
    args.frameStep = stepSize * args.nStream;
    args.inputProbe = inputProbe.data();
    args.outputProbe = outputProbe.data();
    args.inputChannel = inputChannel.data();
    args.outputChannel = outputChannel.data();
    args.input = input.data();
    args.inputBuf = inputBuf.data();
    args.target = target.data();
    args.targetBuf = targetBuf.data();

    //verify(numFrame % nStream == 0);
    verify(args.numFrame % (args.nStream * stepSize) == 0);

    /* Link probes and resize sequence buffers */
    portIter_end = inputPorts.end();
    for(portIter = inputPorts.begin(), i = 0; portIter != portIter_end; portIter++, i++)
    {
        ChannelInfo channelInfo;


        inputProbe[i] = dynamic_cast<InputProbe *>(std::get<0>(*portIter));
        verify(inputProbe[i] != NULL);

        inputChannel[i] = std::get<1>(*portIter);

        channelInfo = stream.GetChannelInfo(inputChannel[i]);

        switch(channelInfo.dataType)
        {
            case ChannelInfo::DATATYPE_VECTOR:
                verify(inputProbe[i]->GetLayerSize() == channelInfo.frameDim);
                verify(channelInfo.frameDim == channelInfo.frameSize);

                input[i].SetDataType(MultiTypeMatrix::DATATYPE_FLOAT);
                inputBuf[i].SetDataType(MultiTypeMatrix::DATATYPE_FLOAT);
            break;

            case ChannelInfo::DATATYPE_INDEX:
                verify(inputProbe[i]->GetLayerSize() == channelInfo.frameDim);
                verify(channelInfo.frameSize == 1);

                input[i].SetDataType(MultiTypeMatrix::DATATYPE_INT);
                inputBuf[i].SetDataType(MultiTypeMatrix::DATATYPE_INT);
            break;

            default:
                verify(false);
        }


        input[i].Resize(channelInfo.frameSize, args.frameStep);
        input[i].SetEngine(engine);
        inputBuf[i].Resize(channelInfo.frameSize, args.frameStep);
        inputBuf[i].SetEngine(engine);
    }

    portIter_end = outputPorts.end();
    for(portIter = outputPorts.begin(), i = 0; portIter != portIter_end; portIter++, i++)
    {
        ChannelInfo channelInfo;


        outputProbe[i] = dynamic_cast<TrainableProbe *>(std::get<0>(*portIter));
        verify(outputProbe[i] != NULL);


        outputProbe[i]->InitTraining(args.nStream, args.batchSize / args.nStream);

        outputChannel[i] = std::get<1>(*portIter);

        channelInfo = stream.GetChannelInfo(outputChannel[i]);

        switch(channelInfo.dataType)
        {
            case ChannelInfo::DATATYPE_VECTOR:
                verify(outputProbe[i]->GetLayerSize() == channelInfo.frameDim);
                verify(channelInfo.frameDim == channelInfo.frameSize);

                target[i].SetDataType(MultiTypeMatrix::DATATYPE_FLOAT);
                targetBuf[i].SetDataType(MultiTypeMatrix::DATATYPE_FLOAT);
            break;

            case ChannelInfo::DATATYPE_INDEX:
                verify(outputProbe[i]->GetLayerSize() == channelInfo.frameDim);
                verify(channelInfo.frameSize == 1);

            case ChannelInfo::DATATYPE_SEQ:
                target[i].SetDataType(MultiTypeMatrix::DATATYPE_INT);
                targetBuf[i].SetDataType(MultiTypeMatrix::DATATYPE_INT);
            break;

            default:
                verify(false);
        }

        target[i].Resize(channelInfo.frameSize, args.frameStep);
        target[i].SetEngine(engine);
        targetBuf[i].Resize(channelInfo.frameSize, args.frameStep);
        targetBuf[i].SetEngine(engine);
    }


    /* Initialize the RNN */
    rnn.SetBatchSize(args.nStream, args.batchSize / args.nStream);
    rnn.InitForward(0, args.batchSize / args.nStream - 1);
    rnn.EnableDropout(true);


    /* Main loop */
    engine->StreamCreate(pStreamDataTransferToBuf, LOC);
    engine->StreamCreate(pStreamDataTransferToRnn, LOC);
    engine->EventCreate(pEventDataTransferToBuf, LOC);
    engine->EventCreate(pEventDataTransferToRnn, LOC);

    pipe[0].Init();
    pipe[1].Init();
    pipe[2].Init();
    pipe[3].Init();

    std::thread thdPipe0(BackpropPipe0, this, std::ref(args));
    std::thread thdPipe1(BackpropPipe1, this, std::ref(args));
    std::thread thdPipe2(BackpropPipe2, this, std::ref(args));
    std::thread thdPipe3(BackpropPipe3, this, std::ref(args));

    pipe[0].SendSignal();
    pipe[1].SendSignal();
    pipe[2].SendSignal();

    thdPipe0.join();
    thdPipe1.join();
    thdPipe2.join();
    thdPipe3.join();

    engine->StreamDestroy(pStreamDataTransferToBuf);
    engine->StreamDestroy(pStreamDataTransferToRnn);
    engine->EventDestroy(pEventDataTransferToBuf);
    engine->EventDestroy(pEventDataTransferToRnn);

    rnn.ProcessWeights((FLOAT) 0);
    rnn.EnableDropout(false);
    rnn.Synchronize();
}


void Optimizer::BackpropPipe0(Optimizer *optimizer, BackpropArgs &args)
{
    Engine *engine = args.rnn->GetEngine();

    for(unsigned long frameIdx = 0; frameIdx < args.numFrame; frameIdx += args.frameStep)
    {
        unsigned long batchFrom = frameIdx % args.batchSize;
        unsigned long batchTo = batchFrom + std::min(args.numFrame - frameIdx, args.frameStep) - 1;
        unsigned long nForwardFrame = batchTo - batchFrom + 1;

        optimizer->pipe[0].Wait(1);


        /* Wait until the memory transfer to the engine finishes */

        engine->StreamSynchronize(optimizer->pStreamDataTransferToBuf);


        /* Generate sequences from the streams */

        for(unsigned long streamIdx = 0; streamIdx < args.nStream; streamIdx++)
        {
            for(unsigned long i = 0; i < nForwardFrame / args.nStream; i++)
            {
                for(unsigned long j = 0; j < args.nInput; j++)
                {
                    void *ptr = NULL;

                    switch(args.input[j].GetDataType())
                    {
                        case MultiTypeMatrix::DATATYPE_FLOAT:
                            {
                                Matrix<FLOAT> *mat = reinterpret_cast<Matrix<FLOAT> *>(args.input[j].GetMatrix());
                                ptr = mat->GetHostData() + (i * args.nStream + streamIdx) * mat->GetNumRows();
                            }
                            break;

                        case MultiTypeMatrix::DATATYPE_INT:
                            {
                                Matrix<INT> *mat = reinterpret_cast<Matrix<INT> *>(args.input[j].GetMatrix());
                                ptr = mat->GetHostData() + (i * args.nStream + streamIdx) * mat->GetNumRows();
                            }
                            break;

                        default:
                            verify(false);
                    }

                    args.stream->GenerateFrame(streamIdx, args.inputChannel[j], ptr);
                }

                for(unsigned long j = 0; j < args.nOutput; j++)
                {
                    void *ptr = NULL;

                    switch(args.target[j].GetDataType())
                    {
                        case MultiTypeMatrix::DATATYPE_FLOAT:
                            {
                                Matrix<FLOAT> *mat = reinterpret_cast<Matrix<FLOAT> *>(args.target[j].GetMatrix());
                                ptr = mat->GetHostData() + (i * args.nStream + streamIdx) * mat->GetNumRows();
                            }
                            break;

                        case MultiTypeMatrix::DATATYPE_INT:
                            {
                                Matrix<INT> *mat = reinterpret_cast<Matrix<INT> *>(args.target[j].GetMatrix());
                                ptr = mat->GetHostData() + (i * args.nStream + streamIdx) * mat->GetNumRows();
                            }
                            break;

                        default:
                            verify(false);
                    }

                    args.stream->GenerateFrame(streamIdx, args.outputChannel[j], ptr);
                }

                args.stream->Next(streamIdx);
            }
        }

        /* Push the matrices */

        for(unsigned long i = 0; i < args.nInput; i++)
        {
            switch(args.input[i].GetDataType())
            {
                case MultiTypeMatrix::DATATYPE_FLOAT:
                    {
                        Matrix<FLOAT> *matrix = reinterpret_cast<Matrix<FLOAT> *>(args.input[i].GetMatrix());
                        matrix->HostPush();
                    }
                    break;

                case MultiTypeMatrix::DATATYPE_INT:
                    {
                        Matrix<INT> *matrix = reinterpret_cast<Matrix<INT> *>(args.input[i].GetMatrix());
                        matrix->HostPush();
                    }
                    break;

                default:
                    verify(false);
            }
        }

        for(unsigned long i = 0; i < args.nOutput; i++)
        {
            switch(args.target[i].GetDataType())
            {
                case MultiTypeMatrix::DATATYPE_FLOAT:
                    {
                        Matrix<FLOAT> *matrix = reinterpret_cast<Matrix<FLOAT> *>(args.target[i].GetMatrix());
                        matrix->HostPush();
                    }
                    break;

                case MultiTypeMatrix::DATATYPE_INT:
                    {
                        Matrix<INT> *matrix = reinterpret_cast<Matrix<INT> *>(args.target[i].GetMatrix());
                        matrix->HostPush();
                    }
                    break;

                default:
                    verify(false);
            }
        }


        optimizer->pipe[1].SendSignal();
    }
}


void Optimizer::BackpropPipe1(Optimizer *optimizer, BackpropArgs &args)
{
    Engine *engine = args.rnn->GetEngine();

    for(unsigned long frameIdx = 0; frameIdx < args.numFrame; frameIdx += args.frameStep)
    {
        optimizer->pipe[1].Wait(2);

        engine->StreamWaitEvent(optimizer->pStreamDataTransferToBuf, optimizer->pEventDataTransferToRnn);

        /* Copy the sequences to the buffers */

        for(unsigned long i = 0; i < args.nInput; i++)
        {
            args.inputBuf[i].Swap(args.input[i]);

            switch(args.inputBuf[i].GetDataType())
            {
                case MultiTypeMatrix::DATATYPE_FLOAT:
                    {
                        Matrix<FLOAT> *matrix = reinterpret_cast<Matrix<FLOAT> *>(args.inputBuf[i].GetMatrix());
                        engine->MemPull(matrix->GetMem(), LOC, optimizer->pStreamDataTransferToBuf);
                    }
                    break;

                case MultiTypeMatrix::DATATYPE_INT:
                    {
                        Matrix<INT> *matrix = reinterpret_cast<Matrix<INT> *>(args.inputBuf[i].GetMatrix());
                        engine->MemPull(matrix->GetMem(), LOC, optimizer->pStreamDataTransferToBuf);
                    }
                    break;

                default:
                    verify(false);
            }

            //args.inputBuf[i].Pull(LOC, optimizer->pStreamDataTransferToBuf);
        }

        for(unsigned long i = 0; i < args.nOutput; i++)
        {
            args.targetBuf[i].Swap(args.target[i]);

            switch(args.targetBuf[i].GetDataType())
            {
                case MultiTypeMatrix::DATATYPE_FLOAT:
                    {
                        Matrix<FLOAT> *matrix = reinterpret_cast<Matrix<FLOAT> *>(args.targetBuf[i].GetMatrix());
                        engine->MemPull(matrix->GetMem(), LOC, optimizer->pStreamDataTransferToBuf);
                    }
                    break;

                case MultiTypeMatrix::DATATYPE_INT:
                    {
                        Matrix<INT> *matrix = reinterpret_cast<Matrix<INT> *>(args.targetBuf[i].GetMatrix());
                        engine->MemPull(matrix->GetMem(), LOC, optimizer->pStreamDataTransferToBuf);
                    }
                    break;

                default:
                    verify(false);
            }

            //args.targetBuf[i].Pull(LOC, optimizer->pStreamDataTransferToBuf);
        }

        engine->EventRecord(optimizer->pEventDataTransferToBuf, optimizer->pStreamDataTransferToBuf);

        optimizer->pipe[0].SendSignal();
        optimizer->pipe[2].SendSignal();
    }
}


void Optimizer::BackpropPipe2(Optimizer *optimizer, BackpropArgs &args)
{
    Engine *engine = args.rnn->GetEngine();

    for(unsigned long frameIdx = 0; frameIdx < args.numFrame; frameIdx += args.frameStep)
    {
        unsigned long batchFrom = frameIdx % args.batchSize;
        unsigned long batchTo = batchFrom + std::min(args.numFrame - frameIdx, args.frameStep) - 1;

        optimizer->pipe[2].Wait(2);

        //args.rnn->Synchronize();
        engine->StreamWaitEvent(optimizer->pStreamDataTransferToRnn, optimizer->pEventDataTransferToBuf);
        args.rnn->StreamWait(optimizer->pStreamDataTransferToRnn);


        /* Copy the sequences to the RNN */

        for(unsigned long i = 0; i < args.nInput; i++)
        {
            ChannelInfo channelInfo = args.stream->GetChannelInfo(args.inputChannel[i]);
            Matrix<FLOAT> stateSub(args.inputProbe[i]->GetState(), batchFrom, batchTo);

            switch(channelInfo.dataType)
            {
                case ChannelInfo::DATATYPE_VECTOR:
                    {
                        Matrix<FLOAT> *matrix = reinterpret_cast<Matrix<FLOAT> *>(args.inputBuf[i].GetMatrix());
                        Matrix<FLOAT> inputSub(*matrix, 0, batchTo - batchFrom);

                        engine->MatCopy(inputSub, stateSub, optimizer->pStreamDataTransferToRnn);
                    }
                    break;

                case ChannelInfo::DATATYPE_INDEX:
                    {
                        Matrix<INT> *matrix = reinterpret_cast<Matrix<INT> *>(args.inputBuf[i].GetMatrix());
                        Matrix<INT> inputSub(*matrix, 0, batchTo - batchFrom);

                        engine->OneHotEncode(inputSub, stateSub, optimizer->pStreamDataTransferToRnn);
                    }
                    break;

                default:
                    verify(false);
            }


            engine->EventRecord(optimizer->pEventDataTransferToRnn, optimizer->pStreamDataTransferToRnn);
            engine->StreamWaitEvent(args.inputProbe[i]->GetPStream(), optimizer->pEventDataTransferToRnn);

            args.inputProbe[i]->EventRecord();
        }

        if(frameIdx + batchTo - batchFrom >= optimizer->initFrameSkip * args.nStream)
        {
            unsigned long batchFromOffset = std::max((long) 0, (long) (optimizer->initFrameSkip * args.nStream) - (long) frameIdx);
            for(unsigned long i = 0; i < args.nOutput; i++)
            {

                MultiTypeMatrix targetSub(args.targetBuf[i], batchFromOffset, batchTo - batchFrom);

                engine->StreamWaitEvent(args.outputProbe[i]->GetPStream(), optimizer->pEventDataTransferToBuf);
                args.outputProbe[i]->InitErr(0, args.batchSize / args.nStream - 1);
                args.outputProbe[i]->SetTarget(targetSub, (batchFrom + batchFromOffset) / args.nStream, batchTo / args.nStream);

                args.outputProbe[i]->StreamWaitEvent(optimizer->pStreamDataTransferToRnn);
            }
        }


        engine->EventRecord(optimizer->pEventDataTransferToRnn, optimizer->pStreamDataTransferToRnn);

        optimizer->pipe[1].SendSignal();
        optimizer->pipe[3].SendSignal();
    }
}


void Optimizer::BackpropPipe3(Optimizer *optimizer, BackpropArgs &args)
{
    for(unsigned long frameIdx = 0; frameIdx < args.numFrame; frameIdx += args.frameStep)
    {
        optimizer->pipe[3].Wait(1);

        unsigned long batchFrom = frameIdx % args.batchSize;
        unsigned long batchTo = batchFrom + std::min(args.numFrame - frameIdx, args.frameStep) - 1;

        /* Process weights */

        args.rnn->ProcessWeights(optimizer->weightNoise);


        /* Generate dropout mask */

        args.rnn->GenerateDropoutMask(batchFrom / args.nStream, batchTo / args.nStream);


        /* Forward pass */
#ifdef FRACTAL_SEQUENTIAL
        for(long i = (long) batchFrom; i <= (long) batchTo; i += args.nStream)
            args.rnn->Forward(i / args.nStream, (i + args.nStream - 1) / args.nStream);
#else
        //if(frameIdx + args.frameStep > args.batchSize && batchFrom + args.frameStep < args.batchSize)
        //    rnn.Forward((batchFrom + args.frameStep) / args.nStream, (batchSize - 1) / args.nStream);
        //rnn.Forward(0, batchTo / args.nStream);
#ifdef FRACTAL_PIPELINE
        long pStepSize = (PIPELINE_WIDTH + args.nStream - 1) / args.nStream * args.nStream;

        for(long i = (long) batchFrom; i <= (long) batchTo; i += pStepSize)
            args.rnn->Forward(i / args.nStream, std::min(i + pStepSize - 1, (long) batchTo) / args.nStream);
#else
        args.rnn->Forward(batchFrom / args.nStream, batchTo / args.nStream);
#endif /* FRACTAL_PIPELINE */
#endif /* FRACTAL_SEQUENTIAL */


        if(frameIdx + batchTo - batchFrom >= optimizer->initFrameSkip * args.nStream)
        {
            unsigned long batchFromOffset = std::max((long) 0, (long) (optimizer->initFrameSkip * args.nStream) - (long) frameIdx);

            /* Compute output errors */
            for(unsigned long i = 0; i < args.nOutput; i++)
            {
                args.outputProbe[i]->Wait();
                args.outputProbe[i]->ComputeErr((batchFrom + batchFromOffset) / args.nStream, batchTo / args.nStream);
                args.outputProbe[i]->EventRecord();
            }


            /* Compute derivatives of the activation functions */
            args.rnn->CalcActDeriv(0, std::min(frameIdx + args.frameStep, args.batchSize) / args.nStream - 1);


            /* Backward pass */
            batchTo = batchFrom + args.frameStep - 1;
            args.rnn->InitBackward(0, args.batchSize / args.nStream - 1);

#ifdef FRACTAL_SEQUENTIAL
            for(long i = (long) batchTo; i >= 0; i -= args.nStream)
                args.rnn->Backward((i - args.nStream + 1) / args.nStream, i / args.nStream);
            for(long i = (long) args.batchSize - 1; i > (long) batchTo; i -= args.nStream)
                args.rnn->Backward((i - args.nStream + 1) / args.nStream, i / args.nStream);
#else
#ifdef FRACTAL_PIPELINE
            for(long i = (long) batchTo; i >= 0; i -= pStepSize)
                args.rnn->Backward(std::max(i - pStepSize + 1, (long) 0) / args.nStream, i / args.nStream);
            for(long i = (long) args.batchSize - 1; i > (long) batchTo; i -= pStepSize)
                args.rnn->Backward(std::max(i - pStepSize, (long) batchTo) / args.nStream + 1, i / args.nStream);
#else
            args.rnn->Backward(0, batchTo / args.nStream);
            if(frameIdx + args.frameStep > args.batchSize && batchFrom + args.frameStep < args.batchSize)
                args.rnn->Backward((batchFrom + args.frameStep) / args.nStream, args.batchSize / args.nStream - 1);
#endif /* FRACTAL_PIPELINE */
#endif /* FRACTAL_SEQUENTIAL */


            /* Update weights */
            args.rnn->UpdateWeights(0, std::min(frameIdx + args.frameStep, args.batchSize) / args.nStream - 1,
                    optimizer->learningRate, optimizer->momentum, optimizer->adadelta, optimizer->rmsprop);
        }

        optimizer->pipe[2].SendSignal();
    }
}


}

