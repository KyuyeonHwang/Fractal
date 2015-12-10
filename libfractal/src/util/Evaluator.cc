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


#include "Evaluator.h"

#include <thread>


#ifdef FRACTAL_PIPELINE
#define PIPELINE_WIDTH 256
#endif /* FRACTAL_PIPELINE */

#define LOC 1


namespace fractal
{


Evaluator::Evaluator()
{
    initFrameSkip = 0;
}


void Evaluator::Evaluate(Rnn &rnn, Stream &stream, const PortMapList &inputPorts, const PortMapList &outputPorts,
        const unsigned long numFrame, const unsigned long stepSize)
{
    verify(numFrame > 0 && stepSize > 0);

    EvaluateArgs args;
    Engine *engine;
    unsigned long i;

    PortMapList::const_iterator portIter, portIter_end;

    args.nInput = inputPorts.size();
    args.nOutput = outputPorts.size();

    std::vector<InputProbe *> inputProbe(args.nInput);
    std::vector<TrainableProbe *> outputProbe(args.nOutput);

    std::vector<MultiTypeMatrix> input(args.nInput);
    std::vector<MultiTypeMatrix> inputBuf(args.nInput);
    std::vector<Matrix<FLOAT>> output(args.nOutput);
    std::vector<Matrix<FLOAT>> outputBuf(args.nOutput);

    std::vector<MultiTypeMatrix> target(args.nOutput);
    std::vector<MultiTypeMatrix> targetPipe1(args.nOutput);
    std::vector<MultiTypeMatrix> targetPipe2(args.nOutput);
    std::vector<MultiTypeMatrix> targetPipe3(args.nOutput);
    std::vector<MultiTypeMatrix> targetPipe4(args.nOutput);
    std::vector<MultiTypeMatrix> targetPipe5(args.nOutput);

    std::vector<unsigned long> inputChannel(args.nInput);
    std::vector<unsigned long> outputChannel(args.nOutput);

    engine = rnn.GetEngine();
    verify(engine != NULL);

    args.rnn = &rnn;
    args.stream = &stream;
    args.numFrame = numFrame;
    args.nStream = stream.GetNumStream();
    args.frameStep = stepSize * args.nStream;
    args.inputProbe = inputProbe.data();
    args.outputProbe = outputProbe.data();
    args.inputChannel = inputChannel.data();
    args.outputChannel = outputChannel.data();
    args.input = input.data();
    args.inputBuf = inputBuf.data();
    args.output = output.data();
    args.outputBuf = outputBuf.data();
    args.target = target.data();
    args.targetPipe1 = targetPipe1.data();
    args.targetPipe2 = targetPipe2.data();
    args.targetPipe3 = targetPipe3.data();
    args.targetPipe4 = targetPipe4.data();
    args.targetPipe5 = targetPipe5.data();

    verify(args.numFrame % args.nStream == 0);
    //verify(args.numFrame % (args.nStream * stepSize) == 0);


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


        outputProbe[i]->InitEvaluation(args.nStream, args.frameStep / args.nStream);
        
        outputChannel[i] = std::get<1>(*portIter);

        channelInfo = stream.GetChannelInfo(outputChannel[i]);

        switch(channelInfo.dataType)
        {
            case ChannelInfo::DATATYPE_VECTOR:
                verify(outputProbe[i]->GetLayerSize() == channelInfo.frameDim);
                verify(channelInfo.frameDim == channelInfo.frameSize);

                target[i].SetDataType(MultiTypeMatrix::DATATYPE_FLOAT);
                targetPipe1[i].SetDataType(MultiTypeMatrix::DATATYPE_FLOAT);
                targetPipe2[i].SetDataType(MultiTypeMatrix::DATATYPE_FLOAT);
                targetPipe3[i].SetDataType(MultiTypeMatrix::DATATYPE_FLOAT);
                targetPipe4[i].SetDataType(MultiTypeMatrix::DATATYPE_FLOAT);
                targetPipe5[i].SetDataType(MultiTypeMatrix::DATATYPE_FLOAT);
            break;

            case ChannelInfo::DATATYPE_INDEX:
                verify(outputProbe[i]->GetLayerSize() == channelInfo.frameDim);
                verify(channelInfo.frameSize == 1);

            case ChannelInfo::DATATYPE_SEQ:
                target[i].SetDataType(MultiTypeMatrix::DATATYPE_INT);
                targetPipe1[i].SetDataType(MultiTypeMatrix::DATATYPE_INT);
                targetPipe2[i].SetDataType(MultiTypeMatrix::DATATYPE_INT);
                targetPipe3[i].SetDataType(MultiTypeMatrix::DATATYPE_INT);
                targetPipe4[i].SetDataType(MultiTypeMatrix::DATATYPE_INT);
                targetPipe5[i].SetDataType(MultiTypeMatrix::DATATYPE_INT);
            break;

            default:
                verify(false);
        }

        unsigned long frameSize = channelInfo.frameSize;
        unsigned long layerSize = outputProbe[i]->GetLayerSize();

        target[i].Resize(frameSize, args.frameStep);
        target[i].SetEngine(engine);
        targetPipe1[i].Resize(frameSize, args.frameStep);
        targetPipe1[i].SetEngine(engine);
        targetPipe2[i].Resize(frameSize, args.frameStep);
        targetPipe2[i].SetEngine(engine);
        targetPipe3[i].Resize(frameSize, args.frameStep);
        targetPipe3[i].SetEngine(engine);
        targetPipe4[i].Resize(frameSize, args.frameStep);
        targetPipe4[i].SetEngine(engine);
        targetPipe5[i].Resize(frameSize, args.frameStep);
        targetPipe5[i].SetEngine(engine);

        output[i].Resize(layerSize, args.frameStep);
        output[i].SetEngine(engine);
        outputBuf[i].Resize(layerSize, args.frameStep);
        outputBuf[i].SetEngine(engine);
    }


    /* Initialize the RNN */
    rnn.SetBatchSize(args.nStream, args.frameStep / args.nStream);
    rnn.InitForward(0, args.frameStep / args.nStream - 1);
    rnn.ProcessWeights((FLOAT) 0);
    rnn.Synchronize();


    /* Main loop */
    engine->StreamCreate(pStreamDataTransferToBuf, LOC);
    engine->StreamCreate(pStreamDataTransferToRnn, LOC);
    engine->StreamCreate(pStreamDataTransferFromRnn, LOC);
    engine->StreamCreate(pStreamDataTransferFromBuf, LOC);
    engine->EventCreate(pEventDataTransferToBuf, LOC);
    engine->EventCreate(pEventDataTransferToRnn, LOC);
    engine->EventCreate(pEventDataTransferFromRnn, LOC);
    engine->EventCreate(pEventDataTransferFromBuf, LOC);

    pipe[0].Init();
    pipe[1].Init();
    pipe[2].Init();
    pipe[3].Init();
    pipe[4].Init();
    pipe[5].Init();
    pipe[6].Init();

    std::thread thdPipe0(EvaluatePipe0, this, std::ref(args));
    std::thread thdPipe1(EvaluatePipe1, this, std::ref(args));
    std::thread thdPipe2(EvaluatePipe2, this, std::ref(args));
    std::thread thdPipe3(EvaluatePipe3, this, std::ref(args));
    std::thread thdPipe4(EvaluatePipe4, this, std::ref(args));
    std::thread thdPipe5(EvaluatePipe5, this, std::ref(args));
    std::thread thdPipe6(EvaluatePipe6, this, std::ref(args));

    pipe[0].SendSignal();
    pipe[1].SendSignal();
    pipe[2].SendSignal();
    pipe[4].SendSignal();
    pipe[5].SendSignal();

    thdPipe0.join();
    thdPipe1.join();
    thdPipe2.join();
    thdPipe3.join();
    thdPipe4.join();
    thdPipe5.join();
    thdPipe6.join();

    engine->StreamDestroy(pStreamDataTransferToBuf);
    engine->StreamDestroy(pStreamDataTransferToRnn);
    engine->StreamDestroy(pStreamDataTransferFromRnn);
    engine->StreamDestroy(pStreamDataTransferFromBuf);
    engine->EventDestroy(pEventDataTransferToBuf);
    engine->EventDestroy(pEventDataTransferToRnn);
    engine->EventDestroy(pEventDataTransferFromRnn);
    engine->EventDestroy(pEventDataTransferFromBuf);

    rnn.Synchronize();
}


void Evaluator::EvaluatePipe0(Evaluator *evaluator, EvaluateArgs &args)
{
    Engine *engine = args.rnn->GetEngine();

    for(unsigned long frameIdx = 0; frameIdx < args.numFrame; frameIdx += args.frameStep)
    {
        unsigned long batchFrom = 0;
        unsigned long batchTo = batchFrom + std::min(args.numFrame - frameIdx, args.frameStep) - 1;
        unsigned long nForwardFrame = batchTo - batchFrom + 1;

        evaluator->pipe[0].Wait(1);


        /* Wait until the memory transfer to the engine finishes */

        engine->StreamSynchronize(evaluator->pStreamDataTransferToBuf);


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

        evaluator->pipe[1].SendSignal();
    }
}


void Evaluator::EvaluatePipe1(Evaluator *evaluator, EvaluateArgs &args)
{
    Engine *engine = args.rnn->GetEngine();

    for(unsigned long frameIdx = 0; frameIdx < args.numFrame; frameIdx += args.frameStep)
    {
        evaluator->pipe[1].Wait(2);

        engine->StreamWaitEvent(evaluator->pStreamDataTransferToBuf, evaluator->pEventDataTransferToRnn);

        /* Copy the input sequences to the buffers */

        for(unsigned long i = 0; i < args.nInput; i++)
        {
            args.inputBuf[i].Swap(args.input[i]);

            switch(args.inputBuf[i].GetDataType())
            {
                case MultiTypeMatrix::DATATYPE_FLOAT:
                    {
                        Matrix<FLOAT> *matrix = reinterpret_cast<Matrix<FLOAT> *>(args.inputBuf[i].GetMatrix());
                        engine->MemPull(matrix->GetMem(), LOC, evaluator->pStreamDataTransferToBuf);
                    }
                    break;

                case MultiTypeMatrix::DATATYPE_INT:
                    {
                        Matrix<INT> *matrix = reinterpret_cast<Matrix<INT> *>(args.inputBuf[i].GetMatrix());
                        engine->MemPull(matrix->GetMem(), LOC, evaluator->pStreamDataTransferToBuf);
                    }
                    break;

                default:
                    verify(false);
            }

            //args.inputBuf[i].Pull(LOC, evaluator->pStreamDataTransferToBuf);
        }

        /* Propagate the target sequences */
        for(unsigned long i = 0; i < args.nOutput; i++)
        {
            args.targetPipe1[i].Swap(args.target[i]);
        }

        engine->EventRecord(evaluator->pEventDataTransferToBuf, evaluator->pStreamDataTransferToBuf);

        evaluator->pipe[0].SendSignal();
        evaluator->pipe[2].SendSignal();
    }
}


void Evaluator::EvaluatePipe2(Evaluator *evaluator, EvaluateArgs &args)
{
    Engine *engine = args.rnn->GetEngine();

    for(unsigned long frameIdx = 0; frameIdx < args.numFrame; frameIdx += args.frameStep)
    {
        unsigned long batchFrom = 0;
        unsigned long batchTo = batchFrom + std::min(args.numFrame - frameIdx, args.frameStep) - 1;

        evaluator->pipe[2].Wait(2);

        engine->StreamWaitEvent(evaluator->pStreamDataTransferToRnn, evaluator->pEventDataTransferToBuf);
        engine->StreamWaitEvent(evaluator->pStreamDataTransferToRnn, evaluator->pEventDataTransferFromRnn);

        /* Copy the input sequences to the RNN */

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

                        engine->MatCopy(inputSub, stateSub, evaluator->pStreamDataTransferToRnn);
                    }
                    break;

                case ChannelInfo::DATATYPE_INDEX:
                    {
                        Matrix<INT> *matrix = reinterpret_cast<Matrix<INT> *>(args.inputBuf[i].GetMatrix());
                        Matrix<INT> inputSub(*matrix, 0, batchTo - batchFrom);

                        engine->OneHotEncode(inputSub, stateSub, evaluator->pStreamDataTransferToRnn);
                    }
                    break;

                default:
                    verify(false);
            }

            engine->EventRecord(evaluator->pEventDataTransferToRnn, evaluator->pStreamDataTransferToRnn);
            engine->StreamWaitEvent(args.inputProbe[i]->GetPStream(), evaluator->pEventDataTransferToRnn);

            args.inputProbe[i]->EventRecord();
        }

        /* Propagate the target sequences */
        for(unsigned long i = 0; i < args.nOutput; i++)
        {
            args.targetPipe2[i].Swap(args.targetPipe1[i]);
        }

        evaluator->pipe[1].SendSignal();
        evaluator->pipe[3].SendSignal();
    }
}


void Evaluator::EvaluatePipe3(Evaluator *evaluator, EvaluateArgs &args)
{
    for(unsigned long frameIdx = 0; frameIdx < args.numFrame; frameIdx += args.frameStep)
    {
        unsigned long batchFrom = 0;
        unsigned long batchTo = batchFrom + std::min(args.numFrame - frameIdx, args.frameStep) - 1;

        evaluator->pipe[3].Wait(1);

        /* Forward pass */
#ifdef FRACTAL_PIPELINE
        long pStepSize = (PIPELINE_WIDTH + args.nStream - 1) / args.nStream * args.nStream;

        for(long i = (long) batchFrom; i <= (long) batchTo; i += pStepSize)
            args.rnn->Forward(i / args.nStream, std::min(i + pStepSize - 1, (long) batchTo) / args.nStream);
#else
        args.rnn->Forward(batchFrom / args.nStream, batchTo / args.nStream);
#endif /* FRACTAL_PIPELINE */


        for(unsigned long i = 0; i < args.nOutput; i++)
        {
            args.outputProbe[i]->Wait();
            args.outputProbe[i]->EventRecord();
        }

        /* Propagate the target sequences */
        for(unsigned long i = 0; i < args.nOutput; i++)
        {
            args.targetPipe3[i].Swap(args.targetPipe2[i]);
        }

        evaluator->pipe[4].SendSignal();
    }
}


void Evaluator::EvaluatePipe4(Evaluator *evaluator, EvaluateArgs &args)
{
    Engine *engine = args.rnn->GetEngine();

    for(unsigned long frameIdx = 0; frameIdx < args.numFrame; frameIdx += args.frameStep)
    {
        unsigned long batchFrom = 0;
        unsigned long batchTo = batchFrom + std::min(args.numFrame - frameIdx, args.frameStep) - 1;

        evaluator->pipe[4].Wait(2);

        engine->StreamWaitEvent(evaluator->pStreamDataTransferFromRnn, evaluator->pEventDataTransferFromBuf);

        /* Copy the output from the RNN */
        for(unsigned long i = 0; i < args.nOutput; i++)
        {
            Matrix<FLOAT> actSub(args.outputProbe[i]->GetActivation(), batchFrom, batchTo);
            Matrix<FLOAT> outputBufSub(args.outputBuf[i], 0, batchTo - batchFrom);

            args.outputProbe[i]->StreamWaitEvent(evaluator->pStreamDataTransferFromRnn);
            engine->MatCopy(actSub, outputBufSub, evaluator->pStreamDataTransferFromRnn);
        }

        /* Propagate the target sequences */
        for(unsigned long i = 0; i < args.nOutput; i++)
        {
            args.targetPipe4[i].Swap(args.targetPipe3[i]);
        }

        engine->EventRecord(evaluator->pEventDataTransferFromRnn, evaluator->pStreamDataTransferFromRnn);

        evaluator->pipe[2].SendSignal();
        evaluator->pipe[5].SendSignal();
    }
}


void Evaluator::EvaluatePipe5(Evaluator *evaluator, EvaluateArgs &args)
{
    Engine *engine = args.rnn->GetEngine();

    for(unsigned long frameIdx = 0; frameIdx < args.numFrame; frameIdx += args.frameStep)
    {
        evaluator->pipe[5].Wait(2);

        engine->StreamWaitEvent(evaluator->pStreamDataTransferFromBuf, evaluator->pEventDataTransferFromRnn);

        /* Copy the output from the buffer */
        for(unsigned long i = 0; i < args.nOutput; i++)
        {
            args.output[i].Swap(args.outputBuf[i]);
            args.output[i].HostPull(evaluator->pStreamDataTransferFromBuf);
            //args.outputBuf[i].Export(args.output[i], evaluator->pStreamDataTransferFromBuf);
        }

        /* Propagate the target sequences */
        for(unsigned long i = 0; i < args.nOutput; i++)
        {
            args.targetPipe5[i].Swap(args.targetPipe4[i]);
        }

        engine->EventRecord(evaluator->pEventDataTransferFromBuf, evaluator->pStreamDataTransferFromBuf);

        evaluator->pipe[4].SendSignal();
        evaluator->pipe[6].SendSignal();
    }
}


void Evaluator::EvaluatePipe6(Evaluator *evaluator, EvaluateArgs &args)
{
    Engine *engine = args.rnn->GetEngine();

    for(unsigned long frameIdx = 0; frameIdx < args.numFrame; frameIdx += args.frameStep)
    {
        unsigned long batchFrom = 0;
        unsigned long batchTo = batchFrom + std::min(args.numFrame - frameIdx, args.frameStep) - 1;

        evaluator->pipe[6].Wait(1);

        engine->StreamSynchronize(evaluator->pStreamDataTransferFromBuf);
        //args.rnn->Synchronize();

        /* Evaluate frames */

        if(frameIdx + batchTo - batchFrom >= evaluator->initFrameSkip * args.nStream)
        {
            unsigned long batchFromOffset = std::max((long) 0, (long) (evaluator->initFrameSkip * args.nStream) - (long) frameIdx);
        for(unsigned long i = 0; i < args.nOutput; i++)
        {
            Matrix<FLOAT> outputSub(args.output[i], batchFromOffset, batchTo - batchFrom);
            MultiTypeMatrix targetSub(args.targetPipe5[i], batchFromOffset, batchTo - batchFrom);

            args.outputProbe[i]->EvaluateOnHost(targetSub, outputSub);
        }
}

        evaluator->pipe[5].SendSignal();
    }
}

}

