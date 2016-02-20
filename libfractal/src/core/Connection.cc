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


#include "Connection.h"

#include <fstream>
#include <sstream>

#include "Layer.h"
#include "InitWeightParam.h"
//#define FRACTAL_VERBOSE

namespace fractal{

Connection::Connection(Layer *const from, Layer *const to, const ConnParam &_param)
{
    srcLayer = from;
    dstLayer = to;

    param = _param;

    if(param.srcRangeFrom == -1)
    {
        param.srcRangeFrom = 0;
    }

    if(param.srcRangeTo == -1)
    {
        param.srcRangeTo = srcLayer->GetSize() - 1;
    }

    if(param.dstRangeFrom == -1)
    {
        param.dstRangeFrom = 0;
    }

    if(param.dstRangeTo == -1)
    {
        param.dstRangeTo = dstLayer->GetSize() - 1;
    }

    verify(param.srcRangeFrom <= param.srcRangeTo);
    verify(param.srcRangeTo < (long) srcLayer->GetSize());
    verify(param.dstRangeFrom <= param.dstRangeTo);
    verify(param.dstRangeTo < (long) dstLayer->GetSize());

    srcSize = param.srcRangeTo - param.srcRangeFrom + 1;
    dstSize = param.dstRangeTo - param.dstRangeFrom + 1;

    nStream = 0;
    nUnroll = 0;

    switch(param.connType)
    {
        case CONN_FULL:
            weights.Resize(dstSize, srcSize);
            weightsFwd.Resize(dstSize, srcSize);
            weightsBwd.Resize(dstSize, srcSize);
            weightsTrans.Resize(srcSize, dstSize);
            vels.Resize(dstSize, srcSize);

            weightsFwd.Link(weights);
            weightsBwd.Link(weights);
            break;

        case CONN_IDENTITY:
            verify(srcSize == dstSize);
            break;

        case CONN_BROADCAST:
            verify(srcSize == 1);
            /* TODO: More efficient implementation is possible */
            weights.Resize(dstSize, srcSize);
            weightsFwd.Resize(dstSize, srcSize);
            weightsBwd.Resize(dstSize, srcSize);
            weightsTrans.Resize(srcSize, dstSize);

            weightsFwd.Link(weights);
            weightsBwd.Link(weights);
            break;

        default:
            verify(false);
    }

    engine = NULL;
    stream = NULL;

    rmsDecayRate = (FLOAT) 0.9;

    weightsTransValid = false;

    fixed = false;

    /* TODO: More efficient implementation is possible */
    broadcastWeightInitialized = false;
}


Connection::~Connection()
{
    SetEngine(NULL);
}


void Connection::SetEngine(Engine *const engine)
{
    if(this->engine == engine) return;

    weights.SetEngine(engine);
    weightsFwd.SetEngine(engine);
    weightsBwd.SetEngine(engine);
    weightsTrans.SetEngine(engine);
    vels.SetEngine(engine);
    derivs.SetEngine(engine);
    msDeriv.SetEngine(engine);
    msDelta.SetEngine(engine);
    dstAct.SetEngine(engine);
    srcAct.SetEngine(engine);
    dstErr.SetEngine(engine);
    srcErr.SetEngine(engine);

    weightsTransValid = false;

    if(this->engine != NULL)
    {
        SetPStream(NULL);
    }

    this->engine = engine;
}


void Connection::SetBatchSize(const unsigned long nStream, const unsigned long nUnroll)
{
    if(this->nStream == nStream && this->nUnroll == nUnroll) return;

    this->nStream = nStream;
    this->nUnroll = nUnroll;

    unsigned long batchSize = nStream * nUnroll;

    srcAct.Resize(srcSize, batchSize);
    dstAct.Resize(dstSize, batchSize);
    srcErr.Resize(srcSize, batchSize);
    dstErr.Resize(dstSize, batchSize);
}


void Connection::UnlinkMatrices()
{
    //weights.Unlink();
    //weightsTrans.Unlink();
    //vels.Unlink();
    //derivs.Unlink();
    //msDeriv.Unlink();
    //msDelta.Unlink();
    dstAct.Unlink();
    srcAct.Unlink();
    dstErr.Unlink();
    srcErr.Unlink();

    weightsTransValid = false;
}


void Connection::InitWeights(const InitWeightParam &initWeightParam)
{
    if(initWeightParam.IsValid() == false) return;
    if(fixed == true) return;

    verify(engine != NULL);

    if(GetNumWeights() > 0)
    {
        bool success = false;

        try
        {
            const InitWeightParamGaussian &paramGaussian = dynamic_cast<const InitWeightParamGaussian &>(initWeightParam);
            verify(paramGaussian.stdev >= (FLOAT) 0);
            engine->MatRandN(weights, paramGaussian.mean, paramGaussian.stdev, *stream);
            success = true;
        }
        catch(...) {}

        try
        {
            const InitWeightParamUniform &paramUniform = dynamic_cast<const InitWeightParamUniform &>(initWeightParam);
            verify(paramUniform.b >= paramUniform.a);
            engine->MatRandU(weights, paramUniform.a, paramUniform.b, *stream);
            success = true;
        }
        catch(...) {}

        if(success == false)
        {
            /* Unsupported distribution */
            verify(false);
        }

        if(initWeightParam.addToDiag != (FLOAT) 0)
        {
            verify(srcSize == dstSize);
            engine->MatAddToDiag(weights, (FLOAT) initWeightParam.addToDiag, 0, *stream);
        }

        weightsFwd.Link(weights);
        weightsBwd.Link(weights);

        weightsTransValid = false;
    }
}


void Connection::InitAdadelta(const FLOAT decayRate, const bool initDenominator)
{
    verify(engine != NULL);

    if(fixed == true) return;

    if(GetNumWeights() > 0)
    {
        derivs.Resize(weights.GetNumRows(), weights.GetNumCols());
        msDeriv.Resize(weights.GetNumRows(), weights.GetNumCols());
        msDelta.Resize(weights.GetNumRows(), weights.GetNumCols());

        rmsDecayRate = decayRate;

        if(initDenominator == true)
        {
            engine->MatSet(msDeriv, (FLOAT) 1, *stream);
        }

        engine->MatSet(msDelta, (FLOAT) 0, *stream);
    }
}


void Connection::InitNesterov()
{
    verify(engine != NULL);

    if(fixed == true) return;

    if(GetNumWeights() > 0)
    {
        engine->MatSet(vels, (FLOAT) 0, *stream);
    }
}


void Connection::InitRmsprop(const FLOAT decayRate)
{
    verify(engine != NULL);

    if(fixed == true) return;

    if(GetNumWeights() > 0)
    {
        derivs.Resize(weights.GetNumRows(), weights.GetNumCols());
        msDeriv.Resize(weights.GetNumRows(), weights.GetNumCols());

        rmsDecayRate = decayRate;

        engine->MatSet(msDeriv, (FLOAT) 1, *stream);
    }
}


void Connection::InitErr(const unsigned long idxFrom, const unsigned long idxTo)
{
    verify(idxFrom >= 0 && idxTo < nUnroll && idxFrom <= idxTo);
    verify(engine != NULL && stream != NULL);

    Matrix<FLOAT> srcErrSub(srcErr, idxFrom * nStream, idxTo * nStream + nStream - 1);

    engine->MatSet(srcErrSub, (FLOAT) 0, *stream);

    //EventRecord();
}


void Connection::Forward(const unsigned long idxFrom, const unsigned long idxTo)
{
    verify(idxFrom >= 0 && idxTo < nUnroll && idxFrom <= idxTo);
    verify(engine != NULL && stream != NULL);

    unsigned long actFrom, actTo, delay;


#ifdef FRACTAL_VERBOSE
    printf("Connection::Forward: %s -> %s %s (%ld, %ld)\n",
            srcLayer->GetName().c_str(), dstLayer->GetName().c_str(),
            IsDelayed() == true ? "(DELAYED)" : "",
            idxFrom, idxTo);
#endif /* FRACTAL_VERBOSE */

    //srcLayer->StreamWaitEvent(*stream);

    /* TODO: More efficient implementation is possible */
    {
        if(param.connType == CONN_BROADCAST && broadcastWeightInitialized == false)
        {
            engine->MatSet(weights, (FLOAT) 1, *stream);

            weightsFwd.Link(weights);
            weightsBwd.Link(weights);

            broadcastWeightInitialized = true;
            weightsTransValid = false;
        }
    }

    if(IsDelayed() == true)
    {
        delay = param.delayAmount;
        verify(delay < nUnroll);

        actFrom = (idxFrom + nUnroll - delay) % nUnroll;
        actTo = (idxTo + nUnroll - delay) % nUnroll;

        verify(actFrom >= 0 && actTo < nUnroll && actFrom <= actTo);

        Matrix<FLOAT> srcLayerActSub(srcLayer->act, param.srcRangeFrom, param.srcRangeTo,
                actFrom * nStream, actTo * nStream + nStream - 1);
        Matrix<FLOAT> srcActSub(srcAct, idxFrom * nStream, idxTo * nStream + nStream - 1);

        engine->MatCopy(srcLayerActSub, srcActSub, *stream);
    }
    else
    {
        if(stream->loc != srcLayer->stream->loc)
        {
            Matrix<FLOAT> srcLayerActSub(srcLayer->act, param.srcRangeFrom, param.srcRangeTo,
                    idxFrom * nStream, idxTo * nStream + nStream - 1);
            Matrix<FLOAT> srcActSub(srcAct, idxFrom * nStream, idxTo * nStream + nStream - 1);

            engine->MatCopy(srcLayerActSub, srcActSub, *stream);
        }
        else
        {
            Matrix<FLOAT> srcLayerActSub(srcLayer->act, param.srcRangeFrom, param.srcRangeTo,
                    0, srcLayer->act.GetNumCols() - 1);

            srcAct.Link(srcLayerActSub);
        }
    }

    switch(param.connType)
    {
        case CONN_FULL:
            {
                Matrix<FLOAT> srcActSub(srcAct, idxFrom * nStream, idxTo * nStream + nStream - 1);
                Matrix<FLOAT> dstActSub(dstAct, idxFrom * nStream, idxTo * nStream + nStream - 1);

                engine->MatMult(weightsFwd, false, srcActSub, false, dstActSub, (FLOAT) 1, (FLOAT) 0, *stream);
                //if(IsDelayed() == true)
                //    engine->FuncBoundRange(dstActSub, dstActSub, (FLOAT) -10, (FLOAT) 10, *stream);
            }
            break;

        case CONN_IDENTITY:
            dstAct.Link(srcAct);
            break;

        case CONN_BROADCAST:
            /* TODO: More efficient implementation is possible */
            {
                Matrix<FLOAT> srcActSub(srcAct, idxFrom * nStream, idxTo * nStream + nStream - 1);
                Matrix<FLOAT> dstActSub(dstAct, idxFrom * nStream, idxTo * nStream + nStream - 1);

                engine->MatMult(weightsFwd, false, srcActSub, false, dstActSub, (FLOAT) 1, (FLOAT) 0, *stream);
            }
            break;

        default:
            verify(false);
    }

    //EventRecord();
}


void Connection::UpdateDstErr(const unsigned long idxFrom, const unsigned long idxTo)
{
    verify(idxFrom >= 0 && idxTo < nUnroll && idxFrom <= idxTo);
    //verify(engine != NULL && stream != NULL);

    if(dstLayer->performBackward == false) return;

#ifdef FRACTAL_VERBOSE
    printf("Connection::UpdateDstErr: %s <- %s (%ld, %ld)\n",
            srcLayer->GetName().c_str(), dstLayer->GetName().c_str(),
            idxFrom, idxTo);
#endif /* FRACTAL_VERBOSE */

    //dstLayer->StreamWaitEvent(*stream);
    dstLayer->DistributeErr(this, idxFrom, idxTo);
}


void Connection::Backward(const unsigned long idxFrom, const unsigned long idxTo)
{
    verify(idxFrom >= 0 && idxTo < nUnroll && idxFrom <= idxTo);
    verify(engine != NULL && stream != NULL);

    unsigned long srcErrFrom, srcErrTo, delay;

    if(srcLayer->performBackward == false) return;

#ifdef FRACTAL_VERBOSE
    printf("Connection::Backward: %s <- %s %s (%ld, %ld)\n",
            srcLayer->GetName().c_str(), dstLayer->GetName().c_str(),
            IsDelayed() == true ? "(DELAYED)" : "",
            idxFrom, idxTo);
#endif /* FRACTAL_VERBOSE */


    delay = IsDelayed() == true ? param.delayAmount : 0;
    verify(delay < nUnroll);

    srcErrFrom = (idxFrom + nUnroll - delay) % nUnroll;
    srcErrTo = (idxTo + nUnroll - delay) % nUnroll;
    verify(srcErrFrom >= 0 && srcErrTo < nUnroll && srcErrFrom <= srcErrTo);

    Matrix<FLOAT> srcErrSub(srcErr, srcErrFrom * nStream, srcErrTo * nStream + nStream - 1);
    Matrix<FLOAT> dstErrSub(dstErr, idxFrom * nStream, idxTo * nStream + nStream - 1);

    if(dstLayer->performBackward == false)
    {
        engine->MatSet(srcErr, (FLOAT) 0, *stream);
        return;
    }

    switch(param.connType)
    {
        case CONN_FULL:
            //engine->MatMult(weightsBwd, true, dstErrSub, false, srcErrSub, (FLOAT) 1, (FLOAT) 0, *stream);
            if(weightsTransValid == false)
            {
                TransposeWeightMatrix();
            }

            engine->MatMult(weightsTrans, false, dstErrSub, false, srcErrSub, (FLOAT) 1, (FLOAT) 0, *stream);

            /* Clip the gradients */
            //if(IsDelayed() == true)
            //    engine->FuncBoundRange(srcErrSub, srcErrSub, (FLOAT) -1, (FLOAT) 1, *stream);
            break;

        case CONN_IDENTITY:
            if(IsDelayed() == true)
            {
                engine->MatCopy(dstErrSub, srcErrSub, *stream);
            }
            else
            {
                srcErr.Link(dstErr);
            }
            break;

        case CONN_BROADCAST:
            /* TODO: More efficient implementation is possible */
            if(weightsTransValid == false)
            {
                TransposeWeightMatrix();
            }

            engine->MatMult(weightsTrans, false, dstErrSub, false, srcErrSub, (FLOAT) 1, (FLOAT) 0, *stream);

            break;

        default:
            verify(false);
    }

    //EventRecord();
}


void Connection::UpdateWeights(const unsigned long idxFrom, const unsigned long idxTo,
        const FLOAT rate, const FLOAT momentum, const bool adadelta, const bool rmsprop)
{
    verify(idxFrom >= 0 && idxTo < nUnroll && idxFrom <= idxTo);
    verify(engine != NULL && stream != NULL);

    if(GetNumWeights() == 0) return;
    if(fixed == true) return;
    if(dstLayer->performBackward == false) return;


#ifdef FRACTAL_VERBOSE
    printf("Connection::UpdateWeights: %s -> %s (%ld, %ld)\n",
            srcLayer->GetName().c_str(), dstLayer->GetName().c_str(),
            idxFrom, idxTo);
#endif /* FRACTAL_VERBOSE */

    Matrix<FLOAT> srcActSub(srcAct, idxFrom * nStream, idxTo * nStream + nStream - 1);
    Matrix<FLOAT> dstErrSub(dstErr, idxFrom * nStream, idxTo * nStream + nStream - 1);

    /* Simplified Nesterov momentum (Yoshua Bengio) */
    engine->MatAdd(vels, weights, -momentum, *stream);

    if(adadelta == true)
    {
        verify(rmsprop == false);

        switch(param.connType)
        {
            case CONN_FULL:
                engine->MatMult(dstErrSub, false, srcActSub, true,
                        derivs, (FLOAT) 1, (FLOAT) 0, *stream);
                break;

            default:
                verify(false);
        }
        engine->Adadelta(derivs, derivs, msDeriv, msDelta, rate, rmsDecayRate, *stream);
        engine->MatAdd(vels, vels, momentum - (FLOAT) 1, *stream); // vels *= momentum
        engine->MatAdd(derivs, vels, (FLOAT) 1, *stream); // vels += rate * derivs
    }
    else if(rmsprop == true)
    {
        //engine->MatMult(dstErrSub, false, srcActSub, true, derivs, (FLOAT) 1 / (FLOAT) nFrame, (FLOAT) 0, *stream);
        switch(param.connType)
        {
            case CONN_FULL:
                engine->MatMult(dstErrSub, false, srcActSub, true,
                        derivs, (FLOAT) 1, (FLOAT) 0, *stream);
                break;

            default:
                verify(false);
        }
        engine->Rmsprop(derivs, derivs, msDeriv, rmsDecayRate, *stream);
        engine->MatAdd(vels, vels, momentum - (FLOAT) 1, *stream); // vels *= momentum
        engine->MatAdd(derivs, vels, rate, *stream); // vels += rate * derivs
    }
    else
    {
        /* vels = momentum * vels + rate * derivs */
        //engine->MatMult(dstErrSub, false, srcActSub, true, vels, rate / (FLOAT) nFrame, momentum, *stream);
        switch(param.connType)
        {
            case CONN_FULL:
                engine->MatMult(dstErrSub, false, srcActSub, true, vels, rate, momentum, *stream);
                break;

            default:
                verify(false);
        }
    }

    engine->MatAdd(vels, weights, (FLOAT) 1 + momentum, *stream);


    weightsTransValid = false;
}


void Connection::ProcessWeights(const FLOAT noise)
{
    verify(engine != NULL);

    if(GetNumWeights() == 0) return;
    if(fixed == true) return;

    if(noise <= (FLOAT) 0)
    {
        weightsFwd.Link(weights);
        weightsBwd.Link(weights);
        return;
    }

    weightsFwd.Unlink();
    weightsBwd.Unlink();

#if 1
    /* Additive noise */
    engine->MatRandN(weightsFwd, (FLOAT) 0, noise, *stream);
    engine->FuncBoundRange(weightsFwd, weightsFwd, (FLOAT) -3 * noise, (FLOAT) 3 * noise, *stream);
    engine->MatAdd(weights, weightsFwd, (FLOAT) 1, *stream);
#else
    /* Multiplicative noise */
    engine->MatRandN(weightsFwd, (FLOAT) 1, noise, *stream);
    engine->FuncBoundRange(weightsFwd, weightsFwd, (FLOAT) 0, (FLOAT) 2, *stream);
    engine->MatElemMult(weights, weightsFwd, weightsFwd, *stream);
#endif

    weightsBwd.Link(weightsFwd);
    weightsBwd.Link(weights);
    weightsTransValid = false;
}


void Connection::FixWeights(const bool enable)
{
    fixed = enable;
}


void Connection::SetPStream(PStream *const stream)
{
    if(this->stream == stream) return;

    verify(engine != NULL);

    if(this->stream != NULL)
    {
        engine->EventDestroy(event);
    }
    if(stream != NULL)
    {
        verify(stream->engine == engine);
        engine->EventCreate(event, stream->loc);
    }

    this->stream = stream;
}


PStream &Connection::GetPStream()
{
    verify(engine != NULL);
    return *stream;
}


void Connection::EventRecord()
{
    verify(engine != NULL);
    engine->EventRecord(event, *stream);
}


void Connection::StreamWaitEvent(PStream &stream)
{
    verify(engine != NULL);
    engine->StreamWaitEvent(stream, event);
}


void Connection::ForwardWait()
{
    srcLayer->StreamWaitEvent(*stream);
}


void Connection::BackwardWait()
{
    dstLayer->StreamWaitEvent(*stream);
}


void Connection::TransposeWeightMatrix()
{
    verify(engine != NULL);

    switch(param.connType)
    {
        case CONN_FULL:
            engine->MatTranspose(weightsBwd, weightsTrans, *stream);
            weightsTransValid = true;
            break;

        case CONN_IDENTITY:
            break;

        case CONN_BROADCAST:
            /* TODO: More efficient implementation is possible */
            engine->MatTranspose(weightsBwd, weightsTrans, *stream);
            weightsTransValid = true;
            break;

        default:
            verify(false);
    }
}


void Connection::SaveState(const std::string &filename)
{
    if(GetNumWeights() == 0) return;

    /* Save weights, vels, msDeriv */

    weights.Save(filename + ".weights");
    vels.Save(filename + ".vels");
    msDeriv.Save(filename + ".msDeriv");
    msDelta.Save(filename + ".msDelta");


    /* Save rmsDecayRate */

    std::string paramFileName = filename + ".param";
    std::ofstream paramFile;

    paramFile.open(paramFileName, std::ios_base::out);

    verify(paramFile.is_open() == true);

    paramFile << "rmsDecayRate = " << rmsDecayRate << std::endl;

    verify(paramFile.good() == true);

    paramFile.close();
}


void Connection::LoadState(const std::string &filename)
{
    if(GetNumWeights() == 0) return;

    /* Load weights, vels, msDeriv */

    weights.Load(filename + ".weights");
    vels.Load(filename + ".vels");
    msDeriv.Load(filename + ".msDeriv");
    msDelta.Load(filename + ".msDelta");

    weightsFwd.Link(weights);
    weightsBwd.Link(weights);

    weightsTransValid = false;

    /* Load rmsDecayRate */

    std::string paramFileName = filename + ".param";
    std::ifstream paramFile;
    std::string buf, bufLHS, bufRHS;
    size_t pos1, pos2, pos3, pos4, pos5;

    paramFile.open(paramFileName, std::ios_base::in);

    verify(paramFile.is_open() == true);

    while(paramFile.eof() == false)
    {
        std::getline(paramFile, buf);
        verify(paramFile.bad() == false);

        pos1 = buf.find_first_not_of(" \n\r\t");
        if(pos1 == std::string::npos) continue;

        pos5 = buf.find_last_not_of(" \n\r\t");

        pos3 = buf.find_first_of('=');
        verify(pos3 != std::string::npos);
        verify(pos3 > pos1 && pos3 < pos5);

        pos2 = buf.find_last_not_of(" \n\r\t", pos3 - 1);
        pos4 = buf.find_first_not_of(" \n\r\t", pos3 + 1);

        bufLHS = buf.substr(pos1, pos2 - pos1 + 1);
        bufRHS = buf.substr(pos4, pos5 - pos4 + 1);

        //std::cout << bufLHS << "=" << bufRHS << std::endl;

        if(bufLHS == "rmsDecayRate")
        {
            std::istringstream(bufRHS) >> rmsDecayRate;
        }
    }

    paramFile.close();
}


const unsigned long Connection::GetNumWeights()
{
    switch(param.connType)
    {
        case CONN_FULL:
            return srcSize * dstSize;
        case CONN_IDENTITY:
            return 0;
        case CONN_BROADCAST:
            return 0;
        default:
            verify(false);
            return 0;
    }
}

}

