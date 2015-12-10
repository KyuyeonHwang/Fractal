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


#include "Layer.h"

#include "Connection.h"
#include "Probe.h"
#include "TrainableProbe.h"
//#define FRACTAL_VERBOSE

namespace fractal
{

Layer::Layer(const std::string &name, ActType actType, AggType aggType, const unsigned long size, const LayerParam &param)
{
    verify(size > 0);

    this->name = name;
    this->actType = actType;
    this->aggType = aggType;
    this->size = size;
    this->param = param;

    nStream = 0;
    nUnroll = 0;

    dropoutEnabled = false;

    linkedProbe = NULL;

    engine = NULL;
    stream = NULL;

    performBackward = false;
}


Layer::~Layer()
{
    SetEngine(NULL);

    UnlinkProbe();
}


void Layer::SetEngine(Engine *const engine)
{
    if(this->engine == engine) return;

    act.SetEngine(engine);
    state.SetEngine(engine);
    srcErr.SetEngine(engine);
    dstErr.SetEngine(engine);
    dropoutMask.SetEngine(engine);
    idxMax.SetEngine(engine);

    if(IsLinked() == true)
    {
        linkedProbe->SetEngine(engine);
    }

    if(this->engine != NULL)
    {
        SetPStream(NULL);
    }

    this->engine = engine;
}


void Layer::SetBatchSize(const unsigned long nStream, const unsigned long nUnroll)
{
    if(this->nStream == nStream && this->nUnroll == nUnroll) return;

    this->nStream = nStream;
    this->nUnroll = nUnroll;

    unsigned long batchSize = nStream * nUnroll;

    act.Resize(size, batchSize);
    state.Resize(size, batchSize);
    srcErr.Resize(size, batchSize);
    dstErr.Resize(size, batchSize);

    if(actType == ACT_DROPOUT)
    {
        dropoutMask.Resize(size, batchSize);
    }

    if(actType == ACT_CTC_DECODE)
    {
        idxMax.Resize(1, batchSize);
    }
}


void Layer::EnableDropout(const bool enable)
{
    dropoutEnabled = enable && (actType == ACT_DROPOUT) && (param.dropoutRate > (FLOAT) 0);
}


void Layer::UnlinkMatrices()
{
    act.Unlink();
    state.Unlink();
    srcErr.Unlink();
    dstErr.Unlink();
}


void Layer::SetInitVal(const FLOAT val)
{
    param.initVal = val;
}


void Layer::InitAct(const unsigned long idxFrom, const unsigned long idxTo)
{
    verify(idxFrom >= 0 && idxTo < nUnroll && idxFrom <= idxTo);
    verify(engine != NULL && stream != NULL);

    Matrix<FLOAT> actSub(act, idxFrom * nStream, idxTo * nStream + nStream - 1);

    engine->MatSet(actSub, param.initVal, *stream);

    if(actType == ACT_CTC_DECODE)
    {
        Matrix<INT> idxMaxSub(idxMax, idxFrom * nStream, idxTo * nStream + nStream - 1);
        engine->MatSet(idxMaxSub, size - 1, *stream); /* Set to blank label */
    }

    performBackward = false;
    //EventRecord();
}


void Layer::InitErr(const unsigned long idxFrom, const unsigned long idxTo)
{
    verify(idxFrom >= 0 && idxTo < nUnroll && idxFrom <= idxTo);
    verify(engine != NULL && stream != NULL);

    Matrix<FLOAT> dstErrSub(dstErr, idxFrom * nStream, idxTo * nStream + nStream - 1);

    engine->MatSet(dstErrSub, (FLOAT) 0, *stream);

    //EventRecord();
}


void Layer::GenerateDropoutMask(const unsigned long idxFrom, const unsigned long idxTo)
{
    verify(idxFrom >= 0 && idxTo < nUnroll && idxFrom <= idxTo);
    verify(engine != NULL && stream != NULL);

    if(actType != ACT_DROPOUT) return;

    if(dropoutEnabled == true)
    {
        Matrix<FLOAT> dropoutMaskSub(dropoutMask, idxFrom * nStream, idxTo * nStream + nStream - 1);

        engine->GenerateDropoutMask(dropoutMaskSub, param.dropoutRate, *stream);
    }
}


void Layer::Forward(const unsigned long idxFrom, const unsigned long idxTo)
{
    verify(idxFrom >= 0 && idxTo < nUnroll && idxFrom <= idxTo);

#ifdef FRACTAL_VERBOSE
    printf("Layer::Forward: %s (%ld, %ld)\n",
            GetName().c_str(), idxFrom, idxTo);
#endif /* FRACTAL_VERBOSE */

    if(IsLinked() == true && linkedProbe->IsInput() == true)
    {
        //linkedProbe->StreamWaitEvent(*stream);
    }
    else
    {
        UpdateState(idxFrom, idxTo);
    }

    Activation(idxFrom, idxTo);

    performBackward = false;

    for(auto &conn : srcList)
    {
        performBackward = performBackward
            || ((conn->GetNumWeights() > 0) && (conn->fixed == false))
            || (conn->srcLayer->performBackward == true);
    }

    if(param.blockErr == true)
    {
        performBackward = false;
    }
    //EventRecord();
}


void Layer::Backward(const unsigned long idxFrom, const unsigned long idxTo)
{
    verify(idxFrom >= 0 && idxTo < nUnroll && idxFrom <= idxTo);

    if(performBackward == false) return;

#ifdef FRACTAL_VERBOSE
    printf("Layer::Backward: %s (%ld, %ld)\n",
            GetName().c_str(), idxFrom, idxTo);
#endif /* FRACTAL_VERBOSE */


    UpdateDstErr(idxFrom, idxTo);
    UpdateSrcErr(idxFrom, idxTo);

    //EventRecord();
}


void Layer::CalcActDeriv(const unsigned long idxFrom, const unsigned long idxTo)
{
    verify(idxFrom >= 0 && idxTo < nUnroll && idxFrom <= idxTo);
    verify(engine != NULL && stream != NULL);

    if(performBackward == false) return;

#ifdef FRACTAL_VERBOSE
    printf("Layer::CalcActDeriv: %s (%ld, %ld)\n",
            GetName().c_str(), idxFrom, idxTo);
#endif /* FRACTAL_VERBOSE */

    Matrix<FLOAT> srcErrSub(srcErr, idxFrom * nStream, idxTo * nStream + nStream - 1);
    Matrix<FLOAT> actSub(act, idxFrom * nStream, idxTo * nStream + nStream - 1);

    switch(actType)
    {
        case ACT_SIGMOID:
            engine->FuncSigmoidDeriv(actSub, srcErrSub, *stream);
            break;

        case ACT_TANH:
            engine->FuncTanhDeriv(actSub, srcErrSub, *stream);
            break;

        case ACT_SOFTPLUS:
            engine->FuncSoftplusDeriv(actSub, srcErrSub, *stream);
            break;

        case ACT_RECTLINEAR:
            engine->FuncRectLinearDeriv(actSub, srcErrSub, *stream);
            break;

        case ACT_LINEAR:
            break;

        case ACT_ONE_MINUS_LINEAR:
            engine->MatSet(srcErrSub, (FLOAT) -1, *stream);
            break;

        case ACT_INVERSE:
            engine->MatSet(srcErrSub, (FLOAT) -1, *stream);
            break;

        case ACT_SOFTMAX:
            /* Approximate computation using diagonal Jacobi matrix */
            engine->FuncSigmoidDeriv(actSub, srcErrSub, *stream);
            /* Block error propagation */
            //engine->MatSet(srcErrSub, (FLOAT) 0, *stream);
            break;

        case ACT_DROPOUT:
            if(dropoutEnabled == true)
            {
                /* For bug-free code. This is inefficient. */
                Matrix<FLOAT> dropoutMaskSub(dropoutMask, idxFrom * nStream, idxTo * nStream + nStream - 1);
                engine->MatCopy(dropoutMaskSub, srcErrSub, *stream);
            }
            break;

        case ACT_CTC_DECODE:
            engine->MatSet(srcErrSub, (FLOAT) 0, *stream);
            break;

        default:
            verify(false);
    }
}


void Layer::AddSrcConnection(Connection *const conn)
{
    srcList.push_back(conn);
}


void Layer::AddDstConnection(Connection *const conn)
{
    dstList.push_back(conn);
}


void Layer::RemoveSrcConnection(Connection *const conn)
{
    srcList.remove(conn);
}


void Layer::RemoveDstConnection(Connection *const conn)
{
    dstList.remove(conn);
}


void Layer::LinkProbe(Probe *const probe)
{
    if(linkedProbe != probe)
    {
        UnlinkProbe();
        linkedProbe = probe;
        probe->LinkLayer(this);
    }
}


void Layer::UnlinkProbe()
{
    Probe *tmp;

    if(linkedProbe != NULL)
    {
        tmp = linkedProbe;
        linkedProbe = NULL;
        tmp->UnlinkLayer();
    }
}


const bool Layer::IsLinked() const
{
    return (linkedProbe != NULL);
}


void Layer::Activation(const unsigned long idxFrom, const unsigned long idxTo)
{
    verify(idxFrom >= 0 && idxTo < nUnroll && idxFrom <= idxTo);
    verify(engine != NULL && stream != NULL);

    Matrix<FLOAT> stateSub(state, idxFrom * nStream, idxTo * nStream + nStream - 1);
    Matrix<FLOAT> actSub(act, idxFrom * nStream, idxTo * nStream + nStream - 1);

    switch(actType)
    {
        case ACT_BIAS:
            engine->MatSet(actSub, (FLOAT) 1, *stream);
            break;

        case ACT_SIGMOID:
            engine->FuncSigmoid(stateSub, actSub, *stream);
            break;

        case ACT_TANH:
            engine->FuncTanh(stateSub, actSub, *stream);
            break;

        case ACT_SOFTPLUS:
            engine->FuncSoftplus(stateSub, actSub, *stream);
            break;

        case ACT_RECTLINEAR:
            engine->FuncRectLinear(stateSub, actSub, *stream);
            break;

        case ACT_LINEAR:
            //engine->MatCopy(stateSub, actSub, *stream);
            act.Link(state);
            break;

        case ACT_ONE_MINUS_LINEAR:
            engine->MatSet(actSub, (FLOAT) 1, *stream);
            engine->MatSub(actSub, stateSub, actSub, *stream);
            break;

        case ACT_INVERSE:
            engine->MatSet(actSub, (FLOAT) 0, *stream);
            engine->MatSub(actSub, stateSub, actSub, *stream);
            break;

        case ACT_SOFTMAX:
            engine->FuncSoftmax(stateSub, actSub, *stream);
            break;

        case ACT_DROPOUT:
            if(dropoutEnabled == true)
            {
                act.Unlink();
                Matrix<FLOAT> dropoutMaskSub(dropoutMask, idxFrom * nStream, idxTo * nStream + nStream - 1);
                engine->MatElemMult(dropoutMaskSub, stateSub, actSub, *stream);
            }
            else
            {
                act.Link(state);
            }
            break;

        case ACT_CTC_DECODE:
            {
                unsigned long prevIdx = ((idxFrom + nUnroll - 1) % nUnroll);

                Matrix<INT> prevIdxMax(idxMax, prevIdx * nStream, prevIdx * nStream + nStream - 1);
                Matrix<INT> idxMaxSub(idxMax, idxFrom * nStream, idxTo * nStream + nStream - 1);

                engine->FuncCTCDecode(stateSub, actSub, prevIdxMax, idxMaxSub, nStream, *stream);
            }
            break;

        default:
            verify(false);
    }
}


void Layer::UpdateState(const unsigned long idxFrom, const unsigned long idxTo)
{
    verify(idxFrom >= 0 && idxTo < nUnroll && idxFrom <= idxTo);
    verify(engine != NULL && stream != NULL);

    ConnList::const_iterator iter, iter_end;
    Matrix<FLOAT> stateSub(state, idxFrom * nStream, idxTo * nStream + nStream - 1);
    Connection *firstConn = NULL;
    bool isFirst = true;

    if(aggType == AGG_DONTCARE)
    {
        verify(srcList.empty() == true);
        if(actType != ACT_BIAS) verify(dynamic_cast<InputProbe *>(linkedProbe) != NULL);
    }
    else
    {
        verify(srcList.empty() == false);
        verify(dynamic_cast<InputProbe *>(linkedProbe) == NULL);
        verify(actType != ACT_BIAS);
    }

    iter_end = srcList.end();
    for(iter = srcList.begin(); iter != iter_end; ++iter)
    {
        //(*iter)->StreamWaitEvent(*stream);
        if((*iter)->param.dstRangeFrom != 0 || (*iter)->param.dstRangeTo != (long) GetSize() - 1)
        {
            continue;
        }

        if(isFirst == true)
        {
            firstConn = (*iter);
            isFirst = false;
        }
        else
        {
            Matrix<FLOAT> srcSub((*iter)->dstAct, idxFrom * nStream, idxTo * nStream + nStream - 1);

            switch(aggType)
            {
                case AGG_SUM:
                    if(firstConn == NULL)
                    {
                        engine->MatAdd(srcSub, stateSub, stateSub, *stream);
                    }
                    else
                    {
                        Matrix<FLOAT> firstSrcSub(firstConn->dstAct, idxFrom * nStream, idxTo * nStream + nStream - 1);
                        engine->MatAdd(srcSub, firstSrcSub, stateSub, *stream);
                        firstConn = NULL;
                    }
                    break;

                case AGG_MULT:
                    if(firstConn == NULL)
                    {
                        engine->MatElemMult(srcSub, stateSub, stateSub, *stream);
                    }
                    else
                    {
                        Matrix<FLOAT> firstSrcSub(firstConn->dstAct, idxFrom * nStream, idxTo * nStream + nStream - 1);
                        engine->MatElemMult(srcSub, firstSrcSub, stateSub, *stream);
                        firstConn = NULL;
                    }
                    break;

                default:
                    verify(false);
            }
        }
    }

    for(iter = srcList.begin(); iter != iter_end; ++iter)
    {
        //(*iter)->StreamWaitEvent(*stream);
        if((*iter)->param.dstRangeFrom != 0 || (*iter)->param.dstRangeTo != (long) GetSize() - 1)
        {
            if(isFirst == true)
            {
                switch(aggType)
                {
                    case AGG_SUM:
                        engine->MatSet(stateSub, (FLOAT) 0, *stream);
                        break;

                    case AGG_MULT:
                        engine->MatSet(stateSub, (FLOAT) 1, *stream);
                        break;

                    default:
                        verify(false);
                }

                isFirst = false;
            }
            else if(firstConn != NULL)
            {
                Matrix<FLOAT> firstSrcSub(firstConn->dstAct, idxFrom * nStream, idxTo * nStream + nStream - 1);
                engine->MatCopy(firstSrcSub, stateSub, *stream);
                firstConn = NULL;
            }

            Matrix<FLOAT> statePartialSub(state, (*iter)->param.dstRangeFrom, (*iter)->param.dstRangeTo,
                    idxFrom * nStream, idxTo * nStream + nStream - 1);

            Matrix<FLOAT> srcSub((*iter)->dstAct, idxFrom * nStream, idxTo * nStream + nStream - 1);
            engine->MatAdd(srcSub, statePartialSub, statePartialSub, *stream);
        }
    }

    if(firstConn != NULL)
    {
        //Matrix<FLOAT> firstSrcSub(firstConn->dstAct, idxFrom * nStream, idxTo * nStream + nStream - 1);
        //engine->MatCopy(firstSrcSub, stateSub, *stream);
        state.Link(firstConn->dstAct);
    }
}


void Layer::UpdateDstErr(const unsigned long idxFrom, const unsigned long idxTo)
{
    verify(idxFrom >= 0 && idxTo < nUnroll && idxFrom <= idxTo);
    verify(engine != NULL && stream != NULL);

    ConnList::const_iterator iter, iter_end;
    Matrix<FLOAT> dstErrSub(dstErr, idxFrom * nStream, idxTo * nStream + nStream - 1);
    Connection *firstConn = NULL;
    bool isFirst = true;
    bool errInject;

    TrainableProbe *trainableProbe = dynamic_cast<TrainableProbe *>(linkedProbe);

    errInject = (IsLinked() == true && trainableProbe != NULL && trainableProbe->InjectToSrcErr() == false);

    //verify(dstList.empty() == false);

    if(dstList.empty() == true)
    {
        if(errInject == true)
        {
            //trainableProbe->StreamWaitEvent(*stream);

            Matrix<FLOAT> probeErrSub(trainableProbe->GetErr(), idxFrom * nStream, idxTo * nStream + nStream - 1);

            engine->MatCopy(probeErrSub, dstErrSub, *stream);
        }
        else
        {
            engine->MatSet(dstErrSub, (FLOAT) 0, *stream);
        }
        return;
    }

    iter_end = dstList.end();
    for(iter = dstList.begin(); iter != iter_end; ++iter)
    {
        //(*iter)->StreamWaitEvent(*stream);
        if((*iter)->param.srcRangeFrom != 0 || (*iter)->param.srcRangeTo != (long) GetSize() - 1)
        {
            continue;
        }

        if(isFirst == true)
        {
            firstConn = (*iter);
            isFirst = false;
        }
        else
        {
            Matrix<FLOAT> dstSub((*iter)->srcErr, idxFrom * nStream, idxTo * nStream + nStream - 1);

            if(firstConn == NULL)
            {
                engine->MatAdd(dstSub, dstErrSub, dstErrSub, *stream);
            }
            else
            {
                Matrix<FLOAT> firstDstSub(firstConn->srcErr, idxFrom * nStream, idxTo * nStream + nStream - 1);
                engine->MatAdd(dstSub, firstDstSub, dstErrSub, *stream);
                firstConn = NULL;
            }
        }
    }

    for(iter = dstList.begin(); iter != iter_end; ++iter)
    {
        //(*iter)->StreamWaitEvent(*stream);
        if((*iter)->param.srcRangeFrom != 0 || (*iter)->param.srcRangeTo != (long) GetSize() - 1)
        {
            if(isFirst == true)
            {
                engine->MatSet(dstErrSub, (FLOAT) 0, *stream);
                isFirst = false;
            }
            else if(firstConn != NULL)
            {
                Matrix<FLOAT> firstDstSub(firstConn->srcErr, idxFrom * nStream, idxTo * nStream + nStream - 1);
                engine->MatCopy(firstDstSub, dstErrSub, *stream);
                firstConn = NULL;
            }

            Matrix<FLOAT> dstErrPartialSub(dstErr, (*iter)->param.srcRangeFrom, (*iter)->param.srcRangeTo,
                    idxFrom * nStream, idxTo * nStream + nStream - 1);

            Matrix<FLOAT> dstSub((*iter)->srcErr, idxFrom * nStream, idxTo * nStream + nStream - 1);
            engine->MatAdd(dstSub, dstErrPartialSub, dstErrPartialSub, *stream);
        }
    }

    if(firstConn != NULL)
    {
        if(errInject == true)
        {
            dstErr.Unlink();
            //trainableProbe->StreamWaitEvent(*stream);

            Matrix<FLOAT> probeErrSub(trainableProbe->GetErr(), idxFrom * nStream, idxTo * nStream + nStream - 1);
            Matrix<FLOAT> firstDstSub(firstConn->srcErr, idxFrom * nStream, idxTo * nStream + nStream - 1);

            engine->MatAdd(probeErrSub, firstDstSub, dstErrSub, *stream);
        }
        else
        { 
            dstErr.Link(firstConn->srcErr);
        }
    }
    else if(errInject == true)
    {
        //trainableProbe->StreamWaitEvent(*stream);

        Matrix<FLOAT> probeErrSub(trainableProbe->GetErr(), idxFrom * nStream, idxTo * nStream + nStream - 1);

        engine->MatAdd(probeErrSub, dstErrSub, dstErrSub, *stream);
    }
}


void Layer::UpdateSrcErr(const unsigned long idxFrom, const unsigned long idxTo)
{
    verify(idxFrom >= 0 && idxTo < nUnroll && idxFrom <= idxTo);
    verify(engine != NULL && stream != NULL);

    Matrix<FLOAT> dstErrSub(dstErr, idxFrom * nStream, idxTo * nStream + nStream - 1);

    bool errInject;

    TrainableProbe *trainableProbe = dynamic_cast<TrainableProbe *>(linkedProbe);

    errInject = (IsLinked() == true && trainableProbe != NULL && trainableProbe->InjectToSrcErr() == true);



    if(actType == ACT_LINEAR || (actType == ACT_DROPOUT && dropoutEnabled == false))
    {
        srcErr.Link(dstErr);
    }
    else
    {
        srcErr.Unlink();

        Matrix<FLOAT> srcErrSub(srcErr, idxFrom * nStream, idxTo * nStream + nStream - 1);

        /* Multiply the derivative of the activation, which is stored in srcErr */
        engine->MatElemMult(dstErrSub, srcErrSub, srcErrSub, *stream);
    }

    if(errInject == true)
    {
        //trainableProbe->StreamWaitEvent(*stream);

        Matrix<FLOAT> probeErrSub(trainableProbe->GetErr(), idxFrom * nStream, idxTo * nStream + nStream - 1);
        Matrix<FLOAT> srcErrSub(srcErr, idxFrom * nStream, idxTo * nStream + nStream - 1);

        /* dstErr might be affected if srcErr is linked to dstErr */
        engine->MatAdd(probeErrSub, srcErrSub, srcErrSub, *stream);
    }
}


void Layer::DistributeErr(Connection *conn, const unsigned long idxFrom, const unsigned long idxTo)
{
    verify(idxFrom >= 0 && idxTo < nUnroll && idxFrom <= idxTo);
    verify(engine != NULL && stream != NULL);

    Matrix<FLOAT> srcErrSub(srcErr, conn->param.dstRangeFrom, conn->param.dstRangeTo,
            0, srcErr.GetNumCols() - 1);

    switch(aggType)
    {
        case AGG_SUM:
            conn->dstErr.Link(srcErrSub);
            break;

        case AGG_MULT:
            {
                ConnList::const_iterator iter, iter_end;

                Matrix<FLOAT> srcErrPartialSub(srcErr, idxFrom * nStream, idxTo * nStream + nStream - 1);
                Matrix<FLOAT> connErrSub(conn->dstErr, idxFrom * nStream, idxTo * nStream + nStream - 1);

                bool isFirst = true;

                iter_end = srcList.end();
                for(iter = srcList.begin(); iter != iter_end; ++iter)
                {
                    if(*iter == conn) continue;

                    long rangeFrom = conn->param.dstRangeFrom - (*iter)->param.dstRangeFrom;
                    long rangeTo = conn->param.dstRangeTo - (*iter)->param.dstRangeFrom;

                    if(rangeTo < 0 || rangeFrom >= (long) ((*iter)->dstSize)) continue;

                    verify(rangeFrom >= 0);
                    verify(rangeTo < (long) ((*iter)->dstSize));

                    Matrix<FLOAT> srcActSub((*iter)->dstAct, rangeFrom, rangeTo,
                            idxFrom * nStream, idxTo * nStream + nStream - 1);

                    if(isFirst == true)
                    {
                        engine->MatElemMult(srcErrPartialSub, srcActSub, connErrSub, *conn->stream);
                        isFirst = false;
                    }
                    else
                    {
                        engine->MatElemMult(connErrSub, srcActSub, connErrSub, *conn->stream);
                    }
                }

                if(isFirst == true)
                {
                    conn->dstErr.Link(srcErrSub);
                }
            }

            break;

        default:
            verify(false);
    }
}


void Layer::SetPStream(PStream *const stream)
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

    if(engine != NULL && IsLinked())
    {
        linkedProbe->SetPStream();
    }
}


PStream &Layer::GetPStream()
{
    verify(engine != NULL);
    return *stream;
}


void Layer::EventRecord()
{
    verify(engine != NULL);
    engine->EventRecord(event, *stream);
}


void Layer::StreamWaitEvent(PStream &stream)
{
    verify(engine != NULL);
    engine->StreamWaitEvent(stream, event);
}


void Layer::ForwardWait()
{
    ConnList::const_iterator iter, iter_end;

    if(IsLinked() == true && linkedProbe->IsInput() == true)
    {
        linkedProbe->StreamWaitEvent(*stream);
    }

    iter_end = srcList.end();
    for(iter = srcList.begin(); iter != iter_end; ++iter)
    {
        (*iter)->StreamWaitEvent(*stream);
    }
}


void Layer::BackwardWait()
{
    ConnList::const_iterator iter, iter_end;
    bool errInject;

    TrainableProbe *trainableProbe = dynamic_cast<TrainableProbe *>(linkedProbe);
    errInject = (IsLinked() == true && trainableProbe != NULL);

    if(errInject == true)
    {
        linkedProbe->StreamWaitEvent(*stream);
    }

    iter_end = dstList.end();
    for(iter = dstList.begin(); iter != iter_end; ++iter)
    {
        (*iter)->StreamWaitEvent(*stream);
    }
}

}

