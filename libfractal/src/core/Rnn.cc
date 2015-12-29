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




#include "Rnn.h"

#include <sys/stat.h>
#include <sstream>
#include <iostream>


#define MAX_NUM_PSTREAM_PER_LOC 4

namespace fractal
{

/* Special indices for graph algorithms */
static const long UNTOUCHED = -1;
static const long TOUCHED = -2;
static const long SCC_DETERMINED = -3;


Rnn::Rnn()
{
    nStream = 0;
    nUnroll = 0;
    isReady = false;
    engine = NULL;
    defaultPStream = NULL;
}


Rnn::~Rnn()
{
    Clear();
}


void Rnn::SetEngine(Engine *engine)
{
    if(this->engine == engine) return;

    LayerMap::const_iterator layerIter, layerIter_end;
    ConnSet::const_iterator connIter, connIter_end;

    ClearPStreams();
    DestroyDefaultPStream();


    this->engine = engine;

    if(engine != NULL)
    {
        verify(engine->GetNumComputeLocs() > 0);
        computeLoc.resize(1);
        computeLoc[0] = engine->GetComputeLoc(0);

        CreateDefaultPStream(computeLoc[0]);
    }
    else
    {
        computeLoc.resize(0);
    }

    layerIter_end = layerMap.end();
    for(layerIter = layerMap.begin(); layerIter != layerIter_end; ++layerIter)
    {
        layerIter->second->SetEngine(engine);
        layerIter->second->SetPStream(defaultPStream);
    }

    connIter_end = connSet.end();
    for(connIter = connSet.begin(); connIter != connIter_end; ++connIter)
    {
        (*connIter)->SetEngine(engine);
        (*connIter)->SetPStream(defaultPStream);
    }

    isReady = false;
}


Engine *Rnn::GetEngine() const
{
    return engine;
}


void Rnn::AddLayer(const std::string &name, ActType actType, AggType aggType, const unsigned long size, const LayerParam &param)
{
    Layer *layer;

    verify(FindLayer(name) == NULL);

    layer = new Layer(name, actType, aggType, size, param);
    layer->SetBatchSize(nStream, nUnroll);
    layer->SetEngine(engine);
    layer->SetPStream(defaultPStream);
    layerMap.insert(std::pair<std::string, Layer *>(name, layer));

    isReady = false;
}


void Rnn::AddConnection(const std::string &from, const std::string &to, const ConnParam &param)
{
    Layer *layerFrom, *layerTo;

    layerFrom = FindLayer(from);
    layerTo = FindLayer(to);

    verify(layerFrom != NULL);
    verify(layerTo != NULL);

    AddConnection(layerFrom, layerTo, param);

    isReady = false;
}


void Rnn::DeleteLayer(const std::string &name)
{
    Layer *layer;
    LayerMap::const_iterator layerIter;
    Layer::ConnList::const_iterator connIter, connIter_end;

    layerIter = layerMap.find(name);
    verify(layerIter != layerMap.end());

    layer = layerIter->second;

    connIter_end = layer->GetSrcConnections().end();
    for(connIter = layer->GetSrcConnections().begin(); connIter != connIter_end; ++connIter)
    {
        (*connIter)->GetSrcLayer()->UnlinkMatrices();
        (*connIter)->GetSrcLayer()->RemoveDstConnection(*connIter);

        verify(connSet.erase(*connIter) == 1);
        delete *connIter;
    }

    connIter_end = layer->GetDstConnections().end();
    for(connIter = layer->GetDstConnections().begin(); connIter != connIter_end; ++connIter)
    {
        (*connIter)->GetDstLayer()->UnlinkMatrices();
        (*connIter)->GetDstLayer()->RemoveSrcConnection(*connIter);

        verify(connSet.erase(*connIter) == 1);
        delete *connIter;
    }

    layerMap.erase(layerIter);
    delete layer;

    isReady = false;
}


void Rnn::DeleteConnection(const std::string &from, const std::string &to)
{
    Layer *layerFrom, *layerTo;

    layerFrom = FindLayer(from);
    layerTo = FindLayer(to);

    verify(layerFrom != NULL);
    verify(layerTo != NULL);

    DeleteConnection(layerFrom, layerTo);

    isReady = false;
}


void Rnn::LinkProbe(Probe &probe, const std::string &layerName)
{
    Layer *layer;

    layer = FindLayer(layerName);

    verify(layer != NULL);

    LinkProbe(probe, layer);
}


void Rnn::SetBatchSize(const unsigned long nStream, const unsigned long nUnroll)
{
    LayerMap::const_iterator layerIter, layerIter_end;
    ConnSet::const_iterator connIter, connIter_end;

    /* TODO: Reset streams for multi GPUs? */

    this->nStream = nStream;
    this->nUnroll = nUnroll;

    layerIter_end = layerMap.end();
    for(layerIter = layerMap.begin(); layerIter != layerIter_end; ++layerIter)
    {
        layerIter->second->SetBatchSize(nStream, nUnroll);
    }

    connIter_end = connSet.end();
    for(connIter = connSet.begin(); connIter != connIter_end; ++connIter)
    {
        (*connIter)->SetBatchSize(nStream, nUnroll);
    }
}


const unsigned long Rnn::GetNumStreams() const
{
    return nStream;
}


const unsigned long Rnn::GetNumUnrollSteps() const
{
    return nUnroll;
}


void Rnn::InitForward(const unsigned long idxFrom, const unsigned long idxTo)
{
    verify(idxFrom >= 0 && idxTo < nUnroll && idxFrom <= idxTo);

    Ready();

    ConnSet::const_iterator iter, iter_end;

    iter_end = connSet.end();
    for(iter = connSet.begin(); iter != iter_end; ++iter)
    {
        // Since some layers have internal memory, initialize all layers.
        //if((*iter)->IsDelayed() == true)
        //{
        (*iter)->GetSrcLayer()->InitAct(idxFrom, idxTo);
        (*iter)->GetSrcLayer()->EventRecord();
        //}
    }
}


void Rnn::InitBackward(const unsigned long idxFrom, const unsigned long idxTo)
{
    verify(idxFrom >= 0 && idxTo < nUnroll && idxFrom <= idxTo);

    Ready();

    ConnSet::const_iterator iter, iter_end;

    iter_end = connSet.end();
    for(iter = connSet.begin(); iter != iter_end; ++iter)
    {
        if((*iter)->IsDelayed() == true)
        {
            (*iter)->InitErr(idxFrom, idxTo);
            (*iter)->EventRecord();
        }
    }
}


void Rnn::InitWeights(const InitWeightParam &param)
{
    verify(param.IsValid() == true);
    verify(engine != NULL);

    ConnSet::const_iterator iter, iter_end;

    iter_end = connSet.end();
    for(iter = connSet.begin(); iter != iter_end; ++iter)
    {
        (*iter)->InitWeights(param);
        engine->StreamSynchronize((*iter)->GetPStream());
    }
}


void Rnn::InitNesterov()
{
    ConnSet::const_iterator iter, iter_end;

    iter_end = connSet.end();
    for(iter = connSet.begin(); iter != iter_end; ++iter)
    {
        (*iter)->InitNesterov();
    }
}


void Rnn::InitAdadelta(const FLOAT decayRate, const bool initNumerator)
{
    verify(decayRate > (FLOAT) 0);

    ConnSet::const_iterator iter, iter_end;

    iter_end = connSet.end();
    for(iter = connSet.begin(); iter != iter_end; ++iter)
    {
        (*iter)->InitAdadelta(decayRate, initNumerator);
    }
}


void Rnn::InitRmsprop(const FLOAT decayRate)
{
    verify(decayRate > (FLOAT) 0);

    ConnSet::const_iterator iter, iter_end;

    iter_end = connSet.end();
    for(iter = connSet.begin(); iter != iter_end; ++iter)
    {
        (*iter)->InitRmsprop(decayRate);
    }
}


void Rnn::Forward(const unsigned long idxFrom, const unsigned long idxTo)
{
    verify(idxFrom >= 0 && idxTo < nUnroll && idxFrom <= idxTo);
    verify(nStream > 0);

    unsigned long i;
    long group;
    bool loopDetected;

    SccList::const_iterator sccIter, sccIter_end;
    Scc::const_iterator layerIter, layerIter_end;
    Layer::ConnList::const_iterator connIter, connIter_end;

    Scc *scc;
    Layer *layer;
    Connection *conn;

    verify(engine != NULL);

    Ready();

    sccIter_end = sccList.end();
    for(sccIter = sccList.begin(); sccIter != sccIter_end; ++sccIter)
    {
        /* Inter-SCC parallel propagation */

        scc = *sccIter;

        layerIter_end = scc->end();
        for(layerIter = scc->begin(); layerIter != layerIter_end; ++layerIter)
        {
            layer = *layerIter;
            group = layer->GetGroup();

            connIter_end = layer->GetSrcConnections().end();
            for(connIter = layer->GetSrcConnections().begin(); connIter != connIter_end; ++connIter)
            {
                conn = *connIter;
                if(conn->GetSrcLayer()->GetGroup() == group) continue;

                verify(conn->IsDelayed() == false); /* For now, this may cause a problem */

                /* Inter-SCC connection */
                conn->ForwardWait();
                conn->Forward(idxFrom, idxTo);
                conn->EventRecord();
            }
        }

        /* Intra-SCC propagation */

        /* Check if there are loops inside the scc */
        if(scc->size() == 1)
        {
            loopDetected = false;
            layer = scc->front();
            group = layer->GetGroup();

            connIter_end = layer->GetSrcConnections().end();
            for(connIter = layer->GetSrcConnections().begin(); connIter != connIter_end; ++connIter)
            {
                conn = *connIter;
                if(conn->GetSrcLayer()->GetGroup() == group)
                {
                    /* Intra-SCC connection */
                    loopDetected = true;
                    break;
                }
            }
        }
        else
        {
            loopDetected = true;
        }

        if(loopDetected == true)
        {
            /* Sequential propagation */
            for(i = idxFrom; i <= idxTo; i++)
            {
                /* Delayed connection */
                layerIter_end = scc->end();
                for(layerIter = scc->begin(); layerIter != layerIter_end; ++layerIter)
                {
                    layer = *layerIter;
                    group = layer->GetGroup();

                    connIter_end = layer->GetSrcConnections().end();
                    for(connIter = layer->GetSrcConnections().begin(); connIter != connIter_end; ++connIter)
                    {
                        conn = *connIter;
                        if(conn->GetSrcLayer()->GetGroup() != group) continue;
                        if(conn->IsDelayed() == false) continue;

                        /* Intra-SCC connection */
                        if(i == idxFrom) conn->ForwardWait();
                        conn->Forward(i, i);
                        if(i == idxTo) conn->EventRecord();
                    }
                }

                /* Non-delayed connection */
                layerIter_end = scc->end();
                for(layerIter = scc->begin(); layerIter != layerIter_end; ++layerIter)
                {
                    layer = *layerIter;
                    group = layer->GetGroup();

                    connIter_end = layer->GetSrcConnections().end();
                    for(connIter = layer->GetSrcConnections().begin(); connIter != connIter_end; ++connIter)
                    {
                        conn = *connIter;
                        if(conn->GetSrcLayer()->GetGroup() != group) continue;
                        if(conn->IsDelayed() == true) continue;

                        /* Intra-SCC connection */
                        //if(i == idxFrom) conn->ForwardWait();
                        conn->Forward(i, i);
                        //if(i == idxTo) conn->EventRecord();
                    }

                    /* Intra-SCC layer activation */
                    if(i == idxFrom) layer->ForwardWait();
                    layer->Forward(i, i);
                    if(i == idxTo) layer->EventRecord();
                }
            }
        }
        else /* Loop not detected */
        {
            /* Parallel activation */
            layer->ForwardWait();
            layer->Forward(idxFrom, idxTo);
            layer->EventRecord();
        }
    }
}


void Rnn::Backward(const unsigned long idxFrom, const unsigned long idxTo)
{
    verify(idxFrom >= 0 && idxTo < nUnroll && idxFrom <= idxTo);
    verify(nStream > 0);

    long i;
    long group;
    bool loopDetected;

    SccList::const_reverse_iterator sccIter, sccIter_end;
    Scc::const_reverse_iterator layerIter, layerIter_end;
    Layer::ConnList::const_iterator connIter, connIter_end;

    Scc *scc;
    Layer *layer;
    Connection *conn;

    verify(engine != NULL);

    Ready();

    sccIter_end = sccList.rend();
    for(sccIter = sccList.rbegin(); sccIter != sccIter_end; ++sccIter)
    {
        /* Inter-SCC parallel propagation */

        scc = *sccIter;

        layerIter_end = scc->rend();
        for(layerIter = scc->rbegin(); layerIter != layerIter_end; ++layerIter)
        {
            layer = *layerIter;
            group = layer->GetGroup();

            connIter_end = layer->GetDstConnections().end();
            for(connIter = layer->GetDstConnections().begin(); connIter != connIter_end; ++connIter)
            {
                conn = *connIter;
                if(conn->GetDstLayer()->GetGroup() == group) continue;

                verify(conn->IsDelayed() == false); /* For now, this may cause a problem */

                /* Inter-SCC connection */
                conn->BackwardWait();
                conn->UpdateDstErr(idxFrom, idxTo);
                //if(layer->GetSrcConnections().empty() == false && layer->IsLinked() == false)
                if(layer->GetSrcConnections().empty() == false)
                    conn->Backward(idxFrom, idxTo);
                conn->EventRecord();
            }
        }

        /* Intra-SCC propagation */

        /* Check if there are loops inside the scc */
        if(scc->size() == 1)
        {
            loopDetected = false;
            layer = scc->front();
            group = layer->GetGroup();

            connIter_end = layer->GetDstConnections().end();
            for(connIter = layer->GetDstConnections().begin(); connIter != connIter_end; ++connIter)
            {
                conn = *connIter;
                if(conn->GetDstLayer()->GetGroup() == group)
                {
                    /* Intra-SCC connection */
                    loopDetected = true;
                    break;
                }
            }
        }
        else
        {
            loopDetected = true;
        }

        if(loopDetected == true)
        {
            /* Sequential propagation */
            for(i = idxTo; i >= (long) idxFrom; i--)
            {
                /* Non-delayed connection */
                layerIter_end = scc->rend();
                for(layerIter = scc->rbegin(); layerIter != layerIter_end; ++layerIter)
                {
                    layer = *layerIter;
                    if(layer->GetSrcConnections().empty() == true) continue;
                    group = layer->GetGroup();

                    /* Intra-SCC layer activation */
                    //if(layer->IsLinked() == false)
                    if(i == (long) idxTo) layer->BackwardWait();
                    layer->Backward(i, i);
                    if(i == (long) idxFrom) layer->EventRecord();

                    connIter_end = layer->GetSrcConnections().end();
                    for(connIter = layer->GetSrcConnections().begin(); connIter != connIter_end; ++connIter)
                    {
                        conn = *connIter;
                        if(conn->GetSrcLayer()->GetGroup() != group) continue;
                        if(conn->IsDelayed() == true) continue;

                        /* Intra-SCC connection */
                        //if(i == idxTo) conn->BackwardWait();
                        conn->UpdateDstErr(i, i);
                        //if(conn->GetSrcLayer()->GetSrcConnections().empty() == false && conn->GetSrcLayer()->IsLinked() == false)
                        if(conn->GetSrcLayer()->GetSrcConnections().empty() == false)
                            conn->Backward(i, i);
                        //if(i == idxFrom) conn->EventRecord();
                    }
                }

                /* Delayed connection */
                layerIter_end = scc->rend();
                for(layerIter = scc->rbegin(); layerIter != layerIter_end; ++layerIter)
                {
                    layer = *layerIter;
                    group = layer->GetGroup();

                    connIter_end = layer->GetSrcConnections().end();
                    for(connIter = layer->GetSrcConnections().begin(); connIter != connIter_end; ++connIter)
                    {
                        conn = *connIter;
                        if(conn->GetSrcLayer()->GetGroup() != group) continue;
                        if(conn->IsDelayed() == false) continue;

                        /* Intra-SCC connection */
                        //if(i == idxTo) conn->BackwardWait();
                        conn->UpdateDstErr(i, i);
                        //if(conn->GetSrcLayer()->GetSrcConnections().empty() == false && conn->GetSrcLayer()->IsLinked() == false)
                        if(conn->GetSrcLayer()->GetSrcConnections().empty() == false)
                            conn->Backward(i, i);
                        if(i == (long) idxFrom) conn->EventRecord();
                    }
                }
            }
        }
        else /* Loop not detected */
        {
            /* Parallel activation */
            //if(layer->GetSrcConnections().empty() == false && layer->IsLinked() == false)
            if(layer->GetSrcConnections().empty() == false)
            {
                layer->BackwardWait();
                layer->Backward(idxFrom, idxTo);
                layer->EventRecord();
            }
        }
    }
}


void Rnn::CalcActDeriv(const unsigned long idxFrom, const unsigned long idxTo)
{
    LayerMap::const_iterator iter, iter_end;

    verify(isReady == true);
    verify(engine != NULL);

    iter_end = layerMap.end();
    for(iter = layerMap.begin(); iter != iter_end; ++iter)
    {
        //if(iter->second->GetSrcConnections().empty() == false && iter->second->IsLinked() == false)
        /* TODO: remove unnecessary computations */
        if(iter->second->GetSrcConnections().empty() == false)
            iter->second->CalcActDeriv(idxFrom, idxTo);
    }
}


void Rnn::UpdateWeights(const unsigned long idxFrom, const unsigned long idxTo,
        const FLOAT rate, const FLOAT momentum, const bool adadelta, const bool rmsprop)
{
    ConnSet::const_iterator iter, iter_end;

    verify(isReady == true);
    verify(engine != NULL);

    iter_end = connSet.end();
    for(iter = connSet.begin(); iter != iter_end; ++iter)
    {
        (*iter)->UpdateWeights(idxFrom, idxTo, rate, momentum, adadelta, rmsprop);
    }
}


void Rnn::EnableDropout(const bool enable)
{
    LayerMap::const_iterator iter, iter_end;

    iter_end = layerMap.end();
    for(iter = layerMap.begin(); iter != iter_end; ++iter)
    {
        iter->second->EnableDropout(enable);
    }
}


void Rnn::GenerateDropoutMask(const unsigned long idxFrom, const unsigned long idxTo)
{
    LayerMap::const_iterator iter, iter_end;

    verify(isReady == true);
    verify(engine != NULL);

    iter_end = layerMap.end();
    for(iter = layerMap.begin(); iter != iter_end; ++iter)
    {
        iter->second->GenerateDropoutMask(idxFrom, idxTo);
    }
}


void Rnn::ProcessWeights(const FLOAT noise)
{
    ConnSet::const_iterator iter, iter_end;

    verify(isReady == true);
    verify(engine != NULL);

    iter_end = connSet.end();
    for(iter = connSet.begin(); iter != iter_end; ++iter)
    {
        (*iter)->ProcessWeights(noise);
    }
}


void Rnn::Ready()
{
    if(isReady == true) return;

    verify(layerMap.empty() == false);
    verify(engine != NULL);


    LayerMap::const_iterator layerIter, layerIter_end;
    ConnSet::const_iterator connIter, connIter_end;


    layerIter_end = layerMap.end();
    for(layerIter = layerMap.begin(); layerIter != layerIter_end; ++layerIter)
    {
        layerIter->second->UnlinkMatrices();
    }

    connIter_end = connSet.end();
    for(connIter = connSet.begin(); connIter != connIter_end; ++connIter)
    {
        (*connIter)->UnlinkMatrices();
    }


    Tarjan();
    ClearPStreams();
    CreatePStreams();

    isReady = true;
}


void Rnn::PrintNetwork(std::ostream &outStream)
{
    unsigned long sccIdx;
    long group;
    bool loopDetected;

    SccList::const_iterator sccIter, sccIter_end;
    Scc::const_iterator layerIter, layerIter_end;
    Layer::ConnList::const_iterator connIter, connIter_end;

    Scc *scc;
    Layer *layer;
    Connection *conn;

    sccIdx = 0;

    Ready();

    sccIter_end = sccList.end();
    for(sccIter = sccList.begin(); sccIter != sccIter_end; ++sccIter)
    {
        /* Inter-SCC parallel propagation */

        sccIdx++;
        outStream << "SCC " << sccIdx;

        scc = *sccIter;

        /* Check if there are loops inside the scc */
        if(scc->size() == 1)
        {
            loopDetected = false;
            outStream << ": ";
            layer = scc->front();
            group = layer->GetGroup();

            connIter_end = layer->GetSrcConnections().end();
            for(connIter = layer->GetSrcConnections().begin(); connIter != connIter_end; ++connIter)
            {
                conn = *connIter;
                if(conn->GetSrcLayer()->GetGroup() == group)
                {
                    /* Intra-SCC connection */
                    outStream << " (loop, 1 layer):" << std::endl;
                    loopDetected = true;
                    break;
                }
            }
        }
        else
        {
            outStream << " (loop, " << scc->size() << " layers):" << std::endl;
            loopDetected = true;
        }


        layerIter_end = scc->end();
        for(layerIter = scc->begin(); layerIter != layerIter_end; ++layerIter)
        {
            layer = *layerIter;
            group = layer->GetGroup();

            connIter_end = layer->GetSrcConnections().end();
            for(connIter = layer->GetSrcConnections().begin(); connIter != connIter_end; ++connIter)
            {
                conn = *connIter;
                if(conn->GetSrcLayer()->GetGroup() == group) continue;


                /* Inter-SCC connection */
                //outStream << "    Inter-SCC connections:" << std::endl;
            }
        }

        /* Intra-SCC propagation */
        if(loopDetected == true)
        {
            /* Sequential propagation */
            {
                /* Delayed connection */
                layerIter_end = scc->end();
                for(layerIter = scc->begin(); layerIter != layerIter_end; ++layerIter)
                {
                    layer = *layerIter;
                    group = layer->GetGroup();

                    connIter_end = layer->GetSrcConnections().end();
                    for(connIter = layer->GetSrcConnections().begin(); connIter != connIter_end; ++connIter)
                    {
                        conn = *connIter;
                        if(conn->GetSrcLayer()->GetGroup() != group) continue;
                        if(conn->IsDelayed() == false) continue;

                        /* Intra-SCC connection */
                        //conn->Forward(i, i + nStream - 1, nStream);
                    }
                }

                /* Non-delayed connection */
                layerIter_end = scc->end();
                for(layerIter = scc->begin(); layerIter != layerIter_end; ++layerIter)
                {
                    layer = *layerIter;
                    group = layer->GetGroup();

                    connIter_end = layer->GetSrcConnections().end();
                    for(connIter = layer->GetSrcConnections().begin(); connIter != connIter_end; ++connIter)
                    {
                        conn = *connIter;
                        if(conn->GetSrcLayer()->GetGroup() != group) continue;
                        if(conn->IsDelayed() == true) continue;

                        /* Intra-SCC connection */
                        //conn->Forward(i, i + nStream - 1, nStream);
                    }

                    /* Intra-SCC layer activation */
                    outStream << "    " << layer->GetName() << std::endl;
                }
            }
        }
        else /* Loop not detected */
        {
            /* Parallel activation */
            outStream << layer->GetName() << std::endl;
        }
    }

}


void Rnn::Tarjan()
{
    /* Tarjan's Algorithm (non-recursive) */
    /* Find strongly connected components and perform topological sort */

    long index, group;
    Layer *v, *w;
    std::stack<Layer *> dfsStack, sccStack;
    LayerMap::const_iterator layerIter, layerIter_end;
    Layer::ConnList::const_iterator connIter, connIter_end;


    ClearSccList();

    /* Initialize flags */

    layerIter_end = layerMap.end();
    for(layerIter = layerMap.begin(); layerIter != layerIter_end; ++layerIter)
    {
        layerIter->second->SetVisited(false);
        layerIter->second->SetIndex(UNTOUCHED);
    }


    index = 0;
    group = 0;

    layerIter_end = layerMap.end();
    for(layerIter = layerMap.begin(); layerIter != layerIter_end; ++layerIter)
    {
        if(layerIter->second->GetVisited() == true) continue;

        /* Strong connect: depth first search */

        dfsStack.push(layerIter->second);

        while(dfsStack.empty() == false)
        {
            v = dfsStack.top();

            if(v->GetIndex() == UNTOUCHED)
            {
                sccStack.push(v);
                v->SetIndex(index);
                v->SetGroup(index);
                index++;

                const Layer::ConnList srcConnList = v->GetSrcConnections();

                connIter_end = srcConnList.end();
                for(connIter = srcConnList.begin(); connIter != connIter_end; ++connIter)
                {
                    w = (*connIter)->GetSrcLayer();

                    if(w->GetIndex() == UNTOUCHED)
                        dfsStack.push(w);
                }

                continue;
            }

            dfsStack.pop();
            if(v->GetVisited() == true) continue;

            /* Post visit */

            const Layer::ConnList srcConnList = v->GetSrcConnections();

            connIter_end = srcConnList.end();
            for(connIter = srcConnList.begin(); connIter != connIter_end; ++connIter)
            {
                w = (*connIter)->GetSrcLayer();
                if(w->GetIndex() == SCC_DETERMINED) continue;

                v->SetGroup(std::min(v->GetGroup(), w->GetGroup()));
            }

            /* Is v root? Create new SCC */
            if(v->GetIndex() == v->GetGroup())
            {
                sccList.push_back(CreateScc(sccStack, v, group));
                group++;
            }

            v->SetVisited(true);
        }
    }
}


Rnn::Scc *const Rnn::CreateScc(std::stack<Layer *> &sccStack, const Layer *const root, const long group)
{
    /* Create SCC and perform topological sort */
    /* Memory allocation (scc) */

    Scc *scc;
    Layer *v, *w;
    std::stack<Layer *> dfsStack;
    LayerList sccLayerList;
    LayerList::const_iterator layerIter, layerIter_end;
    Layer::ConnList::const_iterator connIter, connIter_end;


    scc = new Scc;

    do
    {
        v = sccStack.top();
        sccStack.pop();

        v->SetIndex(UNTOUCHED);
        v->SetGroup(group);
        v->SetVisited(false);

        sccLayerList.push_front(v);
    } while(v != root);


    layerIter_end = sccLayerList.end();
    for(layerIter = sccLayerList.begin(); layerIter != layerIter_end; ++layerIter)
    {
        if((*layerIter)->GetVisited() == true) continue;


        /* Depth first search */

        dfsStack.push(*layerIter);

        while(dfsStack.empty() == false)
        {
            v = dfsStack.top();

            if(v->GetIndex() == UNTOUCHED)
            {
                v->SetIndex(TOUCHED);

                const Layer::ConnList srcConnList = v->GetSrcConnections();

                connIter_end = srcConnList.end();
                for(connIter = srcConnList.begin(); connIter != connIter_end; ++connIter)
                {
                    if((*connIter)->IsDelayed() == true) continue;

                    w = (*connIter)->GetSrcLayer();

                    if(w->GetIndex() == UNTOUCHED)
                        dfsStack.push(w);
                    else if(w->GetIndex() == TOUCHED)
                    {  /* Loop detected */
                        std::cerr << std::endl << "Loop detected !!" << std::endl;
                        verify(false);
                    }
                }

                continue;
            }

            dfsStack.pop();
            if(v->GetVisited() == true) continue;

            v->SetIndex(SCC_DETERMINED);

            /* Post visit */

            scc->push_back(v);

            v->SetVisited(true);
        }
    }


    return scc;
}


#if 0
void Rnn::CreatePStreams(unsigned long loc)
{
    unsigned long i, j;
    long group;

    SccList::const_iterator sccIter, sccIter_end;
    Scc::const_iterator layerIter, layerIter_end;
    Layer::ConnList::const_iterator connIter, connIter_end;

    Scc *scc;
    Layer *layer;
    Connection *conn;
    PStream *pStream[16];

    for(i = 0; i < 16; i++)
    {
        pStream[i] = new PStream();
        engine->StreamCreate(*pStream[i], loc);
        pStreamList.push_back(pStream[i]);
    }

    sccIter_end = sccList.end();
    for(sccIter = sccList.begin(); sccIter != sccIter_end; ++sccIter)
    {
        scc = *sccIter;

        layerIter_end = scc->end();
        for(layerIter = scc->begin(); layerIter != layerIter_end; ++layerIter)
        {
            layer = *layerIter;
            group = layer->GetGroup();

            i = j = 0;
            layer->SetPStream(pStream[i]);
            printf("%s : %ld\n", layer->GetName().c_str(), i);

            connIter_end = layer->GetDstConnections().end();
            for(connIter = layer->GetDstConnections().begin(); connIter != connIter_end; ++connIter)
            {
                conn = *connIter;
                if(conn->GetDstLayer()->GetGroup() == group)
                {
                    /* Intra-scc connection */

                    conn->SetPStream(pStream[i]);
                    printf("%s -> %s : %ld\n", conn->GetSrcLayer()->GetName().c_str(), conn->GetDstLayer()->GetName().c_str(), i);

                    i = (i + 1) % 16;
                }
                else
                {
                    /* Inter-scc connection */

                    conn->SetPStream(pStream[j]);
                    printf("%s -> %s : %ld\n", conn->GetSrcLayer()->GetName().c_str(), conn->GetDstLayer()->GetName().c_str(), j);

                    j = (j + 1) % 16;
                }
                conn = *connIter;

            }
        }
    }
}
#endif


#if 0
void Rnn::CreatePStreams(unsigned long loc)
{
    long group;

    SccList::const_iterator sccIter, sccIter_end;
    Scc::const_iterator layerIter, layerIter_end;
    Layer::ConnList::const_iterator connIter, connIter_end;

    Scc *scc;
    Layer *layer;
    Connection *conn;
    PStream *interPStream, *intraPStream;


    sccIter_end = sccList.end();
    for(sccIter = sccList.begin(); sccIter != sccIter_end; ++sccIter)
    {
        scc = *sccIter;

        intraPStream = new PStream();
        engine->StreamCreate(*intraPStream, loc);
        pStreamList.push_back(intraPStream);

        layerIter_end = scc->end();
        for(layerIter = scc->begin(); layerIter != layerIter_end; ++layerIter)
        {
            layer = *layerIter;
            group = layer->GetGroup();

            layer->SetPStream(intraPStream);
            //printf("\n%p: %s\n", intraPStream->cudaStream, layer->GetName().c_str());

            connIter_end = layer->GetSrcConnections().end();
            for(connIter = layer->GetSrcConnections().begin(); connIter != connIter_end; ++connIter)
            {
                conn = *connIter;
                if(conn->GetSrcLayer()->GetGroup() == group)
                {
                    /* Intra-scc connection */

                    conn->SetPStream(intraPStream);
                    //printf("%s -> %s\n", conn->GetSrcLayer()->GetName().c_str(), conn->GetDstLayer()->GetName().c_str());
                }
                else
                {
                    /* Inter-scc connection */

                    interPStream = new PStream();
                    engine->StreamCreate(*interPStream, loc);
                    pStreamList.push_back(interPStream);

                    conn->SetPStream(interPStream);
                    //printf("%s -> %s\n", conn->GetSrcLayer()->GetName().c_str(), conn->GetDstLayer()->GetName().c_str());
                }
            }
        }
    }
}
#endif


#if 1
void Rnn::CreatePStreams()
{
    bool loopDetected;
    long group;
    long nLoop, loopIdx;

    SccList::const_iterator sccIter, sccIter_end;
    Scc::const_iterator layerIter, layerIter_end;
    Layer::ConnList::const_iterator connIter, connIter_end;

    Scc *scc;
    Layer *layer;
    Connection *conn;
    PStream *pStream;

    /* Count the number of loops */
    nLoop = 0;
    sccIter_end = sccList.end();
    for(sccIter = sccList.begin(); sccIter != sccIter_end; ++sccIter)
    {
        scc = *sccIter;
        /* Check if there are loops inside the scc */
        if(scc->size() == 1)
        {
            loopDetected = false;
            layer = scc->front();
            group = layer->GetGroup();

            connIter_end = layer->GetSrcConnections().end();
            for(connIter = layer->GetSrcConnections().begin(); connIter != connIter_end; ++connIter)
            {
                conn = *connIter;
                if(conn->GetSrcLayer()->GetGroup() == group)
                {
                    loopDetected = true;
                    break;
                }
            }
        }
        else
        {
            loopDetected = true;
        }

        if(loopDetected == true)
        {
            nLoop++;
        }
    }

    /* Assign PStreams */
    loopIdx = 0;
    pStream = new PStream();
    engine->StreamCreate(*pStream, computeLoc[0]);
    pStreamList.push_back(pStream);

    sccIter_end = sccList.end();
    for(sccIter = sccList.begin(); sccIter != sccIter_end; ++sccIter)
    {
        scc = *sccIter;
        /* Check if there are loops inside the scc */
        if(scc->size() == 1)
        {
            loopDetected = false;
            layer = scc->front();
            group = layer->GetGroup();

            connIter_end = layer->GetSrcConnections().end();
            for(connIter = layer->GetSrcConnections().begin(); connIter != connIter_end; ++connIter)
            {
                conn = *connIter;
                if(conn->GetSrcLayer()->GetGroup() == group)
                {
                    loopDetected = true;
                    break;
                }
            }
        }
        else
        {
            loopDetected = true;
        }


        layerIter_end = scc->end();
        for(layerIter = scc->begin(); layerIter != layerIter_end; ++layerIter)
        {
            layer = *layerIter;
            group = layer->GetGroup();

            layer->SetPStream(pStream);

            connIter_end = layer->GetSrcConnections().end();
            for(connIter = layer->GetSrcConnections().begin(); connIter != connIter_end; ++connIter)
            {
                conn = *connIter;
                conn->SetPStream(pStream);
            }
        }

        if(loopDetected == true)
        {
            if(loopIdx < nLoop - 1 && loopIdx * MAX_NUM_PSTREAM_PER_LOC / nLoop != (loopIdx + 1) * MAX_NUM_PSTREAM_PER_LOC / nLoop)
            {
                pStream = new PStream();
                engine->StreamCreate(*pStream, computeLoc[0]);
                pStreamList.push_back(pStream);
            }
            loopIdx++;
        }
    }
}
#endif


void Rnn::CreateDefaultPStream(const unsigned long loc)
{
    verify(defaultPStream == NULL);

    defaultPStream = new PStream();
    engine->StreamCreate(*defaultPStream, loc);
}


void Rnn::DestroyDefaultPStream()
{
    if(defaultPStream == NULL) return;

    engine->StreamDestroy(*defaultPStream);

    delete defaultPStream;
    defaultPStream = NULL;
}


void Rnn::FixCurrentWeights(const bool enable)
{
    for(auto &conn : connSet)
    {
        conn->FixWeights(enable);
    }
}


void Rnn::Synchronize()
{
    verify(engine != NULL);

    PStreamList::const_iterator iter, iter_end;

    iter_end = pStreamList.end();
    for(iter = pStreamList.begin(); iter != iter_end; ++iter)
    {
        engine->StreamSynchronize(**iter);
    }
}


void Rnn::StreamWait(PStream &stream)
{
    verify(engine != NULL);

    LayerMap::const_iterator layerIter, layerIter_end;
    ConnSet::const_iterator connIter, connIter_end;


    layerIter_end = layerMap.end();
    for(layerIter = layerMap.begin(); layerIter != layerIter_end; ++layerIter)
    {
        layerIter->second->EventRecord();
        layerIter->second->StreamWaitEvent(stream);
    }

    connIter_end = connSet.end();
    for(connIter = connSet.begin(); connIter != connIter_end; ++connIter)
    {
        (*connIter)->EventRecord();
        (*connIter)->StreamWaitEvent(stream);
    }
}


void Rnn::Clear()
{
    ClearSccList();
    ClearConnections();
    ClearLayers();
    ClearPStreams();
    DestroyDefaultPStream();

    SetEngine(NULL);

    isReady = false;
}


Layer *Rnn::FindLayer(const std::string &layerName)
{
    LayerMap::const_iterator iter;

    iter = layerMap.find(layerName);

    if(iter == layerMap.end()) return NULL;
    else return iter->second;
}


void Rnn::AddConnection(Layer *const from, Layer *const to, const ConnParam &param)
{
    Connection *conn = new Connection(from, to, param);

    conn->SetBatchSize(nStream, nUnroll);
    conn->SetEngine(engine);
    conn->SetPStream(defaultPStream);
    conn->InitWeights(param.GetInitWeightParam());

    from->AddDstConnection(conn);
    to->AddSrcConnection(conn);

    connSet.insert(conn);
}


void Rnn::DeleteConnection(Layer *const from, Layer *const to)
{
    Connection *conn;
    Layer::ConnList::const_iterator connIter, connIter_end;

    from->UnlinkMatrices();
    to->UnlinkMatrices();

    connIter_end = to->GetSrcConnections().end();
    for(connIter = to->GetSrcConnections().begin(); connIter != connIter_end; ++connIter)
    {
        if(from == (*connIter)->GetSrcLayer()) break;;
    }

    verify(connIter != connIter_end);

    conn = *connIter;

    from->RemoveDstConnection(conn);
    to->RemoveSrcConnection(conn);

    verify(connSet.erase(conn) == 1);
    delete conn;
}


void Rnn::LinkProbe(Probe &probe, Layer *const layer)
{
    probe.LinkLayer(layer);
}


void Rnn::ClearLayers()
{
    LayerMap::const_iterator iter, iter_end;

    iter_end = layerMap.end();
    for(iter = layerMap.begin(); iter != iter_end; ++iter)
    {
        delete iter->second;
    }

    layerMap.clear();
}


void Rnn::ClearConnections()
{
    ConnSet::const_iterator iter, iter_end;

    iter_end = connSet.end();
    for(iter = connSet.begin(); iter != iter_end; ++iter)
    {
        delete *iter;
    }

    connSet.clear();
}


void Rnn::ClearSccList()
{
    SccList::const_iterator iter, iter_end;

    iter_end = sccList.end();
    for(iter = sccList.begin(); iter != iter_end; ++iter)
    {
        delete *iter;
    }

    sccList.clear();
}


void Rnn::ClearPStreams()
{
    PStreamList::const_iterator iter, iter_end;

    iter_end = pStreamList.end();
    for(iter = pStreamList.begin(); iter != iter_end; ++iter)
    {
        engine->StreamDestroy(**iter);
        delete *iter;
    }

    pStreamList.clear();
}


void Rnn::SaveState(const std::string &path)
{
    /* For Linux */

    verify(path != "");

    verify(system(std::string("mkdir -p " + path).c_str()) == 0);

    //mkdir(path.c_str(), 0755);
    for(auto &layer : layerMap)
    {
        std::string dstLayerName = layer.second->GetName();
        std::string dirPath = path + "/" + dstLayerName;
        mkdir(dirPath.c_str(), 0755);

        //unsigned long i = 0;

        for(auto &conn : layer.second->GetSrcConnections())
        {
            std::string srcLayerName = conn->GetSrcLayer()->GetName();
            std::stringstream filename;
            //filename << dirPath << "/" << i << "." << srcLayerName;
            filename << dirPath << "/" << srcLayerName;

            conn->SaveState(filename.str());

            //i++;
        }
    }
}


void Rnn::LoadState(const std::string &path)
{
    /* For Linux */

    verify(path != "");

    for(auto &layer : layerMap)
    {
        std::string dstLayerName = layer.second->GetName();
        std::string dirPath = path + "/" + dstLayerName;

        //unsigned long i = 0;

        for(auto &conn : layer.second->GetSrcConnections())
        {
            std::string srcLayerName = conn->GetSrcLayer()->GetName();
            std::stringstream filename;
            //filename << dirPath << "/" << i << "." << srcLayerName;
            filename << dirPath << "/" << srcLayerName;

            conn->LoadState(filename.str());

            //i++;
        }
    }
}


const unsigned long Rnn::GetNumWeights()
{
    unsigned long numWeights = 0;

    for(auto &conn : connSet)
    {
        numWeights += conn->GetNumWeights();
    }

    return numWeights;
}


void Rnn::SetComputeLocs(const std::vector<unsigned long> &computeLoc)
{
    verify(engine != NULL);
    verify(computeLoc.size() > 0);

    for(unsigned long i = 0; i < computeLoc.size(); i++)
    {
        verify(engine->GetComputeLocIdx(computeLoc[i]) >= 0);
        verify(engine->GetComputeLocIdx(computeLoc[i]) < engine->GetNumComputeLocs());
        for(unsigned long j = i + 1; j < computeLoc.size(); j++)
        {
            verify(computeLoc[i] != computeLoc[j]);
        }
    }

    this->computeLoc = computeLoc;

    DestroyDefaultPStream();
    CreateDefaultPStream(computeLoc[0]);

    for(auto &layer : layerMap)
    {
        layer.second->SetPStream(defaultPStream);
    }

    for(auto &conn : connSet)
    {
        conn->SetPStream(defaultPStream);
    }

    isReady = false;
}


std::vector<unsigned long > &Rnn::GetComputeLocs()
{
    return computeLoc;
}


}

