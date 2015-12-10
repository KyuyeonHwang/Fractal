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


#include "Probe.h"

#include "Engine.h"
#include "Layer.h"


namespace fractal
{

Probe::Probe(const bool isInput, const bool isOutput) : _input(isInput), _output(isOutput)
{
    linkedLayer = NULL;
    engine = NULL;
    isPStreamSet = false;
}


Probe::~Probe()
{
    SetEngine(NULL);
    UnlinkLayer();
}


void Probe::SetEngine(Engine *engine)
{
    if(this->engine == engine) return;

    if(this->engine != NULL && isPStreamSet == true)
    {
        this->engine->EventDestroy(event);
        this->engine->StreamDestroy(stream);
        isPStreamSet = false;
    }

    this->engine = engine;

    SetPStream();
}


void Probe::SetPStream()
{
    if(engine == NULL)
    {
        verify(isPStreamSet == false);
        return;
    }

    if(isPStreamSet == true)
    {
        if(linkedLayer != NULL && linkedLayer->stream != NULL)
        {
            if(linkedLayer->stream->loc == stream.loc)
            {
                return;
            }
        }

        engine->EventDestroy(event);
        engine->StreamDestroy(stream);
        isPStreamSet = false;
    }

    if(linkedLayer != NULL && linkedLayer->stream != NULL)
    {
        engine->EventCreate(event, linkedLayer->stream->loc);
        engine->StreamCreate(stream, linkedLayer->stream->loc);
        isPStreamSet = true;
    }
}


Engine *Probe::GetEngine() const
{
    return engine;
}


void Probe::LinkLayer(Layer *const layer)
{
    if(linkedLayer != layer)
    {
        UnlinkLayer();
        linkedLayer = layer;
        layer->LinkProbe(this);
        SetEngine(linkedLayer->engine);
    }
}


void Probe::UnlinkLayer()
{
    Layer *tmp;

    if(linkedLayer != NULL)
    {
        tmp = linkedLayer;
        linkedLayer = NULL;
        tmp->UnlinkProbe();
        SetEngine(NULL);
    }
}


const bool Probe::IsLinked() const
{
    return (linkedLayer != NULL);
}


const unsigned long Probe::GetLayerSize() const
{
    verify(linkedLayer != NULL);
    return linkedLayer->GetSize();
}


const std::string &Probe::GetLayerName() const
{
    verify(linkedLayer != NULL);
    return linkedLayer->GetName();
}


Matrix<FLOAT> &Probe::GetActivation()
{
    verify(linkedLayer != NULL);
    return linkedLayer->act;
}


Matrix<FLOAT> &Probe::GetState()
{
    verify(linkedLayer != NULL);
    return linkedLayer->state;
}


Matrix<FLOAT> &Probe::GetSrcErr()
{
    verify(linkedLayer != NULL);
    return linkedLayer->srcErr;
}


Matrix<FLOAT> &Probe::GetDstErr()
{
    verify(linkedLayer != NULL);
    return linkedLayer->dstErr;
}


PStream &Probe::GetPStream()
{
    verify(engine != NULL);
    return stream;
}


void Probe::EventRecord()
{
    verify(engine != NULL);
    engine->EventRecord(event, stream);
}


void Probe::EventSynchronize()
{
    verify(engine != NULL);
    engine->EventSynchronize(event);
}


void Probe::StreamWaitEvent(PStream &stream)
{
    verify(engine != NULL);
    engine->StreamWaitEvent(stream, event);
}


void Probe::Wait()
{
    verify(engine != NULL);
    verify(linkedLayer != NULL);

    engine->StreamWaitEvent(stream, linkedLayer->event);
}

}

