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


#ifndef FRACTAL_PROBE_H_
#define FRACTAL_PROBE_H_

#include "Engine.h"
#include "Matrix.h"
#include "FractalCommon.h"


namespace fractal
{

class Layer;


class Probe
{
public:
    Probe(const bool isInput, const bool isOutput);
    virtual ~Probe();

    Engine *GetEngine() const;

    inline const bool IsInput() const { return _input; }
    inline const bool IsOutput() const { return _output; }

    void LinkLayer(Layer *const layer);
    void UnlinkLayer();

    const bool IsLinked() const;

    const unsigned long GetLayerSize() const;
    const std::string &GetLayerName() const;

    Matrix<FLOAT> &GetActivation();
    Matrix<FLOAT> &GetState();
    Matrix<FLOAT> &GetSrcErr();
    Matrix<FLOAT> &GetDstErr();

    PStream &GetPStream();

    void EventRecord();
    void EventSynchronize();
    void StreamWaitEvent(PStream &stream);
    void Wait();

protected:
    virtual void SetEngine(Engine *engine);
    void SetPStream();

    const bool _input, _output;
    Engine *engine;
    PStream stream;
    PEvent event;
    Layer *linkedLayer;

    bool isPStreamSet;

    friend Layer;
};


class InputProbe : public Probe
{
public:
    InputProbe() : Probe(true, false) {}
};


class OutputProbe : public Probe
{
public:
    OutputProbe() : Probe(false, true) {}
};

}

#endif /* FRACTAL_PROBE_H_ */

