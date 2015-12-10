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


#ifndef FRACTAL_LAYER_H_
#define FRACTAL_LAYER_H_


#include <string>
#include <list>

#include "Engine.h"
#include "Matrix.h"
#include "FractalCommon.h"



namespace fractal
{

enum ActType {ACT_BIAS, ACT_SIGMOID, ACT_TANH, ACT_SOFTPLUS,
    ACT_RECTLINEAR, ACT_LINEAR, ACT_ONE_MINUS_LINEAR, ACT_INVERSE,
    ACT_SOFTMAX, ACT_DROPOUT, ACT_CTC_DECODE};

enum AggType {AGG_DONTCARE, AGG_SUM, AGG_MULT};

class Connection;
class Probe;


class LayerParam
{
public:
    LayerParam() : initVal((FLOAT) 0), dropoutRate((FLOAT) 0), blockErr(false) {}

    FLOAT initVal;
    FLOAT dropoutRate;

    bool blockErr;
};


class Layer
{
public:
    typedef std::list<Connection *> ConnList;

    Layer(const std::string &name, ActType actType, AggType aggType, const unsigned long size, const LayerParam &param);
    virtual ~Layer();

    void SetEngine(Engine *const engine);
    void SetPStream(PStream *const stream);
    PStream &GetPStream();

    void AddSrcConnection(Connection *const conn);
    void AddDstConnection(Connection *const conn);
    void RemoveSrcConnection(Connection *const conn);
    void RemoveDstConnection(Connection *const conn);

    inline const std::string &GetName() const { return name; }
    inline const unsigned long GetSize() const { return size; }

    void SetBatchSize(const unsigned long nStream, const unsigned long nUnroll);

    inline const unsigned long GetNumStreams() const { return nStream; }
    inline const unsigned long GetNumUnrollSteps() const { return nUnroll; }

    void SetInitVal(const FLOAT val);

    void UnlinkMatrices();

    void InitAct(const unsigned long idxFrom, const unsigned long idxTo);
    void InitErr(const unsigned long idxFrom, const unsigned long idxTo);

    void EnableDropout(const bool enable);
    void GenerateDropoutMask(const unsigned long idxFrom, const unsigned long idxTo);

    void Forward(const unsigned long idxFrom, const unsigned long idxTo);
    void Backward(const unsigned long idxFrom, const unsigned long idxTo);

    void CalcActDeriv(const unsigned long idxFrom, const unsigned long idxTo);

    void LinkProbe(Probe *const probe);
    void UnlinkProbe();
    const bool IsLinked() const;

    inline const ConnList &GetSrcConnections() const { return srcList; }
    inline const ConnList &GetDstConnections() const { return dstList; }

    /* For graph algorithms */
    inline void SetVisited(const bool isVisited) { this->isVisited = isVisited; }
    inline const bool GetVisited() const { return isVisited; }
    inline void SetIndex(const long index) { this->index = index; }
    inline const long GetIndex() const { return index; }
    inline void SetGroup(const long group) { this->group = group; }
    inline const long GetGroup() const { return group; }

    void EventRecord();
    void StreamWaitEvent(PStream &stream);

    void ForwardWait();
    void BackwardWait();

protected:
    Layer(const Layer &obj);

    void Activation(const unsigned long idxFrom, const unsigned long idxTo);
    void UpdateState(const unsigned long idxFrom, const unsigned long idxTo);

    void UpdateDstErr(const unsigned long idxFrom, const unsigned long idxTo);
    void UpdateSrcErr(const unsigned long idxFrom, const unsigned long idxTo);
    void DistributeErr(Connection *conn, const unsigned long idxFrom, const unsigned long idxTo);

    Engine *engine;

    ActType actType;
    AggType aggType;

    std::string name;
    unsigned long size;
    unsigned long nStream;
    unsigned long nUnroll;

    ConnList srcList;
    ConnList dstList;

    Matrix<FLOAT> act, state, srcErr, dstErr;

    /* For dropout */
    bool dropoutEnabled;
    Matrix<FLOAT> dropoutMask;

    /* For CTC decoder layer */
    Matrix<INT> idxMax;

    Probe *linkedProbe;

    LayerParam param;

    /* For graph algorithms */
    bool isVisited;
    long index, group;

    bool performBackward;

    PStream *stream;
    PEvent event;

    friend Probe;
    friend Connection;
};

}

#endif /* FRACTAL_LAYER_H_ */

