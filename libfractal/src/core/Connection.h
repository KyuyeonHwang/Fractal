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


#ifndef FRACTAL_CONNECTION_H_
#define FRACTAL_CONNECTION_H_


#include <string>

#include "Engine.h"
#include "Matrix.h"
#include "InitWeightParam.h"
#include "FractalCommon.h"


namespace fractal
{

enum ConnType {CONN_FULL, CONN_IDENTITY, CONN_BROADCAST};

class Layer;


class ConnParam
{
public:
    ConnParam(const ConnType connType = CONN_FULL,
            const unsigned long delayAmount = (unsigned long) 0,
            const InitWeightParam &initWeightParam = InitWeightParam())
        : connType(connType),
        delayAmount(delayAmount),
        srcRangeFrom((long) -1),
        srcRangeTo((long) -1),
        dstRangeFrom((long) -1),
        dstRangeTo((long) -1),
        initWeightParam(initWeightParam) {}

    ConnParam(const unsigned long delayAmount,
            const InitWeightParam &initWeightParam = InitWeightParam())
        : connType(CONN_FULL),
        delayAmount(delayAmount),
        srcRangeFrom((long) -1),
        srcRangeTo((long) -1),
        dstRangeFrom((long) -1),
        dstRangeTo((long) -1),
        initWeightParam(initWeightParam) {}

    ConnParam(const InitWeightParam &initWeightParam)
        : connType(CONN_FULL),
        delayAmount((unsigned long) 0),
        srcRangeFrom((long) -1),
        srcRangeTo((long) -1),
        dstRangeFrom((long) -1),
        dstRangeTo((long) -1),
        initWeightParam(initWeightParam) {}

    ConnParam(const ConnType connType,
            const InitWeightParam &initWeightParam)
        : connType(connType),
        delayAmount((unsigned long) 0),
        srcRangeFrom((long) -1),
        srcRangeTo((long) -1),
        dstRangeFrom((long) -1),
        dstRangeTo((long) -1),
        initWeightParam(initWeightParam) {}

    ConnType connType;

    unsigned long delayAmount;
    long srcRangeFrom;
    long srcRangeTo;
    long dstRangeFrom;
    long dstRangeTo;

    InitWeightParam initWeightParam;
};


class Connection
{
public:
    Connection(Layer *const from, Layer *const to, const ConnParam &_param);
    virtual ~Connection();

    void SetEngine(Engine *const engine);
    void SetPStream(PStream *const stream);
    PStream &GetPStream();

    void SetBatchSize(const unsigned long nStream, const unsigned long nUnroll);

    void UnlinkMatrices();

    /// Initialize the weights.
    /*! \param initWeightParam Weight initialization parameter.
     */
    void InitWeights(const InitWeightParam &initWeightParam);

    /// Initialize AdaDelta.
    /*! Initialize mean-square variables.
     *  \param decayRate Decay rate of exponential averaging.
     *  \param initDenominator Set to \c true to initialize the denominator mean-square variable.
     */
    void InitAdadelta(const FLOAT decayRate, const bool initDenominator);

    /// Initialize Nesterov momentum.
    /*! Initialize the velocities to zero.
     */
    void InitNesterov();

    /// Initialize RMSprop.
    /*! Initialize the mean-square variables to one.
     *  \param decayRate Decay rate of exponential averaging.
     */
    void InitRmsprop(const FLOAT decayRate);

    void InitErr(const unsigned long idxFrom, const unsigned long idxTo);

    void Forward(const unsigned long idxFrom, const unsigned long idxTo);
    void UpdateDstErr(const unsigned long idxFrom, const unsigned long idxTo);
    void Backward(const unsigned long idxFrom, const unsigned long idxTo);

    void UpdateWeights(const unsigned long idxFrom, const unsigned long idxTo,
            const FLOAT rate, const FLOAT momentum, const bool adaptiveRates, const bool rmsprop);

    void ProcessWeights(const FLOAT noise);

    void FixWeights(const bool enable);

    inline const bool IsDelayed() const { return param.delayAmount > 0; }
    inline Layer *const GetSrcLayer() const { return srcLayer; }
    inline Layer *const GetDstLayer() const { return dstLayer; }

    void EventRecord();
    void StreamWaitEvent(PStream &stream);

    void ForwardWait();
    void BackwardWait();

    void SaveState(const std::string &filename);
    void LoadState(const std::string &filename);

    const unsigned long GetNumWeights();

protected:
    void TransposeWeightMatrix();

    Engine *engine;

    Layer *srcLayer, *dstLayer;

    unsigned long nStream;
    unsigned long nUnroll;

    FLOAT rmsDecayRate;

    Matrix<FLOAT> weights, weightsTrans;
    Matrix<FLOAT> weightsFwd, weightsBwd;
    bool weightsTransValid;

    Matrix<FLOAT> vels; /* momentum */
    Matrix<FLOAT> derivs, msDeriv; /* Rmsprop, Adadelta */
    Matrix<FLOAT> msDelta; /* Adadelta */
    Matrix<FLOAT> dstAct, srcAct;
    Matrix<FLOAT> dstErr, srcErr;

    PStream *stream;
    PEvent event;

    ConnParam param;

    unsigned long srcSize;
    unsigned long dstSize;

    bool broadcastWeightInitialized;

    bool fixed;

    friend Layer;
};

}

#endif /* FRACTAL_CONNECTION_H_ */

