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


#include "BasicLayers.h"

#include <string>

#include "../core/Rnn.h"

namespace fractal
{

namespace basicLayers
{
    void AddLstmLayer(Rnn &rnn, const std::string &name, const std::string &biasLayer, const unsigned long delayAmount,
            const unsigned long size, const bool selfLoop, const InitWeightParam &initWeightParam, const FLOAT initForgetGateBias)
    {
        const std::string prefix = name + ".";
        InitWeightParam initForgetGateBiasParam = initWeightParam;

        if(initForgetGateBias != (FLOAT) 0)
        {
            verify(initForgetGateBiasParam.IsValid() == true);
            initForgetGateBiasParam.mean = initForgetGateBias;
        }

        rnn.AddLayer(prefix + "INPUT", ACT_TANH, AGG_SUM, size);
        rnn.AddLayer(prefix + "INPUT_GATE_PEEP", ACT_LINEAR, AGG_MULT, size);
        rnn.AddLayer(prefix + "FORGET_GATE_PEEP", ACT_LINEAR, AGG_MULT, size);
        rnn.AddLayer(prefix + "OUTPUT_GATE_PEEP", ACT_LINEAR, AGG_MULT, size);
        rnn.AddLayer(prefix + "INPUT_GATE", ACT_SIGMOID, AGG_SUM, size);
        rnn.AddLayer(prefix + "INPUT_GATE_MULT", ACT_LINEAR, AGG_MULT, size);
        rnn.AddLayer(prefix + "MEMORY_CELL", ACT_LINEAR, AGG_SUM, size);
        rnn.AddLayer(prefix + "MEMORY_CELL.DELAYED", ACT_LINEAR, AGG_MULT, size);
        rnn.AddLayer(prefix + "FORGET_GATE", ACT_SIGMOID, AGG_SUM, size);
        rnn.AddLayer(prefix + "FORGET_GATE_MULT", ACT_LINEAR, AGG_MULT, size);
        rnn.AddLayer(prefix + "OUTPUT_SQUASH", ACT_TANH, AGG_SUM, size);
        rnn.AddLayer(prefix + "OUTPUT_GATE", ACT_SIGMOID, AGG_SUM, size);
        rnn.AddLayer(prefix + "OUTPUT", ACT_LINEAR, AGG_MULT, size);

        rnn.AddConnection(prefix + "INPUT", prefix + "INPUT_GATE_MULT", CONN_IDENTITY);
        rnn.AddConnection(prefix + "INPUT_GATE", prefix + "INPUT_GATE_MULT", CONN_IDENTITY);
        rnn.AddConnection(prefix + "INPUT_GATE_MULT", prefix + "MEMORY_CELL", CONN_IDENTITY);
        rnn.AddConnection(prefix + "MEMORY_CELL", prefix + "MEMORY_CELL.DELAYED", {CONN_IDENTITY, delayAmount});
        rnn.AddConnection(prefix + "MEMORY_CELL.DELAYED", prefix + "FORGET_GATE_MULT", CONN_IDENTITY);
        rnn.AddConnection(prefix + "FORGET_GATE", prefix + "FORGET_GATE_MULT", CONN_IDENTITY);
        rnn.AddConnection(prefix + "FORGET_GATE_MULT", prefix + "MEMORY_CELL", CONN_IDENTITY);
        rnn.AddConnection(prefix + "MEMORY_CELL", prefix + "OUTPUT_SQUASH", CONN_IDENTITY);
        rnn.AddConnection(prefix + "OUTPUT_SQUASH", prefix + "OUTPUT", CONN_IDENTITY);
        rnn.AddConnection(prefix + "OUTPUT_GATE", prefix + "OUTPUT", CONN_IDENTITY);

        /* Biases */
        rnn.AddConnection(biasLayer, prefix + "INPUT", initWeightParam);
        rnn.AddConnection(biasLayer, prefix + "INPUT_GATE", initWeightParam);
        rnn.AddConnection(biasLayer, prefix + "FORGET_GATE", initForgetGateBiasParam);
        rnn.AddConnection(biasLayer, prefix + "OUTPUT_GATE", initWeightParam);

        /* Peephole connections */
        rnn.AddConnection(prefix + "MEMORY_CELL.DELAYED", prefix + "INPUT_GATE_PEEP", CONN_IDENTITY);
        rnn.AddConnection(prefix + "MEMORY_CELL.DELAYED", prefix + "FORGET_GATE_PEEP", CONN_IDENTITY);
        rnn.AddConnection(prefix + "MEMORY_CELL", prefix + "OUTPUT_GATE_PEEP", CONN_IDENTITY);
        rnn.AddConnection(biasLayer, prefix + "INPUT_GATE_PEEP", initWeightParam);
        rnn.AddConnection(biasLayer, prefix + "FORGET_GATE_PEEP", initWeightParam);
        rnn.AddConnection(biasLayer, prefix + "OUTPUT_GATE_PEEP", initWeightParam);
        rnn.AddConnection(prefix + "INPUT_GATE_PEEP", prefix + "INPUT_GATE", CONN_IDENTITY);
        rnn.AddConnection(prefix + "FORGET_GATE_PEEP", prefix + "FORGET_GATE", CONN_IDENTITY);
        rnn.AddConnection(prefix + "OUTPUT_GATE_PEEP", prefix + "OUTPUT_GATE", CONN_IDENTITY);

        if(selfLoop == true)
        {
            rnn.AddLayer(prefix + "OUTPUT.DELAYED", ACT_LINEAR, AGG_MULT, size);

            rnn.AddConnection(prefix + "OUTPUT", prefix + "OUTPUT.DELAYED", {CONN_IDENTITY, delayAmount});
            rnn.AddConnection(prefix + "OUTPUT.DELAYED", prefix + "INPUT_GATE", initWeightParam);
            rnn.AddConnection(prefix + "OUTPUT.DELAYED", prefix + "FORGET_GATE", initWeightParam);
            rnn.AddConnection(prefix + "OUTPUT.DELAYED", prefix + "OUTPUT_GATE", initWeightParam);
            rnn.AddConnection(prefix + "OUTPUT.DELAYED", prefix + "INPUT", initWeightParam);
        }
    }


    void AddFastLstmLayer(Rnn &rnn, const std::string &name, const std::string &biasLayer, const unsigned long delayAmount,
            const unsigned long size, const bool selfLoop, const InitWeightParam &initWeightParam)
    {
        const std::string prefix = name + ".";

        rnn.AddLayer(prefix + "INPUT", ACT_LINEAR, AGG_SUM, 4 * size);
        rnn.AddLayer(prefix + "INPUT_SQUASH", ACT_TANH, AGG_SUM, size);
        rnn.AddLayer(prefix + "INPUT_GATE_PEEP", ACT_LINEAR, AGG_MULT, size);
        rnn.AddLayer(prefix + "FORGET_GATE_PEEP", ACT_LINEAR, AGG_MULT, size);
        rnn.AddLayer(prefix + "OUTPUT_GATE_PEEP", ACT_LINEAR, AGG_MULT, size);
        rnn.AddLayer(prefix + "INPUT_GATE", ACT_SIGMOID, AGG_SUM, size);
        rnn.AddLayer(prefix + "INPUT_GATE_MULT", ACT_LINEAR, AGG_MULT, size);
        rnn.AddLayer(prefix + "MEMORY_CELL", ACT_LINEAR, AGG_SUM, size);
        rnn.AddLayer(prefix + "MEMORY_CELL.DELAYED", ACT_LINEAR, AGG_MULT, size);
        rnn.AddLayer(prefix + "FORGET_GATE", ACT_SIGMOID, AGG_SUM, size);
        rnn.AddLayer(prefix + "FORGET_GATE_MULT", ACT_LINEAR, AGG_MULT, size);
        rnn.AddLayer(prefix + "OUTPUT_SQUASH", ACT_TANH, AGG_SUM, size);
        rnn.AddLayer(prefix + "OUTPUT_GATE", ACT_SIGMOID, AGG_SUM, size);
        rnn.AddLayer(prefix + "OUTPUT", ACT_LINEAR, AGG_MULT, size);

        ConnParam connParam(CONN_IDENTITY);

        connParam.srcRangeFrom = 0;
        connParam.srcRangeTo = size - 1;
        rnn.AddConnection(prefix + "INPUT", prefix + "INPUT_SQUASH", connParam);

        connParam.srcRangeFrom += size;
        connParam.srcRangeTo += size;
        rnn.AddConnection(prefix + "INPUT", prefix + "INPUT_GATE", connParam);

        connParam.srcRangeFrom += size;
        connParam.srcRangeTo += size;
        rnn.AddConnection(prefix + "INPUT", prefix + "FORGET_GATE", connParam);

        connParam.srcRangeFrom += size;
        connParam.srcRangeTo += size;
        rnn.AddConnection(prefix + "INPUT", prefix + "OUTPUT_GATE", connParam);

        rnn.AddConnection(prefix + "INPUT_SQUASH", prefix + "INPUT_GATE_MULT", CONN_IDENTITY);
        rnn.AddConnection(prefix + "INPUT_GATE", prefix + "INPUT_GATE_MULT", CONN_IDENTITY);
        rnn.AddConnection(prefix + "INPUT_GATE_MULT", prefix + "MEMORY_CELL", CONN_IDENTITY);
        rnn.AddConnection(prefix + "MEMORY_CELL", prefix + "MEMORY_CELL.DELAYED", {CONN_IDENTITY, delayAmount});
        rnn.AddConnection(prefix + "MEMORY_CELL.DELAYED", prefix + "FORGET_GATE_MULT", CONN_IDENTITY);
        rnn.AddConnection(prefix + "FORGET_GATE", prefix + "FORGET_GATE_MULT", CONN_IDENTITY);
        rnn.AddConnection(prefix + "FORGET_GATE_MULT", prefix + "MEMORY_CELL", CONN_IDENTITY);
        rnn.AddConnection(prefix + "MEMORY_CELL", prefix + "OUTPUT_SQUASH", CONN_IDENTITY);
        rnn.AddConnection(prefix + "OUTPUT_SQUASH", prefix + "OUTPUT", CONN_IDENTITY);
        rnn.AddConnection(prefix + "OUTPUT_GATE", prefix + "OUTPUT", CONN_IDENTITY);

        /* Bias */
        rnn.AddConnection(biasLayer, prefix + "INPUT", initWeightParam);

        /* Peephole connections */
        rnn.AddConnection(prefix + "MEMORY_CELL.DELAYED", prefix + "INPUT_GATE_PEEP", CONN_IDENTITY);
        rnn.AddConnection(prefix + "MEMORY_CELL.DELAYED", prefix + "FORGET_GATE_PEEP", CONN_IDENTITY);
        rnn.AddConnection(prefix + "MEMORY_CELL", prefix + "OUTPUT_GATE_PEEP", CONN_IDENTITY);
        rnn.AddConnection(biasLayer, prefix + "INPUT_GATE_PEEP", initWeightParam);
        rnn.AddConnection(biasLayer, prefix + "FORGET_GATE_PEEP", initWeightParam);
        rnn.AddConnection(biasLayer, prefix + "OUTPUT_GATE_PEEP", initWeightParam);
        rnn.AddConnection(prefix + "INPUT_GATE_PEEP", prefix + "INPUT_GATE", CONN_IDENTITY);
        rnn.AddConnection(prefix + "FORGET_GATE_PEEP", prefix + "FORGET_GATE", CONN_IDENTITY);
        rnn.AddConnection(prefix + "OUTPUT_GATE_PEEP", prefix + "OUTPUT_GATE", CONN_IDENTITY);

        if(selfLoop == true)
        {
            rnn.AddLayer(prefix + "OUTPUT.DELAYED", ACT_LINEAR, AGG_MULT, size);

            rnn.AddConnection(prefix + "OUTPUT", prefix + "OUTPUT.DELAYED", {CONN_IDENTITY, delayAmount});
            rnn.AddConnection(prefix + "OUTPUT.DELAYED", prefix + "INPUT", initWeightParam);
        }
    }


    void AddClockedLstmLayer(Rnn &rnn, const std::string &name, const std::string &biasLayer, const unsigned long delayAmount,
            const unsigned long size, const bool selfLoop, const InitWeightParam &initWeightParam)
    {
        const std::string prefix = name + ".";

        rnn.AddLayer(prefix + "CLOCK", ACT_LINEAR, AGG_MULT, 1);
        rnn.AddLayer(prefix + "INPUT", ACT_LINEAR, AGG_SUM, 4 * size);
        rnn.AddLayer(prefix + "INPUT_SQUASH", ACT_TANH, AGG_SUM, size);
        rnn.AddLayer(prefix + "INPUT_GATE_PEEP", ACT_LINEAR, AGG_MULT, size);
        rnn.AddLayer(prefix + "FORGET_GATE_PEEP", ACT_LINEAR, AGG_MULT, size);
        rnn.AddLayer(prefix + "OUTPUT_GATE_PEEP", ACT_LINEAR, AGG_MULT, size);
        rnn.AddLayer(prefix + "INPUT_GATE", ACT_SIGMOID, AGG_SUM, size);
        rnn.AddLayer(prefix + "INPUT_GATE_MULT", ACT_LINEAR, AGG_MULT, size);
        rnn.AddLayer(prefix + "FORGET_GATE", ACT_SIGMOID, AGG_SUM, size);
        rnn.AddLayer(prefix + "FORGET_GATE_MULT", ACT_LINEAR, AGG_MULT, size);
        rnn.AddLayer(prefix + "OUTPUT_SQUASH", ACT_TANH, AGG_SUM, size);
        rnn.AddLayer(prefix + "OUTPUT_GATE", ACT_SIGMOID, AGG_SUM, size);

        /* Clocked layers */
        AddClockedLayer(rnn, prefix + "MEMORY_CELL", ACT_LINEAR, AGG_SUM, delayAmount, size);
        AddClockedLayer(rnn, prefix + "OUTPUT", ACT_LINEAR, AGG_MULT, delayAmount, size);

        rnn.AddConnection(prefix + "CLOCK", prefix + "MEMORY_CELL.CLOCK", CONN_IDENTITY);
        rnn.AddConnection(prefix + "CLOCK", prefix + "OUTPUT.CLOCK", CONN_IDENTITY);

        
        ConnParam connParam(CONN_IDENTITY);

        connParam.srcRangeFrom = 0;
        connParam.srcRangeTo = size - 1;
        rnn.AddConnection(prefix + "INPUT", prefix + "INPUT_SQUASH", connParam);

        connParam.srcRangeFrom += size;
        connParam.srcRangeTo += size;
        rnn.AddConnection(prefix + "INPUT", prefix + "INPUT_GATE", connParam);

        connParam.srcRangeFrom += size;
        connParam.srcRangeTo += size;
        rnn.AddConnection(prefix + "INPUT", prefix + "FORGET_GATE", connParam);

        connParam.srcRangeFrom += size;
        connParam.srcRangeTo += size;
        rnn.AddConnection(prefix + "INPUT", prefix + "OUTPUT_GATE", connParam);

        rnn.AddConnection(prefix + "INPUT_SQUASH", prefix + "INPUT_GATE_MULT", CONN_IDENTITY);
        rnn.AddConnection(prefix + "INPUT_GATE", prefix + "INPUT_GATE_MULT", CONN_IDENTITY);
        rnn.AddConnection(prefix + "INPUT_GATE_MULT", prefix + "MEMORY_CELL.INPUT", CONN_IDENTITY);
        rnn.AddConnection(prefix + "MEMORY_CELL.DELAYED", prefix + "FORGET_GATE_MULT", CONN_IDENTITY);
        rnn.AddConnection(prefix + "FORGET_GATE", prefix + "FORGET_GATE_MULT", CONN_IDENTITY);
        rnn.AddConnection(prefix + "FORGET_GATE_MULT", prefix + "MEMORY_CELL.INPUT", CONN_IDENTITY);
        rnn.AddConnection(prefix + "MEMORY_CELL", prefix + "OUTPUT_SQUASH", CONN_IDENTITY);
        rnn.AddConnection(prefix + "OUTPUT_SQUASH", prefix + "OUTPUT.INPUT", CONN_IDENTITY);
        rnn.AddConnection(prefix + "OUTPUT_GATE", prefix + "OUTPUT.INPUT", CONN_IDENTITY);

        /* Bias */
        rnn.AddConnection(biasLayer, prefix + "INPUT", initWeightParam);

        /* Peephole connections */
        rnn.AddConnection(prefix + "MEMORY_CELL.DELAYED", prefix + "INPUT_GATE_PEEP", CONN_IDENTITY);
        rnn.AddConnection(prefix + "MEMORY_CELL.DELAYED", prefix + "FORGET_GATE_PEEP", CONN_IDENTITY);
        rnn.AddConnection(prefix + "MEMORY_CELL", prefix + "OUTPUT_GATE_PEEP", CONN_IDENTITY);
        rnn.AddConnection(biasLayer, prefix + "INPUT_GATE_PEEP", initWeightParam);
        rnn.AddConnection(biasLayer, prefix + "FORGET_GATE_PEEP", initWeightParam);
        rnn.AddConnection(biasLayer, prefix + "OUTPUT_GATE_PEEP", initWeightParam);
        rnn.AddConnection(prefix + "INPUT_GATE_PEEP", prefix + "INPUT_GATE", CONN_IDENTITY);
        rnn.AddConnection(prefix + "FORGET_GATE_PEEP", prefix + "FORGET_GATE", CONN_IDENTITY);
        rnn.AddConnection(prefix + "OUTPUT_GATE_PEEP", prefix + "OUTPUT_GATE", CONN_IDENTITY);

        if(selfLoop == true)
        {
            rnn.AddConnection(prefix + "OUTPUT.DELAYED", prefix + "INPUT", initWeightParam);
        }
    }


    void AddClockedLayer(Rnn &rnn, const std::string &name, const ActType actType, const AggType aggType,
            const unsigned long delayAmount, const unsigned long size)
    {
        const std::string prefix = name + ".";
        
        rnn.AddLayer(name, ACT_LINEAR, AGG_SUM, size);
        rnn.AddLayer(prefix + "INPUT", actType, aggType, size);
        rnn.AddLayer(prefix + "DELAYED", ACT_LINEAR, AGG_MULT, size);

        rnn.AddLayer(prefix + "SELECT_I", ACT_LINEAR, AGG_MULT, size);
        rnn.AddLayer(prefix + "SELECT_O", ACT_LINEAR, AGG_MULT, size);

        rnn.AddLayer(prefix + "CLOCK", ACT_LINEAR, AGG_MULT, 1);
        rnn.AddLayer(prefix + "CLOCK_BAR", ACT_ONE_MINUS_LINEAR, AGG_SUM, 1);

        rnn.AddConnection(prefix + "CLOCK", prefix + "CLOCK_BAR", CONN_IDENTITY);

        rnn.AddConnection(name, prefix + "DELAYED", {CONN_IDENTITY, delayAmount});

        rnn.AddConnection(prefix + "CLOCK", prefix + "SELECT_I", CONN_BROADCAST);
        rnn.AddConnection(prefix + "INPUT", prefix + "SELECT_I", CONN_IDENTITY);

        rnn.AddConnection(prefix + "CLOCK_BAR", prefix + "SELECT_O", CONN_BROADCAST);
        rnn.AddConnection(prefix + "DELAYED", prefix + "SELECT_O", CONN_IDENTITY);

        rnn.AddConnection(prefix + "SELECT_I", name, CONN_IDENTITY);
        rnn.AddConnection(prefix + "SELECT_O", name, CONN_IDENTITY);
    }
}

}

