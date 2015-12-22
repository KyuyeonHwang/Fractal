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


#ifndef FRACTAL_BASICLAYERS_H_
#define FRACTAL_BASICLAYERS_H_


#include "../core/Rnn.h"
#include "../core/FractalCommon.h"


namespace fractal
{

namespace basicLayers
{
    void AddLstmLayer(Rnn &rnn, const std::string &name, const std::string &biasLayer, const unsigned long delayAmount, const unsigned long size, const bool selfLoop, const InitWeightParam &initWeightParam = InitWeightParam(), const FLOAT initForgetGateBias = (FLOAT) 0);

    void AddFastLstmLayer(Rnn &rnn, const std::string &name, const std::string &biasLayer, const unsigned long delayAmount, const unsigned long size, const bool selfLoop, const InitWeightParam &initWeightParam = InitWeightParam());

    void AddClockedLstmLayer(Rnn &rnn, const std::string &name, const std::string &biasLayer, const unsigned long delayAmount, const unsigned long size, const bool selfLoop, const InitWeightParam &initWeightParam = InitWeightParam());

    void AddClockedLayer(Rnn &rnn, const std::string &name, const ActType actType, const AggType aggType,
            const unsigned long delayAmount, const unsigned long size);

    void AddGruLayer(Rnn &rnn, const std::string &name, const std::string &biasLayer, const unsigned long delayAmount, const unsigned long size, const InitWeightParam &initWeightParam = InitWeightParam());

    void AddFastGruLayer(Rnn &rnn, const std::string &name, const std::string &biasLayer, const unsigned long delayAmount, const unsigned long size, const InitWeightParam &initWeightParam = InitWeightParam());
}

}

#endif /* FRACTAL_BASICLAYERS_H_ */

