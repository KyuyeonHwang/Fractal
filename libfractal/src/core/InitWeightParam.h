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


#ifndef FRACTAL_INITWEIGHTPARAM_H_
#define FRACTAL_INITWEIGHTPARAM_H_

#include "FractalCommon.h"

namespace fractal
{

class InitWeightParam
{
public:
    InitWeightParam() : isValid(false), mean((FLOAT) 0), stdev((FLOAT) 0), addToDiag((FLOAT) 0) {}
    InitWeightParam(FLOAT stdev) : isValid(true), mean((FLOAT) 0), stdev(stdev), addToDiag((FLOAT) 0) {}
    InitWeightParam(FLOAT mean, FLOAT stdev) : isValid(true), mean(mean), stdev(stdev), addToDiag((FLOAT) 0) {}

    const bool IsValid() const { return isValid; }

    bool isValid;
    FLOAT mean;
    FLOAT stdev;
    FLOAT addToDiag;
};

}

#endif /* FRACTAL_INITWEIGHTPARAM_H_ */

