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
    InitWeightParam() : isValid(false), addToDiag((FLOAT) 0) {}
    virtual ~InitWeightParam() {}

    const bool IsValid() const { return isValid; }

    virtual InitWeightParam *Clone() const { return new InitWeightParam(*this); }

    bool isValid;
    FLOAT addToDiag;
};


class InitWeightParamGaussian : public InitWeightParam
{
public:
    InitWeightParamGaussian() : InitWeightParam(), mean((FLOAT) 0), stdev((FLOAT) 0) {}
    InitWeightParamGaussian(FLOAT stdev) : InitWeightParam(), mean((FLOAT) 0), stdev(stdev) { isValid = true; }
    InitWeightParamGaussian(FLOAT mean, FLOAT stdev) : InitWeightParam(), mean(mean), stdev(stdev) { isValid = true; }

    virtual InitWeightParamGaussian *Clone() const { return new InitWeightParamGaussian(*this); }

    FLOAT mean;
    FLOAT stdev;
};


class InitWeightParamUniform : public InitWeightParam
{
public:
    InitWeightParamUniform() : InitWeightParam(), a((FLOAT) 0), b((FLOAT) 0) {}
    InitWeightParamUniform(FLOAT b) : InitWeightParam(), a(-b), b(b) { isValid = true; }
    InitWeightParamUniform(FLOAT a, FLOAT b) : InitWeightParam(), a(a), b(b) { isValid = true; }

    virtual InitWeightParamUniform *Clone() const { return new InitWeightParamUniform(*this); }

    FLOAT a;
    FLOAT b;
};

}

#endif /* FRACTAL_INITWEIGHTPARAM_H_ */

