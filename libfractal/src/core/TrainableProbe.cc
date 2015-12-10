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


#include "TrainableProbe.h"

#include "Engine.h"
#include "Layer.h"


namespace fractal
{

void TrainableProbe::InitTraining(const unsigned long nStream, const unsigned long nUnroll)
{
    verify(engine != NULL);
    verify(IsLinked() == true);

    this->nStream = nStream;
    this->nUnroll = nUnroll; 

    err.Resize(linkedLayer->GetSize(), nStream * nUnroll);
}


void TrainableProbe::InitEvaluation(const unsigned long nStream, const unsigned long nUnroll)
{
    verify(engine != NULL);
    verify(IsLinked() == true);

    this->nStream = nStream;
    this->nUnroll = nUnroll; 
}


void TrainableProbe::InitErr(const unsigned long idxFrom, const unsigned long idxTo)
{
    verify(engine != NULL);

    Matrix<FLOAT> errSub(err, idxFrom * nStream, idxTo * nStream + nStream - 1);

    engine->MatSet(errSub, (FLOAT) 0, stream);

    EventRecord();
}


void TrainableProbe::SetEngine(Engine *engine)
{
    Probe::SetEngine(engine);

    err.SetEngine(engine);
}


}

