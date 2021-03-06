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


#ifndef FRACTAL_PIPE_H_
#define FRACTAL_PIPE_H_


#include <mutex>
#include <condition_variable>

#include "../core/FractalCommon.h"


namespace fractal
{

class Pipe
{
public:
    Pipe();

    void Init();

    void SendSignal();
    void Wait(const unsigned long count);

protected:
    unsigned long signalCount;

    std::mutex mtx;
    std::condition_variable cv;
};

}

#endif /* FRACTAL_PIPE_H_ */

