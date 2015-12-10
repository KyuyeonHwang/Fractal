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


#include "Pipe.h"

namespace fractal
{

Pipe::Pipe()
{
    Init();
}


void Pipe::Init()
{
    signalCount = 0;
}


void Pipe::SendSignal()
{
    mtx.lock();
    signalCount++;
    mtx.unlock();
    cv.notify_one();
}


void Pipe::Wait(const unsigned long count)
{
    std::unique_lock<std::mutex> lock(mtx);
    while(signalCount < count)
    {
        cv.wait(lock);
    }
    signalCount -= count;
}

}

