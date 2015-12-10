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


#include "Mem.h"

#include "Engine.h"

namespace fractal
{

Mem::Mem(Engine *const engine, const size_t size) : engine(engine)
{
    unsigned long i;

    numLoc = engine->GetNumLocs();
    recentLoc = engine->GetHostLoc();
    ptr = new void *[numLoc];
    valid = new bool[numLoc];
    this->size = size;

    for(i = 0; i < numLoc; i++)
    {
        ptr[i] = NULL;
        valid[i] = false;
    }

    engine->MemAdd(this);
}


Mem::~Mem()
{
    engine->MemDealloc(this);
    engine->MemDel(this);

    delete[] ptr;
    delete[] valid;
}


void Mem::CopyFromHost(const size_t offsetDst, const void *ptrSrc, const size_t size, PStream &stream)
{
    verify(engine != NULL);

    engine->MemCopyFromHost(this, offsetDst, ptrSrc, size, stream);
}


void Mem::CopyToHost(const size_t offsetSrc, void *ptrDst, const size_t size, PStream &stream) const
{
    verify(engine != NULL);

    engine->MemCopyToHost(this, offsetSrc, ptrDst, size, stream);
}


void Mem::Validate(const unsigned long loc)
{
    Lock();

    recentLoc = loc;

    valid[recentLoc] = true;

    Unlock();
}


void Mem::Invalidate()
{
    Lock();

    for(unsigned long i = 0; i < numLoc; i++)
        valid[i] = false;

    Unlock();
}


void Mem::Pull(const unsigned long loc, PStream &stream)
{
    engine->MemPull(this, loc, stream);
}


void Mem::Push(const unsigned long loc)
{
    Lock();

    for(unsigned long i = 0; i < numLoc; i++)
        valid[i] = (i == loc);

    recentLoc = loc;

    Unlock();
}

}

