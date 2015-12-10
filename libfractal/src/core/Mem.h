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


#ifndef FRACTAL_MEM_H_
#define FRACTAL_MEM_H_


#include <mutex>

#include "FractalCommon.h"


namespace fractal
{

class PStream;
class Engine;

class Mem
{
public:
    Mem(Engine *const engine, size_t size);
    virtual ~Mem();

    inline const Engine *GetEngine() const { return engine; }
    inline void *const GetPtr(const unsigned long loc) const { return ptr[loc]; }
    inline void **const GetPtrs() { return ptr; }
    inline const bool IsValid(const unsigned long loc) const { return valid[loc]; }
    inline const unsigned long GetRecentLoc() const { return recentLoc; }
    inline const size_t GetSize() const { return size; }
    inline void SetSize(size_t size) { this->size = size; }

    void CopyFromHost(const size_t offsetDst, const void *ptrSrc, const size_t size, PStream &stream);
    void CopyToHost(const size_t offsetSrc, void *ptrDst, const size_t size, PStream &stream) const;

    void Validate(const unsigned long loc);
    void Invalidate();

    void Pull(const unsigned long loc, PStream &stream);
    void Push(const unsigned long loc);

    inline void Lock() { mtx.lock(); }
    inline void Unlock() { mtx.unlock(); }

protected:
    Mem(const Mem& mem);

    unsigned long numLoc;
    unsigned long recentLoc;
    size_t size;
    void **ptr;
    bool *valid;

    std::recursive_mutex mtx;

    Engine *engine;
};

}

#endif /* FRACTAL_MEM_H_ */

