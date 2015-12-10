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


#ifndef FRACTAL_STREAM_H_
#define FRACTAL_STREAM_H_

#include <vector>

#include "ChannelInfo.h"
#include "../core/FractalCommon.h"


namespace fractal
{


class Stream
{
public:
    virtual void SetNumStream(const unsigned long nStream) = 0;

    virtual const unsigned long GetNumStream() const = 0;
    virtual const unsigned long GetNumChannel() const = 0;
    virtual const ChannelInfo GetChannelInfo(const unsigned long channelIdx) const = 0;

    virtual void Reset() = 0;
    virtual void Next(const unsigned long streamIdx) = 0;
    virtual void GenerateFrame(const unsigned long streamIdx, const unsigned long channelIdx, void *const frame) = 0;
};



}


#endif /* FRACTAL_STREAM_H_ */

