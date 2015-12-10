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


#include "MovAvgStream.h"

#include <cstdlib>


using namespace fractal;


const unsigned long MovAvgStream::CHANNEL_INPUT = 0;
const unsigned long MovAvgStream::CHANNEL_TARGET = 1;

MovAvgStream::MovAvgStream()
{
    windowSize = 5;
    nStream = 1;

    Init();
}


void MovAvgStream::Init()
{
    state.resize(nStream * windowSize);
    state.shrink_to_fit();

    idx.resize(nStream);
    idx.shrink_to_fit();

    Reset();
}


void MovAvgStream::SetNumStream(const unsigned long nStream)
{
    verify(nStream > 0);

    this->nStream = nStream;

    Init();
}


const unsigned long MovAvgStream::GetNumStream() const
{
    return nStream;
}


const unsigned long MovAvgStream::GetNumChannel() const
{
    return 2;
}


const ChannelInfo MovAvgStream::GetChannelInfo(const unsigned long channelIdx) const
{
    ChannelInfo channelInfo;

    switch(channelIdx)
    {
        case CHANNEL_INPUT:
            channelInfo.dataType = ChannelInfo::DATATYPE_VECTOR;
            channelInfo.frameSize = 1;
            channelInfo.frameDim = 1;
            break;

        case CHANNEL_TARGET:
            channelInfo.dataType = ChannelInfo::DATATYPE_VECTOR;
            channelInfo.frameSize = 1;
            channelInfo.frameDim = 1;
            break;

        default:
            verify(false);
    }

    return channelInfo;
}


void MovAvgStream::Reset()
{
    unsigned long i;

    for(i = 0; i < nStream * windowSize; i++)
        state[i] = 0;

    for(i = 0; i < nStream; i++)
        idx[i] = 0;
}


void MovAvgStream::Next(const unsigned long streamIdx)
{
    idx[streamIdx] = (idx[streamIdx] + 1) % windowSize;
    state[streamIdx * windowSize + idx[streamIdx]] = rand() % 2;
}


void MovAvgStream::GenerateFrame(const unsigned long streamIdx, const unsigned long channelIdx, void *const frame)
{
    unsigned long i, j;

    switch(channelIdx)
    {
        case CHANNEL_INPUT:
            reinterpret_cast<FLOAT *>(frame)[0] = (FLOAT) state[streamIdx * windowSize + idx[streamIdx]];

            break;

        case CHANNEL_TARGET:
            j = 0;

            for(i = 0; i < windowSize; i++)
            {
                j += state[streamIdx * windowSize + i];
            }

            reinterpret_cast<FLOAT *>(frame)[0] = (FLOAT) j / windowSize;

            break;

        default:
            verify(false);
    }
}


void MovAvgStream::SetWindowSize(const unsigned long windowSize)
{
    verify(windowSize > 0);

    this->windowSize = windowSize;

    Init();
}


