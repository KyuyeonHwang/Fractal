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


#ifndef __MOVAVGSTREAM_H__
#define __MOVAVGSTREAM_H__

#include <fractal/fractal.h>


class MovAvgStream : public fractal::Stream
{
public:
    static const unsigned long CHANNEL_INPUT;
    static const unsigned long CHANNEL_TARGET;

    MovAvgStream();

    void SetNumStream(const unsigned long nStream);

    const unsigned long GetNumStream() const;
    const unsigned long GetNumChannel() const;
    const fractal::ChannelInfo GetChannelInfo(const unsigned long channelIdx) const;

    void Reset();
    void Next(const unsigned long streamIdx);
    void GenerateFrame(const unsigned long streamIdx, const unsigned long channelIdx, void *const frame);

    void SetWindowSize(const unsigned long windowSize);

protected:
    void Init();

    unsigned long nStream;
    unsigned long windowSize;

    std::vector<unsigned long> state;
    std::vector<unsigned long> idx;
};



#endif /* __MOVAVGSTREAM_H__ */

