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


#ifndef FRACTAL_DATASTREAM_H_
#define FRACTAL_DATASTREAM_H_


#include <vector>
#include <random>

#include "Stream.h"
#include "../core/FractalCommon.h"


namespace fractal
{

class DataSet;


class DataStream : public Stream
{
public:
    enum DataOrder {ORDER_SHUFFLE, ORDER_RANDOM, ORDER_SEQUENTIAL};

    DataStream();

    void SetNumStream(const unsigned long nStream);
    const unsigned long GetNumStream() const;
    const unsigned long GetNumChannel() const;
    const ChannelInfo GetChannelInfo(const unsigned long channelIdx) const;

    void Reset();
    void Next(const unsigned long streamIdx);
    void GenerateFrame(const unsigned long streamIdx, const unsigned long channelIdx, void *const frame);

    void SetDelay(const unsigned long channelIdx, const unsigned long delay);
    void LinkDataSet(DataSet *dataSet);
    void UnlinkDataSet();

    void SetRandomSeed(const unsigned long long seed);
    void SetDataOrder(const DataOrder order);

protected:
    typedef struct
    {
        unsigned long seqIdx;
        unsigned long frameIdx;
    } DataIdx;

    void Alloc();
    const unsigned long GetNewSeqIdx(const unsigned long streamIdx);
    void Shuffle();

    unsigned long nStream;
    unsigned long nChannel;

    std::vector<unsigned long> delay;

    std::vector<unsigned long> dataIdxHistoryIdx;
    std::vector<std::vector<DataIdx>> dataIdxHistory;

    std::vector<unsigned long> shuffledSeqIdx;
    unsigned long nextSeqIdx;

    DataSet *dataSet;

    DataOrder dataOrder;

    std::mt19937_64 randGen;
};

}

#endif /* FRACTAL_DATASTREAM_H_ */

