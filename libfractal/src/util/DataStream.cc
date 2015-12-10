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


#include "DataStream.h"

#include <cstring>

#include "DataSet.h"


namespace fractal
{

DataStream::DataStream()
{
    nStream = 1;
    nChannel = 0;
    dataSet = NULL;
    dataOrder = ORDER_SHUFFLE;

    Alloc();
}


void DataStream::LinkDataSet(DataSet *dataSet)
{
    unsigned long channelIdx;

    this->dataSet = dataSet;

    nChannel = dataSet->GetNumChannel();

    delay.clear();
    delay.shrink_to_fit();
    delay.resize(nChannel);

    for(channelIdx = 0; channelIdx < nChannel; channelIdx++)
    {
        delay[channelIdx] = 0;
    }

    Alloc();
    Reset();
}


void DataStream::UnlinkDataSet()
{
    nChannel = 0;
    dataSet = NULL;

    Alloc();
}


void DataStream::Alloc()
{
    unsigned long streamIdx, channelIdx, maxDelay;

    dataIdxHistoryIdx.clear();
    dataIdxHistory.clear();

    dataIdxHistoryIdx.shrink_to_fit();
    dataIdxHistory.shrink_to_fit();

    dataIdxHistoryIdx.resize(nStream);
    dataIdxHistory.resize(nStream);

    maxDelay = 0;

    for(channelIdx = 0; channelIdx < nChannel; channelIdx++)
    {
        maxDelay = maxDelay >= delay[channelIdx] ? maxDelay : delay[channelIdx];
    }

    for(streamIdx = 0; streamIdx < nStream; streamIdx++)
    {
        dataIdxHistory[streamIdx].resize(maxDelay + 1);
    }
}


void DataStream::SetNumStream(const unsigned long nStream)
{
    verify(nStream > 0);

    this->nStream = nStream;

    Alloc();
    Reset();
}


const unsigned long DataStream::GetNumStream() const
{
    return nStream;
}


const unsigned long DataStream::GetNumChannel() const
{
    return nChannel;
}


const ChannelInfo DataStream::GetChannelInfo(const unsigned long channelIdx) const
{
    verify(channelIdx < nChannel);
    verify(dataSet != NULL);

    return dataSet->GetChannelInfo(channelIdx);
}


void DataStream::Reset()
{
    unsigned long streamIdx, channelIdx;
    unsigned long i, maxDelay;

    for(streamIdx = 0; streamIdx < nStream; streamIdx++)
    {
        dataIdxHistoryIdx[streamIdx] = 0;
    }

    maxDelay = 0;
    for(channelIdx = 0; channelIdx < nChannel; channelIdx++)
    {
        maxDelay = maxDelay >= delay[channelIdx] ? maxDelay : delay[channelIdx];
    }

    switch(dataOrder)
    {
        case(ORDER_SHUFFLE):
            Shuffle();
            nextSeqIdx = 0;
            break;

        case(ORDER_RANDOM):
            break;

        case(ORDER_SEQUENTIAL):
            nextSeqIdx = 0;
            break;

        default:
            verify(false);
    }

    for(streamIdx = 0; streamIdx < nStream; streamIdx++)
    {
        dataIdxHistory[streamIdx][0].seqIdx = GetNewSeqIdx(streamIdx);
        dataIdxHistory[streamIdx][0].frameIdx = 0;

        for(i = 0; i < maxDelay; i++)
        {
            Next(streamIdx);
        }
    }
}


void DataStream::Next(const unsigned long streamIdx)
{
    DataIdx curDataIdx, newDataIdx;
    unsigned long historySize, curDataIdxHistoryIdx, newDataIdxHistoryIdx;

    verify(dataSet != NULL);

    historySize = dataIdxHistory[streamIdx].size();

    curDataIdxHistoryIdx = dataIdxHistoryIdx[streamIdx];
    newDataIdxHistoryIdx = (curDataIdxHistoryIdx + 1) % historySize;
    dataIdxHistoryIdx[streamIdx] = newDataIdxHistoryIdx;

    /* Increase the data index */
    curDataIdx = dataIdxHistory[streamIdx][curDataIdxHistoryIdx];


    if(curDataIdx.frameIdx + 1 == dataSet->GetNumFrame(curDataIdx.seqIdx))
    {
        newDataIdx.seqIdx = GetNewSeqIdx(streamIdx);
        newDataIdx.frameIdx = 0;
    }
    else
    {
        newDataIdx.seqIdx = curDataIdx.seqIdx;
        newDataIdx.frameIdx = curDataIdx.frameIdx + 1;
    }

    dataIdxHistory[streamIdx][newDataIdxHistoryIdx] = newDataIdx;
}


void DataStream::GenerateFrame(const unsigned long streamIdx, const unsigned long channelIdx, void *const frame)
{
    DataIdx delayedDataIdx;
    unsigned long historySize, delayedDataIdxHistoryIdx;

    verify(dataSet != NULL);

    historySize = dataIdxHistory[streamIdx].size();
    delayedDataIdxHistoryIdx = (dataIdxHistoryIdx[streamIdx] + historySize - delay[channelIdx]) % historySize;
    delayedDataIdx = dataIdxHistory[streamIdx][delayedDataIdxHistoryIdx];

    dataSet->GetFrameData(delayedDataIdx.seqIdx, channelIdx, delayedDataIdx.frameIdx, frame);
}


void DataStream::SetDelay(const unsigned long channelIdx, const unsigned long delay)
{
    verify(channelIdx < nChannel);

    this->delay[channelIdx] = delay;

    Alloc();
    Reset();
}


const unsigned long DataStream::GetNewSeqIdx(const unsigned long streamIdx)
{
    unsigned long nSeq, newSeqIdx;

    verify(dataSet != NULL);

    nSeq = dataSet->GetNumSeq();
    verify(nSeq > 0);

    switch(dataOrder)
    {
        case(ORDER_SHUFFLE):

            verify(shuffledSeqIdx.size() == nSeq);

            newSeqIdx = shuffledSeqIdx[nextSeqIdx];
            nextSeqIdx++;
            if(nextSeqIdx >= nSeq)
            {
                Shuffle();
                nextSeqIdx = 0;
            }

            break;

        case(ORDER_RANDOM):
            {
                std::uniform_int_distribution<unsigned long> randDist(0, nSeq - 1);
                newSeqIdx = randDist(randGen);
            }

            break;

        case(ORDER_SEQUENTIAL):

            newSeqIdx = nextSeqIdx;
            nextSeqIdx = (nextSeqIdx + 1) % nSeq;

            break;

        default:
            verify(false);
    }

    verify(dataSet->GetNumFrame(newSeqIdx) > 0);

    return newSeqIdx;
}


void DataStream::Shuffle()
{
    unsigned long nSeq, tmp;

    verify(dataSet != NULL);

    nSeq = dataSet->GetNumSeq();

    if(nSeq != shuffledSeqIdx.size())
    {
        shuffledSeqIdx.resize(nSeq);
        shuffledSeqIdx.shrink_to_fit();
        for(unsigned long i = 0; i < nSeq; i++)
        {
            shuffledSeqIdx[i] = i;
        }
    }

    for(unsigned long i = 0; i < nSeq - 1; i++)
    {
        std::uniform_int_distribution<unsigned long> randDist(i, nSeq - 1);
        unsigned long j = randDist(randGen);

        /* Swap */
        tmp = shuffledSeqIdx[i];
        shuffledSeqIdx[i] = shuffledSeqIdx[j];
        shuffledSeqIdx[j] = tmp;
    }
}


void DataStream::SetRandomSeed(const unsigned long long seed)
{
    randGen.seed(seed);
}


void DataStream::SetDataOrder(const DataOrder order)
{
    dataOrder = order;
}


}

