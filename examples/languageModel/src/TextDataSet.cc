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


#include "TextDataSet.h"

#include <cstring>
#include <fstream>
#include <cstdint>
#include <cmath>


using namespace fractal;

const unsigned long TextDataSet::CHANNEL_TEXT1 = 0;
const unsigned long TextDataSet::CHANNEL_TEXT2 = 1;
const unsigned long TextDataSet::CHANNEL_SIG_NEWSEQ = 2;


TextDataSet::TextDataSet()
{
    nSeq = 0;
}


void TextDataSet::Split(TextDataSet &target, const double fraction)
{
    unsigned long i, j, k, tmp, n, nSrc, nTarget;

    std::vector<unsigned long> originalNFrame(nFrame);
    std::vector<std::vector<unsigned char>> originalText(text);

    n = text.size();

    nTarget = ((double) n * fraction + 0.5);
    verify(nTarget > 0);

    nSrc = n - nTarget;
    verify(nSrc > 0);

    std::vector<unsigned long> index(n);

    for(i = 0; i < n; i++)
    {
        index[i] = i;
    }

    /* Shuffle */
    for(i = 0; i < n - 1; i++)
    {
        j = i + (rand() % (n - i));

        tmp = index[i];
        index[i] = index[j];
        index[j] = tmp;
    }

    text.clear();
    text.shrink_to_fit();
    text.resize(nSrc);

    nFrame.clear();
    nFrame.shrink_to_fit();
    nFrame.resize(nSrc);

    target.text.clear();
    target.text.shrink_to_fit();
    target.text.resize(nTarget);

    target.nFrame.clear();
    target.nFrame.shrink_to_fit();
    target.nFrame.resize(nTarget);

    for(i = j = k = 0; i < n; i++)
    {
        if(index[i] < nTarget)
        {
            target.text[j] = originalText[i];
            target.nFrame[j] = originalNFrame[i];
            j++;
        }
        else
        {
            text[k] = originalText[i];
            nFrame[k] = originalNFrame[i];
            k++;
        }
    }

    nSeq = nSrc;
    target.nSeq = nTarget;
}


const unsigned long TextDataSet::ReadTextData(const std::string &filename)
{
    std::list<std::string> textList;
    std::ifstream file;
    std::string buf;
    size_t pos1, pos2;

    text.clear();

    file.open(filename, std::ios_base::in);

    verify(file.is_open() == true);

    while(file.eof() == false)
    {
        std::getline(file, buf);
        verify(file.bad() == false);

        pos1 = buf.find_first_not_of(" \n\r\t");
        if(pos1 == std::string::npos) continue;

        pos2 = buf.find_last_not_of(" \n\r\t");

        if(pos2 >= pos1)
        {
            textList.push_back(buf.substr(pos1, pos2 - pos1 + 1) + "\n");
        }
    }

    file.close();

    nSeq = textList.size();
    text.resize(nSeq);
    text.shrink_to_fit();
    nFrame.resize(nSeq);
    nFrame.shrink_to_fit();

    unsigned long i = 0;

    for(auto &tmpStr : textList)
    {
        nFrame[i] = tmpStr.size();
        text[i].resize(nFrame[i]);
        text[i].shrink_to_fit();

        for(size_t pos = 0; pos < nFrame[i]; pos++)
        {
            text[i][pos] = tmpStr.c_str()[pos];
        }

        i++;
    }


    return text.size();
}


const unsigned long TextDataSet::GetNumChannel() const
{
    return 3;
}


const fractal::ChannelInfo TextDataSet::GetChannelInfo(const unsigned long channelIdx) const
{
    ChannelInfo channelInfo;

    switch(channelIdx)
    {
        case CHANNEL_TEXT1:
        case CHANNEL_TEXT2:
            channelInfo.dataType = ChannelInfo::DATATYPE_INDEX;
            channelInfo.frameSize = 1;
            channelInfo.frameDim = 256;
            break;

        case CHANNEL_SIG_NEWSEQ:
            channelInfo.dataType = ChannelInfo::DATATYPE_VECTOR;
            channelInfo.frameSize = 1;
            channelInfo.frameDim = 1;
            break;

        default:
            verify(false);
    }

    return channelInfo;
}


const unsigned long TextDataSet::GetNumSeq() const
{
    return nSeq;
}


const unsigned long TextDataSet::GetNumFrame(const unsigned long seqIdx) const
{
    verify(seqIdx < nSeq);

    return nFrame[seqIdx];
}


void TextDataSet::GetFrameData(const unsigned long seqIdx, const unsigned long channelIdx,
        const unsigned long frameIdx, void *const frame)
{
    verify(seqIdx < nSeq);
    verify(frameIdx < nFrame[seqIdx]);

    switch(channelIdx)
    {
        case CHANNEL_TEXT1:
        case CHANNEL_TEXT2:
            *reinterpret_cast<INT *>(frame) = (INT) (text[seqIdx][frameIdx]);
            break;

        case CHANNEL_SIG_NEWSEQ:
            reinterpret_cast<FLOAT *>(frame)[0] = (FLOAT) (frameIdx == 0);
            break;

        default:
            verify(false);
    }
}

