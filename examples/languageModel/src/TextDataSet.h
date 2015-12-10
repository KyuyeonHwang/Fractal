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


#ifndef __TEXTDATASET_H__
#define __TEXTDATASET_H__

#include <vector>
#include <string>
#include <list>

#include <fractal/fractal.h>



class TextDataSet : public fractal::DataSet
{
public:
    static const unsigned long CHANNEL_TEXT1;
    static const unsigned long CHANNEL_TEXT2;
    static const unsigned long CHANNEL_SIG_NEWSEQ;

    TextDataSet();

    void Split(TextDataSet &target, const double fraction);

    const unsigned long ReadTextData(const std::string &filename);

    const unsigned long GetNumChannel() const;
    const fractal::ChannelInfo GetChannelInfo(const unsigned long channelIdx) const;
    const unsigned long GetNumSeq() const;
    const unsigned long GetNumFrame(const unsigned long seqIdx) const;

    void GetFrameData(const unsigned long seqIdx, const unsigned long channelIdx,
            const unsigned long frameIdx, void *const frame);

protected:
    unsigned long nSeq;

    std::vector<unsigned long> nFrame;

    std::vector<std::vector<unsigned char>> text;
};


#endif /* __TEXTDATASET_H__ */

