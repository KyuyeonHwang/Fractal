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


#ifndef FRACTAL_CHANNELINFO_H_
#define FRACTAL_CHANNELINFO_H_

#include "../core/FractalCommon.h"

namespace fractal
{

class ChannelInfo
{
public:
    enum DataType {DATATYPE_UNDEFINED, DATATYPE_VECTOR, DATATYPE_INDEX, DATATYPE_SEQ};

    ChannelInfo() : dataType(DATATYPE_UNDEFINED), frameSize(0), frameDim(0) {}

    DataType dataType;
    unsigned long frameSize;
    unsigned long frameDim;
};

}

#endif /* FRACTAL_CHANNELINFO_H_ */

