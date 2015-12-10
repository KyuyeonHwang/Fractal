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


#ifndef FRACTAL_PORTMAP_H_
#define FRACTAL_PORTMAP_H_


#include <tuple>
#include <list>

#include "../core/Probe.h"
#include "../core/FractalCommon.h"


namespace fractal
{

typedef std::tuple<Probe *, unsigned long> PortMap;
typedef std::list<PortMap> PortMapList;

}

#endif /* FRACTAL_PORTMAP_H_ */

