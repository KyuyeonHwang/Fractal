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


#ifndef FRACTAL_FRACTALCOMMON_H_
#define FRACTAL_FRACTALCOMMON_H_

#ifdef HAVE_CONFIG_H
    #include <config.h>
#endif /* HAVE_CONFIG_H */

#include <cstddef> /* for the definition of NULL */

namespace fractal
{

#ifdef FRACTAL_DOUBLE_PRECISION
    typedef double FLOAT;
#else
#define FRACTAL_SINGLE_PRECISION
    typedef float FLOAT;
#endif /* FRACTAL_DOUBLE_PRECISION */

    typedef long INT;

}

#ifdef NDEBUG
    #define verify(expression) ((void)(expression))
#else
    #include <cassert>
    #define verify(expression) assert(expression)
#endif /* NDEBUG */


#endif /* FRACTAL_FRACTALCOMMON_H_ */

