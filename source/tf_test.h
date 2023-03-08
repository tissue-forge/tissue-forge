/*******************************************************************************
 * This file is part of Tissue Forge.
 * Copyright (c) 2022, 2023 T.J. Sego
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * 
 ******************************************************************************/

#ifndef _SOURCE_TF_TEST_H_
#define _SOURCE_TF_TEST_H_

#include "TissueForge_private.h"


namespace TissueForge { 
    
    
    namespace test {


        typedef std::tuple<char*, size_t> testImage_t;
        CPPAPI_FUNC(testImage_t) testImage();

        typedef std::unordered_map<std::string, std::string> testHeadless_t;
        CPPAPI_FUNC(testHeadless_t) testHeadless();

}};

#endif // _SOURCE_TF_TEST_H_