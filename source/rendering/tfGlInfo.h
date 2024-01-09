/*******************************************************************************
 * This file is part of Tissue Forge.
 * Copyright (c) 2022-2024 T.J. Sego
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

#ifndef _SOURCE_RENDERING_TFGLINFO_H_
#define _SOURCE_RENDERING_TFGLINFO_H_

#include <TissueForge.h>

#include <string>
#include <unordered_map>
#include <vector>


namespace TissueForge {


    namespace rendering {


        class CAPI_EXPORT GLInfo {

        public:

            GLInfo() {};
            ~GLInfo() {};

            static const std::unordered_map<std::string, std::string> getInfo();
            static const std::vector<std::string> getExtensionsInfo();

        };

        std::unordered_map<std::string, std::string> glInfo();

        std::string eglInfo();

        std::string gl_info();

}}

#endif /* _SOURCE_RENDERING_TFGLINFO_H_ */