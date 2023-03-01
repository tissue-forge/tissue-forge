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

#include "tfIO.h"

#include "tfThreeDFIO.h"
#include "tfFIO.h"


namespace TissueForge::io {

    ThreeDFStructure *fromFile3DF(const std::string &filePath) {
        return ThreeDFIO::fromFile(filePath);
    }

    HRESULT toFile3DF(const std::string &format, const std::string &filePath, const unsigned int &pRefinements) {
        return ThreeDFIO::toFile(format, filePath, pRefinements);
    }

    HRESULT toFile(const std::string &saveFilePath) {
        return FIO::toFile(saveFilePath);
    }

    std::string toString() {
        return FIO::toString();
    }

    int mapImportParticleId(const unsigned int &pId) {
        if(FIO::importSummary == NULL) 
            return -1;
        
        auto itr = FIO::importSummary->particleIdMap.find(pId);
        if(itr == FIO::importSummary->particleIdMap.end()) 
            return -1;
        
        return itr->second;
    }

    int mapImportParticleTypeId(const unsigned int &pId) {
        if(FIO::importSummary == NULL) 
            return -1;
        
        auto itr = FIO::importSummary->particleTypeIdMap.find(pId);
        if(itr == FIO::importSummary->particleTypeIdMap.end()) 
            return -1;
        
        return itr->second;
    }

};
