/*******************************************************************************
 * This file is part of Tissue Forge.
 * Copyright (c) 2022 T.J. Sego
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

/**
 * @file tfIO.h
 * 
 */

#ifndef _SOURCE_IO_TFIO_H_
#define _SOURCE_IO_TFIO_H_

#include <TissueForge_private.h>

#include "tfThreeDFStructure.h"


namespace TissueForge::io {


    /**
     * @brief Tissue Forge import/export interface
     * 
     */

    /**
     * @brief Load a 3D format file
     * 
     * @param filePath path of file
     * @return ThreeDFStructure* 3D format data container
     */
    CPPAPI_FUNC(ThreeDFStructure*) fromFile3DF(const std::string &filePath);

    /**
     * @brief Export engine state to a 3D format file
     * 
     * @param format format of file
     * @param filePath path of file
     * @param pRefinements mesh refinements applied when generating meshes
     * @return HRESULT 
     */
    CPPAPI_FUNC(HRESULT) toFile3DF(const std::string &format, const std::string &filePath, const unsigned int &pRefinements=0);

    /**
     * @brief Save a simulation to file
     * 
     * @param saveFilePath absolute path to file
     * @return HRESULT 
     */
    CPPAPI_FUNC(HRESULT) toFile(const std::string &saveFilePath);

    /**
     * @brief Return a simulation state as a JSON string
     * 
     * @return std::string 
     */
    CPPAPI_FUNC(std::string) toString();

    /**
     * @brief Get the id of a particle according to import data that 
     * corresponds to a particle id of current data. 
     * 
     * Only valid between initialization and the first simulation step, 
     * after which the import summary data is purged. 
     * 
     * @param pId id of particle in exported data
     * @return int >=0 if particle is found; -1 otherwise
     */
    CPPAPI_FUNC(int) mapImportParticleId(const unsigned int &pId);

    /**
     * @brief Get the id of a particle type according to import data that 
     * corresponds to a particle type id of current data. 
     * 
     * Only valid between initialization and the first simulation step, 
     * after which the import summary data is purged. 
     * 
     * @param pId id of particle type in exported data
     * @return int >=0 if particle type is found; -1 otherwise
     */
    CPPAPI_FUNC(int) mapImportParticleTypeId(const unsigned int &pId);


};

#endif // _SOURCE_IO_TFIO_H_