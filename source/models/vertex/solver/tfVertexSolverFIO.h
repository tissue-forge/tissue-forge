/*******************************************************************************
 * This file is part of Tissue Forge.
 * Copyright (c) 2022 T.J. Sego and Tien Comlekoglu
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
 * @file tfVertexSolverFIO.h
 * 
 */

#ifndef _MODELS_VERTEX_SOLVER_TFVERTEXSOLVERFIO_H_
#define _MODELS_VERTEX_SOLVER_TFVERTEXSOLVERFIO_H_

#include <tf_port.h>

#include <io/tfFIO.h>

#include <vector>


namespace TissueForge::models::vertex::io {


    /**
     * @brief Vertex solver module data import summary. 
     * 
     * Not every datum of an imported Tissue Forge simulation state is conserved 
     * (e.g., object id). This class provides the information necessary 
     * to exactly translate the imported data of a previously exported simulation. 
     * 
     */
    struct CAPI_EXPORT VertexSolverFIOImportSummary {

        /** Map of imported vertex ids to loaded vertex ids */
        std::unordered_map<unsigned int, unsigned int> vertexIdMap;

        /** Map of imported surface ids to loaded surface ids */
        std::unordered_map<unsigned int, unsigned int> surfaceIdMap;

        /** Map of imported body ids to loaded body ids */
        std::unordered_map<unsigned int, unsigned int> bodyIdMap;

        /** Map of imported surface type ids to loaded surface ids */
        std::unordered_map<unsigned int, unsigned int> surfaceTypeIdMap;

        /** Map of imported body type ids to loaded body ids */
        std::unordered_map<unsigned int, unsigned int> bodyTypeIdMap;

    };

    struct VertexSolverFIOModule : TissueForge::io::FIOModule {

        /** Import summary of most recent import */
        inline static VertexSolverFIOImportSummary *importSummary = NULL;

        std::string moduleName() override;
        HRESULT toFile(const TissueForge::io::MetaData &metaData, TissueForge::io::IOElement *fileElement) override;
        HRESULT fromFile(const TissueForge::io::MetaData &metaData, const TissueForge::io::IOElement &fileElement) override;

        /**
        * @brief Test whether imported data is available. 
        */
        static bool hasImport();

        /**
        * @brief Clear imported data, if available. 
        */
        static HRESULT clearImport();

    };

}

#endif // _MODELS_VERTEX_SOLVER_TFVERTEXSOLVERFIO_H_