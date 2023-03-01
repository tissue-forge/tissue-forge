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

%{

#include <io/tfThreeDFVertexData.h>
#include <io/tfThreeDFEdgeData.h>
#include <io/tfThreeDFFaceData.h>
#include <io/tfThreeDFMeshData.h>
#include <io/tfThreeDFRenderData.h>
#include <io/tfIO.h>
#include <io/tfFIO.h>
#include <io/tf_io.h>

%}


// Declared here to make good use to type hints

%template(vectorThreeDFVertexData_p) std::vector<TissueForge::io::ThreeDFVertexData*>;
%template(vectorThreeDFEdgeData_p)   std::vector<TissueForge::io::ThreeDFEdgeData*>;
%template(vectorThreeDFFaceData_p)   std::vector<TissueForge::io::ThreeDFFaceData*>;
%template(vectorThreeDFMeshData_p)   std::vector<TissueForge::io::ThreeDFMeshData*>;

%ignore TissueForge::io::deleteElement;
%ignore TissueForge::io::toStr;
%ignore TissueForge::io::fromStr;
%ignore TissueForge::io::FIO;
%ignore TissueForge::io::FIOModule;

%rename(_io_ThreeDFVertexData) ThreeDFVertexData;
%rename(_io_ThreeDFEdgeData) ThreeDFEdgeData;
%rename(_io_ThreeDFFaceData) ThreeDFFaceData;
%rename(_io_ThreeDFMeshData) ThreeDFMeshData;
%rename(_io_ThreeDFStructure) ThreeDFStructure;
%rename(_io_fromFile3DF) TissueForge::io::fromFile3DF;
%rename(_io_toFile3DF) TissueForge::io::toFile3DF;
%rename(_io_toFile) TissueForge::io::toFile;
%rename(_io_toString) TissueForge::io::toString;
%rename(_io_mapImportParticleId) TissueForge::io::mapImportParticleId;
%rename(_io_mapImportParticleTypeId) TissueForge::io::mapImportParticleTypeId;
%rename(_io_ThreeDFRenderData) TissueForge::io::ThreeDFRenderData;

%include <io/tfThreeDFRenderData.h>
%include "tfThreeDFVertexData.i"
%include "tfThreeDFEdgeData.i"
%include "tfThreeDFFaceData.i"
%include "tfThreeDFMeshData.i"
%include "tfThreeDFStructure.i"
%include <io/tfIO.h>
%include <io/tfFIO.h>
