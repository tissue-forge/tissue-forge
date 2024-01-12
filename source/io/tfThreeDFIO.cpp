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

#include <assimp/postprocess.h>

#include <tfEngine.h>
#include <rendering/tfStyle.h>
#include <tfParticleList.h>
#include "generators/tfThreeDFAngleMeshGenerator.h"
#include "generators/tfThreeDFBondMeshGenerator.h"
#include "generators/tfThreeDFDihedralMeshGenerator.h"
#include "generators/tfThreeDFPCloudMeshGenerator.h"

#include "tfThreeDFIO.h"


namespace TissueForge::io {


    ThreeDFStructure *ThreeDFIO::fromFile(const std::string &filePath) {
        ThreeDFStructure *result = new ThreeDFStructure();
        
        if(result->fromFile(filePath) != S_OK) 
            return NULL;

        return result;
    }

    ThreeDFMeshData *generate3DFMeshByType(ParticleType *pType, const unsigned int &pRefinements) {

        ThreeDFPCloudMeshGenerator generatorPCloud;
        generatorPCloud.pList = ParticleList(pType->items());
        generatorPCloud.pRefinements = pRefinements;

        if(generatorPCloud.pList.nr_parts == 0 || generatorPCloud.process() != S_OK) 
            return 0;

        auto mesh = generatorPCloud.getMesh();
        mesh->name += " (" + std::string(pType->name) + ")";

        mesh->renderData = new ThreeDFRenderData();
        mesh->renderData->color = pType->style->color;
        
        return mesh;

    }

    HRESULT ThreeDFIO::toFile(const std::string &format, const std::string &filePath, const unsigned int &pRefinements) {

        // Build structure

        ThreeDFStructure structure;

        // Generate point cloud mesh by particle type
        
        if(_Engine.s.nr_parts > 0) 
            for(unsigned int i = 0; i < _Engine.nr_types; i++) {
                auto mesh = generate3DFMeshByType(&_Engine.types[i], pRefinements);
                if(mesh != NULL) 
                    structure.add(mesh);
            }

        // Generate bond mesh

        ThreeDFBondMeshGenerator generatorBonds;
        generatorBonds.pRefinements = pRefinements;

        if(_Engine.nr_bonds > 0) {
            
            generatorBonds.bonds.reserve(_Engine.nr_bonds);

            for(unsigned int i = 0; i < _Engine.nr_bonds; i++) {

                auto b = _Engine.bonds[i];
                if(b.flags & BOND_ACTIVE)
                    generatorBonds.bonds.push_back(BondHandle(b.id));

            }

            if(generatorBonds.process() == S_OK) 
                structure.add(generatorBonds.getMesh());

        }

        // Generate angle mesh

        ThreeDFAngleMeshGenerator generatorAngles;
        generatorAngles.pRefinements = pRefinements;

        if(_Engine.nr_angles > 0) {

            generatorAngles.angles.reserve(_Engine.nr_angles);

            for(unsigned int i = 0; i < _Engine.nr_angles; i++) {

                auto a = _Engine.angles[i];

                if(a.flags & ANGLE_ACTIVE) 
                    generatorAngles.angles.push_back(AngleHandle(i));

            }

            if(generatorAngles.process() == S_OK) 
                structure.add(generatorAngles.getMesh());

        }

        // Generate dihedral mesh

        ThreeDFDihedralMeshGenerator generatorDihedrals;
        generatorDihedrals.pRefinements = pRefinements;

        if(_Engine.nr_dihedrals > 0) {

            generatorDihedrals.dihedrals.reserve(_Engine.nr_dihedrals);

            for(unsigned int i = 0; i < _Engine.nr_dihedrals; i++) {

                generatorDihedrals.dihedrals.push_back(DihedralHandle(i));
                
            }

            if(generatorDihedrals.process() == S_OK) 
                structure.add(generatorDihedrals.getMesh());

        }

        // Export

        if(structure.toFile(format, filePath) != S_OK) 
            return E_FAIL;

        if(structure.clear() != S_OK) 
            return E_FAIL;

        return S_OK;
    }

};
