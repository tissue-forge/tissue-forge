/*******************************************************************************
 * This file is part of Tissue Forge.
 * Copyright (c) 2022-2024 T.J. Sego and Tien Comlekoglu
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

#ifndef _MODELS_VERTEX_SOLVER_TFMESHRENDERER_H_
#define _MODELS_VERTEX_SOLVER_TFMESHRENDERER_H_

#include <rendering/tfSubRenderer.h>
#include <shaders/tfFlat3D.h>

#include <Magnum/GL/GL.h>
#include <Magnum/GL/Buffer.h>
#include <Magnum/GL/Mesh.h>
#include <Magnum/Shaders/Shaders.h>


namespace TissueForge::models::vertex { 


    struct MeshRenderer : rendering::SubRenderer {

        /**
         * Get the mesh renderer.
         * 
         * If a mesh renderer does not yet exist, then it is created. 
         */
        static MeshRenderer *get();
        
        HRESULT start(const std::vector<fVector4> &clipPlanes) override;

        HRESULT draw(rendering::ArcBallCamera *camera, const iVector2 &viewportSize, const fMatrix4 &modelViewMat) override;

    private:

        std::vector<Magnum::Vector4> _clipPlanes;

        Magnum::GL::Buffer _bufferFaces{Corrade::Containers::NoCreate};
        Magnum::GL::Buffer _bufferEdges{Corrade::Containers::NoCreate};
        Magnum::GL::Mesh _meshFaces{Corrade::Containers::NoCreate};
        Magnum::GL::Mesh _meshEdges{Corrade::Containers::NoCreate};
        shaders::Flat3D _shaderFaces{Corrade::Containers::NoCreate};
        shaders::Flat3D _shaderEdges{Corrade::Containers::NoCreate};

    };

}

#endif // _MODELS_VERTEX_SOLVER_TFMESHRENDERER_H_