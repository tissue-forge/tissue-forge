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

#ifndef _SOURCE_RENDERING_TFANGLERENDERER_H_
#define _SOURCE_RENDERING_TFANGLERENDERER_H_

#include "tfSubRenderer.h"

#include <shaders/tfFlat3D.h>
#include "tfStyle.h"

#include <Magnum/GL/Mesh.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Mesh.h>


namespace TissueForge::rendering {


    struct AngleRenderer : SubRenderer {
        HRESULT start(const std::vector<fVector4> &clipPlanes) override;
        HRESULT draw(TissueForge::rendering::ArcBallCamera *camera, const iVector2 &viewportSize, const fMatrix4 &modelViewMat) override;
        const unsigned addClipPlaneEquation(const Magnum::Vector4& pe) override;
        const unsigned removeClipPlaneEquation(const unsigned int &id) override;
        void setClipPlaneEquation(unsigned id, const Magnum::Vector4& pe) override;

    private:
        
        std::vector<Magnum::Vector4> _clipPlanes;

        shaders::Flat3D _shader{Corrade::Containers::NoCreate};
        Magnum::GL::Buffer _buffer{Corrade::Containers::NoCreate};
        Magnum::GL::Mesh _mesh{Corrade::Containers::NoCreate};
    };

}

#endif // _SOURCE_RENDERING_TFANGLERENDERER_H_