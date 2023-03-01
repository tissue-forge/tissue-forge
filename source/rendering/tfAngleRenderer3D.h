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

#ifndef _SOURCE_RENDERING_TFANGLERENDERER3D_H_
#define _SOURCE_RENDERING_TFANGLERENDERER3D_H_

#include "tfSubRenderer.h"

#include <shaders/tfPhong.h>
#include "tfStyle.h"

#include <Magnum/GL/Mesh.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Mesh.h>


namespace TissueForge::rendering {


    struct Angle3DInstanceData {
        Magnum::Vector3 position;
        Magnum::Vector3 normal;
        Magnum::Color4 color;
    };

    struct AngleRenderer3D : SubRenderer {
        HRESULT start(const std::vector<fVector4> &clipPlanes) override;
        HRESULT draw(TissueForge::rendering::ArcBallCamera *camera, const iVector2 &viewportSize, const fMatrix4 &modelViewMat) override;
        const unsigned addClipPlaneEquation(const Magnum::Vector4& pe) override;
        const unsigned removeClipPlaneEquation(const unsigned int &id) override;
        void setClipPlaneEquation(unsigned id, const Magnum::Vector4& pe) override;

    private:

        int _detailBnds = 10;
        float _radiusBnds = -0.1;
        int _segmentsArcs = 20;
        std::vector<Magnum::Vector4> _clipPlanes;

        shaders::Phong _shaderArcs{Corrade::Containers::NoCreate};
        shaders::Phong _shaderBnds{Corrade::Containers::NoCreate};
        Magnum::GL::Buffer _bufferArcs{Corrade::Containers::NoCreate};
        Magnum::GL::Buffer _bufferBnds{Corrade::Containers::NoCreate};
        Magnum::GL::Mesh _meshArcs{Corrade::Containers::NoCreate};
        Magnum::GL::Mesh _meshBnds{Corrade::Containers::NoCreate};
    };

    void render_arc3d(
        rendering::Angle3DInstanceData *arcData, 
        const unsigned int &idx, 
        const fVector3 &posi, 
        const fVector3 &posj, 
        const fVector3 &posk, 
        const rendering::Style &s, 
        const unsigned int &numSegments, 
        const unsigned int &faceDetail, 
        const float &radius
    );

}

#endif // _SOURCE_RENDERING_TFANGLERENDERER3D_H_