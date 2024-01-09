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

/*
Derived from Magnum with the following notice:

    Original authors — credit is appreciated but not required:

        2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019 —
            Vladimír Vondruš <mosra@centrum.cz>
        2019 — Nghia Truong <nghiatruong.vn@gmail.com>

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2.1 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.
 */

#ifndef _SOURCE_SHADERS_TFPARTICLESPHERESHADER_H_
#define _SOURCE_SHADERS_TFPARTICLESPHERESHADER_H_

#include <tf_port.h>
#include <Magnum/GL/AbstractShaderProgram.h>
#include <Magnum/Math/Vector3.h>


using namespace Magnum;


namespace TissueForge::shaders { 


    class CAPI_EXPORT ParticleSphereShader: public GL::AbstractShaderProgram {
    public:

        struct Vertex {
            Magnum::Vector3 pos;
            Magnum::Int index;
            Magnum::Float radius;
        };

        typedef Magnum::GL::Attribute<0, Magnum::Vector3> Position;
        typedef Magnum::GL::Attribute<1, Magnum::Int> Index;
        typedef Magnum::GL::Attribute<2, Magnum::Float> Radius;


        enum ColorMode {
            UniformDiffuseColor = 0,
            RampColorById,
            ConsistentRandom
        };

        explicit ParticleSphereShader();

        ParticleSphereShader& setNumParticles(Int numParticles);

        ParticleSphereShader& setPointSizeScale(Float scale);
        ParticleSphereShader& setColorMode(Int colorMode);
        ParticleSphereShader& setAmbientColor(const Color3& color);
        ParticleSphereShader& setDiffuseColor(const Color3& color);
        ParticleSphereShader& setSpecularColor(const Color3& color);
        ParticleSphereShader& setShininess(Float shininess);

        ParticleSphereShader& setViewport(const Vector2i& viewport);
        ParticleSphereShader& setViewMatrix(const Matrix4& matrix);
        ParticleSphereShader& setProjectionMatrix(const Matrix4& matrix);
        ParticleSphereShader& setLightDirection(const Vector3& lightDir);

    private:
        Int _uNumParticles,
        _uPointSizeScale,
        _uColorMode,
        _uAmbientColor,
        _uDiffuseColor,
        _uSpecularColor,
        _uShininess,
        _uViewMatrix,
        _uProjectionMatrix,
        _uLightDir;
    };

}

#endif // _SOURCE_SHADERS_TFPARTICLESPHERESHADER_H_