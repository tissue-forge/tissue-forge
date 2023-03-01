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


#include "tfParticleSphereShader.h"

#include <Corrade/Containers/Reference.h>
#include <Corrade/Utility/Resource.h>
#include <Magnum/GL/Shader.h>
#include <Magnum/GL/Version.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Math/Matrix4.h>


#include <assert.h>


namespace TissueForge::shaders { 


    ParticleSphereShader::ParticleSphereShader() {

        assert(Utility::Resource::hasGroup("tfMeshShaderProgram"));


        Utility::Resource rs("tfMeshShaderProgram");

        std::string s = rs.get("tfParticleSphereShader.vert");

        GL::Shader vertShader{GL::Version::GL330, GL::Shader::Type::Vertex};
        GL::Shader fragShader{GL::Version::GL330, GL::Shader::Type::Fragment};
        vertShader.addSource(rs.get("tfParticleSphereShader.vert"));
        fragShader.addSource(rs.get("tfParticleSphereShader.frag"));

        CORRADE_INTERNAL_ASSERT(GL::Shader::compile({vertShader, fragShader}));
        attachShaders({vertShader, fragShader});
        CORRADE_INTERNAL_ASSERT(link());

        _uNumParticles = uniformLocation("numParticles");

        _uPointSizeScale = uniformLocation("pointSizeScale");
        _uColorMode = uniformLocation("colorMode");
        _uAmbientColor = uniformLocation("ambientColor");
        _uDiffuseColor = uniformLocation("diffuseColor");
        _uSpecularColor = uniformLocation("specularColor");
        _uShininess = uniformLocation("shininess");

        _uViewMatrix = uniformLocation("viewMatrix");
        _uProjectionMatrix = uniformLocation("projectionMatrix");
        _uLightDir = uniformLocation("lightDir");
    }

    ParticleSphereShader& ParticleSphereShader::setNumParticles(Int numParticles) {
        setUniform(_uNumParticles, numParticles);
        return *this;
    }


    ParticleSphereShader& ParticleSphereShader::setPointSizeScale(Float scale) {
        setUniform(_uPointSizeScale, scale);
        return *this;
    }

    ParticleSphereShader& ParticleSphereShader::setColorMode(Int colorMode) {
        setUniform(_uColorMode, colorMode);
        return *this;
    }

    ParticleSphereShader& ParticleSphereShader::setAmbientColor(const Color3& color) {
        setUniform(_uAmbientColor, color);
        return *this;
    }

    ParticleSphereShader& ParticleSphereShader::setDiffuseColor(const Color3& color) {
        setUniform(_uDiffuseColor, color);
        return *this;
    }

    ParticleSphereShader& ParticleSphereShader::setSpecularColor(const Color3& color) {
        setUniform(_uSpecularColor, color);
        return *this;
    }

    ParticleSphereShader& ParticleSphereShader::setShininess(Float shininess) {
        setUniform(_uShininess, shininess);
        return *this;
    }

    ParticleSphereShader& ParticleSphereShader::setViewMatrix(const Matrix4& matrix) {
        setUniform(_uViewMatrix, matrix);
        return *this;
    }

    ParticleSphereShader& ParticleSphereShader::setProjectionMatrix(const Matrix4& matrix) {
        setUniform(_uProjectionMatrix, matrix);
        return *this;
    }

    ParticleSphereShader& ParticleSphereShader::setLightDirection(const Vector3& lightDir) {
        setUniform(_uLightDir, lightDir);
        return *this;
    }

}
