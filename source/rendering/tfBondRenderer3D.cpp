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

#include "tfBondRenderer3D.h"

#include <Magnum/Trade/MeshData.h>
#include <Magnum/MeshTools/Compile.h>
#include <Magnum/Primitives/Cylinder.h>

#include <tf_fptype.h>
#include <tfBond.h>
#include <tfEngine.h>
#include "tfUniverseRenderer.h"
#include <tfLogger.h>
#include "tfArrowRenderer.h"
#include <tf_metrics.h>


using namespace TissueForge;


HRESULT rendering::BondRenderer3D::start(const std::vector<fVector4> &clipPlanes) {
    
    // create the shaders
    _shader = shaders::Phong {
        shaders::Phong::Flag::VertexColor | 
        shaders::Phong::Flag::InstancedTransformation, 
        1, 0
    };
    _shader.setShininess(2000.0f)
        .setLightPositions({{-20, 40, 20, 0.f}})
        .setLightColors({Magnum::Color3{0.9, 0.9, 0.9}})
        .setShininess(100)
        .setAmbientColor({0.4, 0.4, 0.4, 1})
        .setDiffuseColor({1, 1, 1, 0})
        .setSpecularColor({0.2, 0.2, 0.2, 0});
    
    // create the buffers
    _buffer = Magnum::GL::Buffer();

    // create the meshes
    unsigned int numRings = _detail;
    unsigned int numSegments = _detail;
    _mesh = Magnum::MeshTools::compile(Magnum::Primitives::cylinderSolid(
        numRings, 
        numSegments, 
        0.5
    ));

    _mesh.addVertexBufferInstanced(_buffer, 1, 0, 
        Magnum::Shaders::Phong::TransformationMatrix{}, 
        Magnum::Shaders::Phong::NormalMatrix{}, 
        Magnum::Shaders::Phong::Color4{}
    );

    _mesh.setInstanceCount(0);

    return S_OK;
}

void rendering::render_bond3d(
    Bond3DInstanceData* bondData, 
    const int &idx, 
    const fVector3 &pipos, 
    const fVector3 &pjpos, 
    const Style &s, 
    const float &radius) 
{
    fVector3 pjipos = pjpos - pipos;
    float poslen = pjipos.length();
    Magnum::Vector3 color = s.color;

    fMatrix3 rotationMatrix = rendering::vectorFrameRotation(pjipos);
    Magnum::Matrix4 transformationMatrix = Magnum::Matrix4::from(rotationMatrix, pipos)
        * Magnum::Matrix4::scaling(fVector3(radius, poslen, radius)) 
        * Magnum::Matrix4::translation(fVector3(0.0, 0.5, 0.0));

    bondData[idx].transformationMatrix = transformationMatrix;
    bondData[idx].normalMatrix = transformationMatrix.normalMatrix();
    bondData[idx].color = {color[0], color[1], color[2], (float)s.getVisible()};
}

static inline void _render_bond3d(
    rendering::Bond3DInstanceData* bondData, 
    const int &idx, 
    const unsigned int &i, 
    const unsigned int &j, 
    const rendering::Style &s, 
    const float &radius) 
{

    Particle *pi = _Engine.s.partlist[i];
    Particle *pj = _Engine.s.partlist[j];
    
    const fVector3 pjpos = pj->global_position();
    const fVector3 pipos = pjpos + metrics::relativePosition(pi->global_position(), pjpos);
    const float _radius = radius > 0 ? radius : std::min(pi->radius, pj->radius) * 0.5;

    render_bond3d(bondData, idx, pipos, pjpos, s, _radius);
}

HRESULT rendering::BondRenderer3D::draw(rendering::ArcBallCamera *camera, const iVector2 &viewportSize, const fMatrix4 &modelViewMat) {
    TF_Log(LOG_DEBUG) << "";

    if(_Engine.nr_active_bonds > 0) {
        int vertexCount = _Engine.nr_active_bonds;
        _mesh.setInstanceCount(vertexCount);
        
        _buffer.setData(
            {NULL, vertexCount * sizeof(Bond3DInstanceData)},
            Magnum::GL::BufferUsage::DynamicDraw
        );
        
        // get pointer to data, give me the damned bytes
        Bond3DInstanceData* bondData = (Bond3DInstanceData*)(void*)_buffer.map(
           0,
           vertexCount * sizeof(Bond3DInstanceData),
           Magnum::GL::Buffer::MapFlag::Write|Magnum::GL::Buffer::MapFlag::InvalidateBuffer
        );
        
        int i = 0;
        for(int j = 0; j < _Engine.nr_bonds; ++j) {
            Bond *bond = &_Engine.bonds[j];
            if(bond->flags & BOND_ACTIVE) {
                _render_bond3d(bondData, i, bond->i, bond->j, *bond->style, _radius);
                i++;
            }
        }
        assert(i == _Engine.nr_active_bonds);
        _buffer.unmap();
        
        _shader
            .setProjectionMatrix(camera->projectionMatrix())
            .setTransformationMatrix(camera->cameraMatrix() * modelViewMat)
            .setNormalMatrix(camera->viewMatrix().normalMatrix())
            .draw(_mesh);
    }
    return S_OK;
}

const unsigned rendering::BondRenderer3D::addClipPlaneEquation(const Magnum::Vector4& pe) {
    unsigned int id = _clipPlanes.size();
    _clipPlanes.push_back(pe);
    _shader.setclipPlaneEquation(id, pe);
    return id;
}

const unsigned rendering::BondRenderer3D::removeClipPlaneEquation(const unsigned int &id) {
    _clipPlanes.erase(_clipPlanes.begin() + id);

    for(unsigned int i = id; i < _clipPlanes.size(); i++) {
        _shader.setclipPlaneEquation(i, _clipPlanes[i]);
    }

    return _clipPlanes.size();
}

void rendering::BondRenderer3D::setClipPlaneEquation(unsigned id, const Magnum::Vector4& pe) {
    if(id > _shader.clipPlaneCount()) {
        tf_exp(std::invalid_argument("invalid id for clip plane"));
    }

    _shader.setclipPlaneEquation(id, pe);
    _clipPlanes[id] = pe;
}
