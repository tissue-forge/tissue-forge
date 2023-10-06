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

#include "tfDihedralRenderer3D.h"

#include <Magnum/Trade/MeshData.h>
#include <Magnum/MeshTools/Compile.h>
#include <Magnum/Primitives/Cylinder.h>
#include <Magnum/Primitives/Square.h>

#include "tfBondRenderer3D.h"
#include <tfDihedral.h>
#include <tfEngine.h>
#include "tfUniverseRenderer.h"
#include <tfLogger.h>
#include <tf_metrics.h>


using namespace TissueForge;


HRESULT rendering::DihedralRenderer3D::start(const std::vector<fVector4> &clipPlanes) {

    // create the shader
    _shader = shaders::Phong {
        shaders::Phong::Flag::VertexColor | 
        shaders::Phong::Flag::InstancedTransformation, 
        1, 
        (unsigned int)rendering::UniverseRenderer::maxClipPlaneCount()
    };
    _shader.setShininess(2000.0f)
        .setLightPositions({{-20, 40, 20, 0.f}})
        .setLightColors({Magnum::Color3{0.9, 0.9, 0.9}})
        .setShininess(100)
        .setAmbientColor({0.4, 0.4, 0.4, 1})
        .setDiffuseColor({1, 1, 1, 0})
        .setSpecularColor({0.2, 0.2, 0.2, 0});
    
    // create the buffers
    _bufferBnds = Magnum::GL::Buffer();
    _bufferPlns = Magnum::GL::Buffer();

    // create the meshes
    unsigned int numRings = _detailBnds;
    unsigned int numSegments = _detailBnds;
    _meshBnds = Magnum::MeshTools::compile(Magnum::Primitives::cylinderSolid(
        numRings, 
        numSegments, 
        0.5
    ));

    _meshPlns = Magnum::MeshTools::compile(Magnum::Primitives::squareSolid());

    _meshBnds.addVertexBufferInstanced(_bufferBnds, 1, 0, 
        Magnum::Shaders::Phong::TransformationMatrix{}, 
        Magnum::Shaders::Phong::NormalMatrix{}, 
        Magnum::Shaders::Phong::Color4{}
    );
    _meshPlns.addVertexBufferInstanced(_bufferPlns, 1, 0, 
        Magnum::Shaders::Phong::TransformationMatrix{}, 
        Magnum::Shaders::Phong::NormalMatrix{}, 
        Magnum::Shaders::Phong::Color4{}
    );

    _meshBnds.setInstanceCount(0);
    _meshPlns.setInstanceCount(0);

    return S_OK;
}

void rendering::render_plane3d(
    rendering::Plane3DInstanceData *planeData, 
    const unsigned int &idx, 
    const fVector3 &posi, 
    const fVector3 &posj, 
    const fVector3 &posk, 
    const fVector4 &color) 
{
    fMatrix4 transformationMatrix = fMatrix4::from(fMatrix3(posk - posj, posi - posj, posk - posj), posj);
    transformationMatrix = transformationMatrix * fMatrix4::translation(fVector3(0.5, 0.5, 0)) * fMatrix4::scaling(fVector3(0.5));
    planeData[idx].transformationMatrix = transformationMatrix;
    planeData[idx].normalMatrix = transformationMatrix.normalMatrix();
    planeData[idx].color = color;
}

HRESULT rendering::DihedralRenderer3D::draw(rendering::ArcBallCamera *camera, const iVector2 &viewportSize, const fMatrix4 &modelViewMat) {
    TF_Log(LOG_DEBUG) << "";

    if(_Engine.nr_active_dihedrals > 0) {
        int countBnds = _Engine.nr_active_dihedrals * 3;
        int countPlns = _Engine.nr_active_dihedrals * 2;
        _meshBnds.setInstanceCount(countBnds);
        _meshPlns.setInstanceCount(countPlns);

        _bufferBnds.setData(
            {NULL, countBnds * sizeof(Bond3DInstanceData)},
            Magnum::GL::BufferUsage::DynamicDraw
        );
        _bufferPlns.setData(
            {NULL, countPlns * sizeof(Plane3DInstanceData)}, 
            Magnum::GL::BufferUsage::DynamicDraw
        );
        
        Bond3DInstanceData* dataBnds = (Bond3DInstanceData*)(void*)_bufferBnds.map(
           0,
           countBnds * sizeof(Bond3DInstanceData),
           Magnum::GL::Buffer::MapFlag::Write|Magnum::GL::Buffer::MapFlag::InvalidateBuffer
        );
        Plane3DInstanceData* dataPlns = (Plane3DInstanceData*)(void*)_bufferPlns.map(
            0, 
            countPlns * sizeof(Plane3DInstanceData), 
            Magnum::GL::Buffer::MapFlag::Write|Magnum::GL::Buffer::MapFlag::InvalidateBuffer
        );
        
        int i = 0;
        float radius;
        fVector3 papos, pbpos;
        for(int j = 0; j < _Engine.nr_dihedrals; ++j) {
            Dihedral *bond = &_Engine.dihedrals[j];
            if(bond->flags & DIHEDRAL_ACTIVE) {
                Particle *pi = _Engine.s.partlist[bond->i];
                Particle *pj = _Engine.s.partlist[bond->j];
                Particle *pk = _Engine.s.partlist[bond->k];
                Particle *pl = _Engine.s.partlist[bond->l];

                radius = _radiusBnds > 0 ? _radiusBnds : std::min(std::min(std::min(pi->radius, pj->radius), pk->radius), pl->radius) * 0.3;

                const fVector3 pjpos = pj->global_position();
                const fVector3 pipos_relj = pjpos + metrics::relativePosition(pi->global_position(), pjpos);
                const fVector3 pkpos = pk->global_position();
                const fVector3 plpos = pl->global_position();
                const fVector3 pjkpos = metrics::relativePosition(pjpos, pkpos);
                const fVector3 pkpos_relj = pjpos - pjkpos;
                const fVector4 color = bond->style->map_color(bond);
                const fVector4 color_plane = {color[0], color[1], color[2], 0.5f * color[3]};

                auto _dataPlns = &dataPlns[i * 2];
                rendering::render_plane3d(_dataPlns, 0, 
                    pipos_relj, 
                    pjpos, 
                    pkpos_relj, 
                    color_plane
                );
                rendering::render_plane3d(_dataPlns, 1,
                    pjpos,
                    pkpos_relj,
                    pjpos + metrics::relativePosition(plpos, pjpos),
                    color_plane
                );
                auto _dataBnds = &dataBnds[i * 3];
                rendering::render_bond3d(_dataBnds, 0,
                    pipos_relj,
                    pjpos,
                    color, radius
                );
                rendering::render_bond3d(_dataBnds, 1,
                    pkpos + pjkpos,
                    pkpos,
                    color, radius
                );
                rendering::render_bond3d(_dataBnds, 2,
                    pkpos,
                    pkpos + metrics::relativePosition(plpos, pkpos),
                    color, radius
                );
                i++;
            }
        }
        assert(i == _Engine.nr_active_dihedrals);
        _bufferBnds.unmap();
        _bufferPlns.unmap();
        
        _shader
            .setProjectionMatrix(camera->projectionMatrix())
            .setTransformationMatrix(camera->cameraMatrix() * modelViewMat)
            .setNormalMatrix(camera->viewMatrix().normalMatrix());
        _shader.draw(_meshBnds);
        _shader.draw(_meshPlns);
    }
    return S_OK;
}

const unsigned rendering::DihedralRenderer3D::addClipPlaneEquation(const Magnum::Vector4& pe) {
    unsigned int id = _clipPlanes.size();
    _clipPlanes.push_back(pe);
    _shader.setclipPlaneEquation(id, pe);
    return id;
}

const unsigned rendering::DihedralRenderer3D::removeClipPlaneEquation(const unsigned int &id) {
    _clipPlanes.erase(_clipPlanes.begin() + id);

    for(unsigned int i = id; i < _clipPlanes.size(); i++) {
        _shader.setclipPlaneEquation(i, _clipPlanes[i]);
    }

    return _clipPlanes.size();
}

void rendering::DihedralRenderer3D::setClipPlaneEquation(unsigned id, const Magnum::Vector4& pe) {
    if(id > _shader.clipPlaneCount()) {
        tf_exp(std::invalid_argument("invalid id for clip plane"));
    }

    _shader.setclipPlaneEquation(id, pe);
    _clipPlanes[id] = pe;
}
