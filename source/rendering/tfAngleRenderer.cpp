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

#include "tfAngleRenderer.h"

#include <tfAngle.h>
#include <tfEngine.h>
#include "tfUniverseRenderer.h"
#include <tfLogger.h>


using namespace TissueForge;


HRESULT rendering::AngleRenderer::start(const std::vector<fVector4> &clipPlanes) {
    
    // create the shader
    _shader = shaders::Flat3D{
        shaders::Flat3D::Flag::VertexColor, 
        (unsigned int)rendering::UniverseRenderer::maxClipPlaneCount()
    };
    
    // create the buffer
    _buffer = Magnum::GL::Buffer{};

    // create the mesh
    _mesh = Magnum::GL::Mesh{};
    _mesh.setPrimitive(Magnum::MeshPrimitive::Lines);
    _mesh.addVertexBuffer(_buffer, 0,
                          shaders::Flat3D::Position{}, 
                          shaders::Flat3D::Color4{});

    return S_OK;
}

static inline int render_angle(rendering::BondsInstanceData* angleData, int i, Angle *angle) {

    if(!(angle->flags & ANGLE_ACTIVE)) 
        return 0;

    const Magnum::Vector4 color = angle->style->map_color(angle);
    Particle *pi = _Engine.s.partlist[angle->i];
    Particle *pj = _Engine.s.partlist[angle->j];
    Particle *pk = _Engine.s.partlist[angle->k];
    
    fVector3 pj_origin = FVector3::from(_Engine.s.celllist[pj->id]->origin);
    
    int shiftij[3], shiftkj[3];
    fVector3 pixij, pixkj;
    
    int *loci = _Engine.s.celllist[angle->i]->loc;
    int *locj = _Engine.s.celllist[angle->j]->loc;
    int *lock = _Engine.s.celllist[angle->k]->loc;
    
    for ( int k = 0 ; k < 3 ; k++ ) {
        int locjk = locj[k];
        shiftij[k] = loci[k] - locjk;
        shiftkj[k] = lock[k] - locjk;
        
        if(shiftij[k] > 1) shiftij[k] = -1;
        else if (shiftij[k] < -1) shiftij[k] = 1;
        
        if(shiftkj[k] > 1) shiftkj[k] = -1;
        else if (shiftkj[k] < -1) shiftkj[k] = 1;

        FloatP_t h = _Engine.s.h[k];
        pixij[k] = pi->x[k] + h * shiftij[k];
        pixkj[k] = pk->x[k] + h * shiftkj[k];
    }

    fVector3 posi = pixij + pj_origin;
    fVector3 posj = pj->position + pj_origin;
    fVector3 posk = pixkj + pj_origin;
    
    angleData[i].position = posi;
    angleData[i].color = color;
    angleData[i+1].position = posj;
    angleData[i+1].color = color;
    
    angleData[i+2].position = posk;
    angleData[i+2].color = color;
    angleData[i+3] = angleData[i+1];

    angleData[i+4].position = 0.5 * (posi + posj);
    angleData[i+4].color = color;
    angleData[i+5].position = 0.5 * (posk + posj);
    angleData[i+5].color = color;
    return 6;
}

HRESULT rendering::AngleRenderer::draw(rendering::ArcBallCamera *camera, const iVector2 &viewportSize, const fMatrix4 &modelViewMat) {
    TF_Log(LOG_DEBUG) << "";

    if(_Engine.nr_active_angles > 0) {
        int vertexCount = _Engine.nr_active_angles * 6;
        _mesh.setCount(vertexCount);
        
        _buffer.setData(
            {NULL, vertexCount * sizeof(BondsInstanceData)},
            GL::BufferUsage::DynamicDraw
        );
        
        BondsInstanceData* angleData = (BondsInstanceData*)(void*)_buffer.map(
           0,
           vertexCount * sizeof(BondsInstanceData),
           GL::Buffer::MapFlag::Write|GL::Buffer::MapFlag::InvalidateBuffer
        );
        
        int i = 0;
        for(int j = 0; j < _Engine.nr_angles; ++j) {
            Angle *angle = &_Engine.angles[j];
            i += render_angle(angleData, i, angle);
        }
        assert(i == vertexCount);
        _buffer.unmap();
        
        _shader
        .setTransformationProjectionMatrix(camera->projectionMatrix() * camera->cameraMatrix() * modelViewMat)
        .draw(_mesh);
    }
    return S_OK;
}

const unsigned rendering::AngleRenderer::addClipPlaneEquation(const Magnum::Vector4& pe) {
    unsigned int id = _clipPlanes.size();
    _clipPlanes.push_back(pe);
    _shader.setclipPlaneEquation(id, pe);
    return id;
}

const unsigned rendering::AngleRenderer::removeClipPlaneEquation(const unsigned int &id) {
    _clipPlanes.erase(_clipPlanes.begin() + id);

    for(unsigned int i = id; i < _clipPlanes.size(); i++) {
        _shader.setclipPlaneEquation(i, _clipPlanes[i]);
    }

    return _clipPlanes.size();
}

void rendering::AngleRenderer::setClipPlaneEquation(unsigned id, const Magnum::Vector4& pe) {
    if(id > _shader.clipPlaneCount()) {
        tf_exp(std::invalid_argument("invalid id for clip plane"));
    }

    _shader.setclipPlaneEquation(id, pe);
    _clipPlanes[id] = pe;
}
