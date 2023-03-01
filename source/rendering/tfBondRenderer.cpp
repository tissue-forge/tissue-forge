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

#include "tfBondRenderer.h"

#include <tf_fptype.h>
#include <tfBond.h>
#include <tfEngine.h>
#include "tfUniverseRenderer.h"
#include <tfLogger.h>


using namespace TissueForge;


HRESULT rendering::BondRenderer::start(const std::vector<fVector4> &clipPlanes) {
    
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
                          shaders::Flat3D::Color3{});

    return S_OK;
}

static inline int render_bond(rendering::BondsInstanceData* bondData, int i, Bond *bond) {

    if(!(bond->flags & BOND_ACTIVE)) 
        return 0;

    Magnum::Vector3 *color = &bond->style->color;
    Particle *pi = _Engine.s.partlist[bond->i];
    Particle *pj = _Engine.s.partlist[bond->j];
    
    fVector3 pj_origin = FVector3::from(_Engine.s.celllist[pj->id]->origin);
    
    int shift[3];
    fVector3 pix;
    
    int *loci = _Engine.s.celllist[ bond->i ]->loc;
    int *locj = _Engine.s.celllist[ bond->j ]->loc;
    
    for ( int k = 0 ; k < 3 ; k++ ) {
        shift[k] = loci[k] - locj[k];
        if ( shift[k] > 1 )
            shift[k] = -1;
        else if ( shift[k] < -1 )
            shift[k] = 1;
        pix[k] = pi->x[k] + _Engine.s.h[k]* shift[k];
    }
                    
    bondData[i].position = pix + pj_origin;
    bondData[i].color = *color;
    bondData[i+1].position = fVector3(pj->position + pj_origin);
    bondData[i+1].color = *color;
    return 2;
}

HRESULT rendering::BondRenderer::draw(rendering::ArcBallCamera *camera, const iVector2 &viewportSize, const fMatrix4 &modelViewMat) {
    TF_Log(LOG_DEBUG) << "";

    if(_Engine.nr_active_bonds > 0) {
        int vertexCount = _Engine.nr_active_bonds * 2;
        _mesh.setCount(vertexCount);
        
        _buffer.setData(
            {NULL, vertexCount * sizeof(rendering::BondsInstanceData)},
            Magnum::GL::BufferUsage::DynamicDraw
        );
        
        // get pointer to data, give me the damned bytes
        rendering::BondsInstanceData* bondData = (rendering::BondsInstanceData*)(void*)_buffer.map(
           0,
           vertexCount * sizeof(rendering::BondsInstanceData),
           Magnum::GL::Buffer::MapFlag::Write|Magnum::GL::Buffer::MapFlag::InvalidateBuffer
        );
        
        int i = 0;
        Magnum::Vector3 *color;
        for(int j = 0; j < _Engine.nr_bonds; ++j) {
            Bond *bond = &_Engine.bonds[j];
            i += render_bond(bondData, i, bond);
        }
        assert(i == 2 * _Engine.nr_active_bonds);
        _buffer.unmap();
        
        _shader
        .setTransformationProjectionMatrix(camera->projectionMatrix() * camera->cameraMatrix() * modelViewMat)
        .draw(_mesh);
    }
    return S_OK;
}

const unsigned rendering::BondRenderer::addClipPlaneEquation(const Magnum::Vector4& pe) {
    unsigned int id = _clipPlanes.size();
    _clipPlanes.push_back(pe);
    _shader.setclipPlaneEquation(id, pe);
    return id;
}

const unsigned rendering::BondRenderer::removeClipPlaneEquation(const unsigned int &id) {
    _clipPlanes.erase(_clipPlanes.begin() + id);

    for(unsigned int i = id; i < _clipPlanes.size(); i++) {
        _shader.setclipPlaneEquation(i, _clipPlanes[i]);
    }

    return _clipPlanes.size();
}

void rendering::BondRenderer::setClipPlaneEquation(unsigned id, const Magnum::Vector4& pe) {
    if(id > _shader.clipPlaneCount()) {
        tf_exp(std::invalid_argument("invalid id for clip plane"));
    }

    _shader.setclipPlaneEquation(id, pe);
    _clipPlanes[id] = pe;
}
