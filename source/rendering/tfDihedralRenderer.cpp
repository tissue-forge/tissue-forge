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

#include "tfDihedralRenderer.h"

#include <tfDihedral.h>
#include <tfEngine.h>
#include "tfUniverseRenderer.h"
#include <tfLogger.h>


using namespace TissueForge;


HRESULT rendering::DihedralRenderer::start(const std::vector<fVector4> &clipPlanes) {
    // create the shader
    _shader = shaders::Flat3D{
        shaders::Flat3D::Flag::VertexColor, 
        (unsigned int)UniverseRenderer::maxClipPlaneCount()
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

static inline int render_dihedral(rendering::BondsInstanceData* dihedralData, int i, Dihedral *dihedral) {

    if(!(dihedral->flags & DIHEDRAL_ACTIVE)) 
        return 0;
    
    Magnum::Vector3 color = dihedral->style->map_color(dihedral).xyz();
    Particle *pi = _Engine.s.partlist[dihedral->i];
    Particle *pj = _Engine.s.partlist[dihedral->j];
    Particle *pk = _Engine.s.partlist[dihedral->k];
    Particle *pl = _Engine.s.partlist[dihedral->l];
    
    fVector3 pj_origin = FVector3::from(_Engine.s.celllist[pj->id]->origin);
    fVector3 pk_origin = FVector3::from(_Engine.s.celllist[pk->id]->origin);

    int shiftik[3], shiftlj[3];
    fVector3 pixik, pixlj;
    
    int *loci = _Engine.s.celllist[dihedral->i]->loc;
    int *locj = _Engine.s.celllist[dihedral->j]->loc;
    int *lock = _Engine.s.celllist[dihedral->k]->loc;
    int *locl = _Engine.s.celllist[dihedral->l]->loc;
    
    for ( int k = 0 ; k < 3 ; k++ ) {
        shiftik[k] = loci[k] - lock[k];
        shiftlj[k] = locl[k] - locj[k];
        
        if(shiftik[k] > 1) shiftik[k] = -1;
        else if (shiftik[k] < -1) shiftik[k] = 1;
        
        if(shiftlj[k] > 1) shiftlj[k] = -1;
        else if (shiftlj[k] < -1) shiftlj[k] = 1;

        FloatP_t h = _Engine.s.h[k];
        pixik[k] = pi->x[k] + h * shiftik[k];
        pixlj[k] = pl->x[k] + h * shiftlj[k];
    }

    fVector3 posi = pixik + pk_origin;
    fVector3 posj = pj->position + pj_origin;
    fVector3 posk = pk->position + pk_origin;
    fVector3 posl = pixlj + pj_origin;
    
    dihedralData[i].position = posi;
    dihedralData[i].color = color;
    dihedralData[i+1].position = posk;
    dihedralData[i+1].color = color;
    dihedralData[i+2].position = posj;
    dihedralData[i+2].color = color;
    dihedralData[i+3].position = posl;
    dihedralData[i+3].color = color;
    dihedralData[i+4].position = 0.5 * (posi + posk);
    dihedralData[i+4].color = color;
    dihedralData[i+5].position = 0.5 * (posl + posj);
    dihedralData[i+5].color = color;
    return 6;
}

HRESULT rendering::DihedralRenderer::draw(rendering::ArcBallCamera *camera, const iVector2 &viewportSize, const fMatrix4 &modelViewMat) {
    TF_Log(LOG_DEBUG) << "";

    if(_Engine.nr_active_dihedrals > 0) {
        int vertexCount = _Engine.nr_active_dihedrals * 6;
        _mesh.setCount(vertexCount);
        
        _buffer.setData(
            {NULL, vertexCount * sizeof(BondsInstanceData)},
            Magnum::GL::BufferUsage::DynamicDraw
        );
        
        BondsInstanceData* dihedralData = (BondsInstanceData*)(void*)_buffer.map(
           0,
           vertexCount * sizeof(BondsInstanceData),
           Magnum::GL::Buffer::MapFlag::Write|Magnum::GL::Buffer::MapFlag::InvalidateBuffer
        );
        
        int i = 0;
        for(int j = 0; j < _Engine.nr_dihedrals; ++j) {
            Dihedral *dihedral = &_Engine.dihedrals[j];
            i += render_dihedral(dihedralData, i, dihedral);
        }
        assert(i == vertexCount);
        _buffer.unmap();
        
        _shader
        .setTransformationProjectionMatrix(camera->projectionMatrix() * camera->cameraMatrix() * modelViewMat)
        .draw(_mesh);
    }
    return S_OK;
}

const unsigned rendering::DihedralRenderer::addClipPlaneEquation(const Magnum::Vector4& pe) {
    unsigned int id = _clipPlanes.size();
    _clipPlanes.push_back(pe);
    _shader.setclipPlaneEquation(id, pe);
    return id;
}

const unsigned rendering::DihedralRenderer::removeClipPlaneEquation(const unsigned int &id) {
    _clipPlanes.erase(_clipPlanes.begin() + id);

    for(unsigned int i = id; i < _clipPlanes.size(); i++) {
        _shader.setclipPlaneEquation(i, _clipPlanes[i]);
    }

    return _clipPlanes.size();
}

void rendering::DihedralRenderer::setClipPlaneEquation(unsigned id, const Magnum::Vector4& pe) {
    if(id > _shader.clipPlaneCount()) {
        tf_exp(std::invalid_argument("invalid id for clip plane"));
    }

    _shader.setclipPlaneEquation(id, pe);
    _clipPlanes[id] = pe;
}
