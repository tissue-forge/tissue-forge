/*******************************************************************************
 * This file is part of Tissue Forge.
 * Copyright (c) 2022 T.J. Sego and Tien Comlekoglu
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

#include "tfMeshRenderer.h"

#include "tfSurface.h"
#include "tfVertex.h"
#include "tfMeshSolver.h"

#include <rendering/tfUniverseRenderer.h>
#include <rendering/tfStyle.h>
#include <tfEngine.h>
#include <tfLogger.h>
#include <tf_system.h>

#include <Magnum/Mesh.h>


using namespace TissueForge;
using namespace TissueForge::models::vertex;


static MeshRenderer *_meshRenderer = NULL;


struct MeshFaceInstanceData {
    Magnum::Vector3 position;
    Magnum::Color3 color;
};

struct MeshEdgeInstanceData {
    Magnum::Vector3 position;
    Magnum::Color3 color;
};


static inline unsigned int render_meshedge(MeshEdgeInstanceData *edgeData, unsigned int i, Vertex *v0, Vertex *v1) {
    Magnum::Vector3 color = {0.f, 0.f, 0.f};
    Particle *pi = v0->particle()->part();
    Particle *pj = v1->particle()->part();
    
    fVector3 pj_origin = FVector3::from(_Engine.s.celllist[pj->id]->origin);
    
    int shift[3];
    fVector3 pix;
    
    int *loci = _Engine.s.celllist[ pi->id ]->loc;
    int *locj = _Engine.s.celllist[ pj->id ]->loc;
    
    for ( int k = 0 ; k < 3 ; k++ ) {
        shift[k] = loci[k] - locj[k];
        if ( shift[k] > 1 )
            shift[k] = -1;
        else if ( shift[k] < -1 )
            shift[k] = 1;
        pix[k] = pi->x[k] + _Engine.s.h[k]* shift[k];
    }
                    
    edgeData[i].position = pix + pj_origin;
    edgeData[i].color = color;
    edgeData[i+1].position = pj->position + pj_origin;
    edgeData[i+1].color = color;
    return 2;
}

static inline HRESULT render_meshFacesEdges(MeshFaceInstanceData *faceData, 
                                            MeshEdgeInstanceData *edgeData, 
                                            const unsigned int &iFace, 
                                            const unsigned int &iEdge, 
                                            Surface *s, 
                                            unsigned int &faceIncr, 
                                            unsigned int &edgeIncr) 
{
    Magnum::Vector3 color;
    
    rendering::Style *style = s->style ? s->style : s->type()->style;
    if(!style) 
        color = {0.2f, 1.f, 1.f};
    else 
        color = style->color;

    faceIncr = 0;
    edgeIncr = 0;

    Magnum::Vector3 centroid = s->getCentroid();

    // cell id of centroid
    int cid, locj[3];

    if((cid = space_get_cellids_for_pos(&_Engine.s, centroid.data(), locj)) < 0) {
        TF_Log(LOG_ERROR);
        return E_FAIL;
    }

    fVector3 pj_origin = FVector3::from(_Engine.s.cells[cid].origin);
    
    int shiftij[3], shiftkj[3];
    fVector3 pixij, pixkj;

    std::vector<Vertex*> vertices = TissueForge::models::vertex::vectorToDerived<Vertex>(s->parents());

    unsigned int j, k;
    for(j = 0; j < vertices.size(); j++) {
        Vertex *vi = vertices[j];
        Vertex *vk = vertices[j == vertices.size() - 1 ? 0 : j + 1];

        edgeIncr += render_meshedge(edgeData, iEdge + edgeIncr, vi, vk);

        Particle *pi = vi->particle()->part();
        Particle *pk = vk->particle()->part();

        int *loci = _Engine.s.celllist[pi->id]->loc;
        int *lock = _Engine.s.celllist[pk->id]->loc;

        for(k = 0; k < 3; k++) {
            int locjk = locj[k];
            shiftij[k] = loci[k] - locjk;
            shiftkj[k] = lock[k] - locjk;
            
            if(shiftij[k] > 1) shiftij[k] = -1;
            else if (shiftij[k] < -1) shiftij[k] = 1;
            
            if(shiftkj[k] > 1) shiftkj[k] = -1;
            else if (shiftkj[k] < -1) shiftkj[k] = 1;

            float h = _Engine.s.h[k];
            pixij[k] = pi->x[k] + h * shiftij[k];
            pixkj[k] = pk->x[k] + h * shiftkj[k];
        }

        fVector3 posi = pixij + pj_origin;
        fVector3 posk = pixkj + pj_origin;

        k = iFace + faceIncr;
        
        faceData[k].position = posi;
        faceData[k].color = color;
        faceData[k+1].position = centroid;
        faceData[k+1].color = color;
        faceData[k+2].position = posk;
        faceData[k+2].color = color;

        faceIncr += 3;
    }

    return S_OK;
}

MeshRenderer *MeshRenderer::get() {
    if(_meshRenderer == NULL) {
        rendering::UniverseRenderer *urenderer = system::getRenderer();
        if(urenderer) {
            _meshRenderer = new MeshRenderer();
            if(urenderer->registerSubRenderer(_meshRenderer) != S_OK) {
                delete _meshRenderer;
                _meshRenderer = NULL;
            }
        }
    }
    return _meshRenderer;
}

HRESULT MeshRenderer::start(const std::vector<fVector4> &clipPlanes) {
    // create the shaders
    _shaderFaces = shaders::Flat3D {
        shaders::Flat3D::Flag::VertexColor, 
        (unsigned int)rendering::UniverseRenderer::maxClipPlaneCount()
    };
    _shaderEdges = shaders::Flat3D{
        shaders::Flat3D::Flag::VertexColor, 
        (unsigned int)rendering::UniverseRenderer::maxClipPlaneCount()
    };
    
    // create the buffers
    _bufferFaces = Magnum::GL::Buffer();
    _bufferEdges = Magnum::GL::Buffer();

    // create the meshes
    _meshFaces = Magnum::GL::Mesh{};
    _meshFaces.setPrimitive(Magnum::MeshPrimitive::Triangles);
    _meshFaces.addVertexBuffer(_bufferFaces, 0, 
                               shaders::Phong::Position{}, 
                               shaders::Phong::Color3{});

    _meshEdges = Magnum::GL::Mesh{};
    _meshEdges.setPrimitive(Magnum::MeshPrimitive::Lines);
    _meshEdges.addVertexBuffer(_bufferEdges, 0,
                               shaders::Flat3D::Position{}, 
                               shaders::Flat3D::Color3{});

    return S_OK;
}

HRESULT MeshRenderer::draw(rendering::ArcBallCamera *camera, const iVector2 &viewportSize, const fMatrix4 &modelViewMat) {
    
    MeshSolver *solver = MeshSolver::get();
    if(!solver) {
        TF_Log(LOG_ERROR);
        return E_FAIL;
    }
    else if(solver->update() != S_OK) {
        TF_Log(LOG_ERROR);
        return E_FAIL;
    }

    MeshSolverTimerInstance t(MeshSolverTimers::Section::RENDERING);

    unsigned int vertexCountF = 3 * solver->_surfaceVertices;
    unsigned int vertexCountE = 2 * solver->_surfaceVertices;
    _meshFaces.setCount(vertexCountF);
    _meshEdges.setCount(vertexCountE);
    
    _bufferFaces.setData(
        {NULL, vertexCountF * sizeof(MeshFaceInstanceData)},
        GL::BufferUsage::DynamicDraw
    );
    _bufferEdges.setData(
        {NULL, vertexCountE * sizeof(MeshEdgeInstanceData)},
        GL::BufferUsage::DynamicDraw
    );
    
    // get pointer to data
    MeshFaceInstanceData* faceData = (MeshFaceInstanceData*)(void*)_bufferFaces.map(
        0,
        vertexCountF * sizeof(MeshFaceInstanceData),
        GL::Buffer::MapFlag::Write|GL::Buffer::MapFlag::InvalidateBuffer
    );
    MeshEdgeInstanceData* edgeData = (MeshEdgeInstanceData*)(void*)_bufferEdges.map(
        0,
        vertexCountE * sizeof(MeshEdgeInstanceData),
        GL::Buffer::MapFlag::Write|GL::Buffer::MapFlag::InvalidateBuffer
    );
    
    unsigned int iFaces = 0, iEdges = 0, faceIncr, edgeIncr;
    for(auto &m : solver->meshes) {
        for(auto &s : m->surfaces) {
            if(s) {
                render_meshFacesEdges(faceData, edgeData, iFaces, iEdges, s, faceIncr, edgeIncr);
                iFaces += faceIncr;
                iEdges += edgeIncr;
            }
        }
    }

    assert(iFaces == vertexCountF);
    assert(iEdges == vertexCountE);
    _bufferFaces.unmap();
    _bufferEdges.unmap();
    
    _shaderFaces
        .setTransformationProjectionMatrix(camera->projectionMatrix() * camera->cameraMatrix() * modelViewMat)
        .draw(_meshFaces);
    _shaderEdges
        .setTransformationProjectionMatrix(camera->projectionMatrix() * camera->cameraMatrix() * modelViewMat)
        .draw(_meshEdges);

    return S_OK;
}
