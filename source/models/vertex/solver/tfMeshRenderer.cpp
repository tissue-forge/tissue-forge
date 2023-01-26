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
#include <tf_metrics.h>
#include <tf_system.h>
#include <tfTaskScheduler.h>

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


static inline HRESULT render_meshFacesEdges(MeshFaceInstanceData *faceData, 
                                            MeshEdgeInstanceData *edgeData, 
                                            const unsigned int &idx, 
                                            Surface *s) 
{
    Magnum::Vector3 color_s, color_e = {0.f, 0.f, 0.f};
    
    rendering::Style *style = s->style ? s->style : s->type()->style;
    if(!style) 
        color_s = {0.2f, 1.f, 1.f};
    else 
        color_s = style->color;

    Magnum::Vector3 centroid = s->getCentroid();

    std::vector<Vertex*> vertices = s->getVertices();

    for(unsigned int j = 0; j < vertices.size(); j++) {
        Vertex *vi = vertices[j];
        Vertex *vk = vertices[j == vertices.size() - 1 ? 0 : j + 1];

        fVector3 posi = centroid + metrics::relativePosition(vi->getPosition(), centroid);
        fVector3 posk = centroid + metrics::relativePosition(vk->getPosition(), centroid);

        MeshEdgeInstanceData *edgeData_j = &edgeData[2 * (idx + j)];
        edgeData_j[0].position = posi;
        edgeData_j[0].color = color_e;
        edgeData_j[1].position = posk;
        edgeData_j[1].color = color_e;

        MeshFaceInstanceData *faceData_j = &faceData[3 * (idx + j)];
        faceData_j[0].position = posi;
        faceData_j[0].color = color_s;
        faceData_j[1].position = centroid;
        faceData_j[1].color = color_s;
        faceData_j[2].position = posk;
        faceData_j[2].color = color_s;
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

    std::vector<unsigned int> surfaceVertexIndices = solver->getSurfaceVertexIndices();

    if(solver->mesh->surfaces->size() > 0) {
        Surface *m_surfaces = &(*solver->mesh->surfaces)[0];
        auto func_surfaces = [&faceData, &edgeData, &m_surfaces, &surfaceVertexIndices](int i) -> void {
            Surface &s = m_surfaces[i];
            if(s.objectId() >= 0) {
                render_meshFacesEdges(faceData, edgeData, surfaceVertexIndices[i], &s);
            }
        };
        parallel_for(solver->mesh->surfaces->size(), func_surfaces);
    }
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
