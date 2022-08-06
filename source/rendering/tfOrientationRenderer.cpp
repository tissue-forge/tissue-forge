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

#include "tfOrientationRenderer.h"

#include <tfEngine.h>
#include <tfLogger.h>
#include <tf_system.h>
#include "tfUniverseRenderer.h"

#include <Magnum/MeshTools/Compile.h>
#include <Magnum/MeshTools/Transform.h>
#include <Magnum/Primitives/Cone.h>
#include <Magnum/Primitives/Cylinder.h>
#include <Magnum/Primitives/Icosphere.h>
#include <Magnum/Shaders/Phong.h>
#include <Magnum/Trade/MeshData.h>

#include <limits>
#include <string>


using namespace TissueForge;


struct OriginInstanceData {
    Matrix4 transformationMatrix;
    Matrix3 normalMatrix;
    Magnum::Color4 color;
};

// todo: make arrow geometry settable

static float _arrowHeadHeight = 0.5;
static float _arrowHeadRadius = 0.25;
static float _arrowCylHeight = 0.5;
static float _arrowCylRadius = 0.1;

static inline HRESULT render_arrow(rendering::ArrowInstanceData *pDataArrow, rendering::ArrowInstanceData *pDataCylinder, int idx, rendering::ArrowData *p, const fVector3 &origin) {
    fVector3 position = p->position + origin;
    float vec_len = p->components.length();
    Magnum::Color4 color = {p->style.color[0], p->style.color[1], p->style.color[2], (float)p->style.getVisible()};

    float arrowHeadHeightProbably = _arrowHeadHeight;
    float arrowHeadHeight = std::min<float>(arrowHeadHeightProbably, vec_len);
    float arrowHeadScaling = arrowHeadHeight / arrowHeadHeightProbably;

    fMatrix3 rotationMatrix = rendering::vectorFrameRotation(p->components);
    Magnum::Matrix4 transformBase = Magnum::Matrix4::from(rotationMatrix, position) * Magnum::Matrix4::scaling(fVector3(p->scale));

    Magnum::Matrix4 transformHead = transformBase * Magnum::Matrix4::translation(fVector3(0.0, vec_len - arrowHeadHeight, 0.0));
    transformHead = transformHead * Magnum::Matrix4::scaling(fVector3(_arrowHeadRadius, _arrowHeadRadius * arrowHeadScaling, _arrowHeadRadius));
    transformHead = transformHead * Magnum::Matrix4::translation(fVector3(0.0, arrowHeadHeightProbably / 2.0 / _arrowHeadRadius, 0.0));

    Magnum::Matrix4 transformCyl = transformBase * Magnum::Matrix4::scaling(fVector3(_arrowCylRadius, _arrowCylRadius * (vec_len - arrowHeadHeight) / _arrowCylHeight, _arrowCylRadius));
    transformCyl = transformCyl * Magnum::Matrix4::translation(fVector3(0.0, _arrowCylHeight / _arrowCylRadius * 0.5, 0.0));

    pDataArrow[idx].transformationMatrix = transformHead;
    pDataArrow[idx].normalMatrix = transformHead.normalMatrix();
    pDataArrow[idx].color = color;

    pDataCylinder[idx].transformationMatrix = transformCyl;
    pDataCylinder[idx].normalMatrix = transformCyl.normalMatrix();
    pDataCylinder[idx].color = color;

    return S_OK;
}

rendering::OrientationRenderer::OrientationRenderer() : 
    staticTransformationMat{Matrix4::translation({0.85f, -0.8f, -0.8f}) * Matrix4::scaling({0.1f, 0.1f, 0.1f})}
{}

rendering::OrientationRenderer::~OrientationRenderer() {
    this->arrows.clear();
}

void loadMesh(
    Magnum::GL::Buffer *_bufferHead,
    Magnum::GL::Buffer *_bufferCylinder, 
    Magnum::GL::Buffer *_bufferOrigin, 
    Magnum::GL::Mesh *_meshHead, 
    Magnum::GL::Mesh *_meshCylinder, 
    Magnum::GL::Mesh *_meshOrigin, 
    const std::vector<rendering::ArrowData *> &arrows)
{
    _meshHead->setInstanceCount(arrows.size());
    _meshCylinder->setInstanceCount(arrows.size());
    _meshOrigin->setInstanceCount(1);

    _bufferHead->setData({NULL, arrows.size() * sizeof(rendering::ArrowInstanceData)});
    _bufferCylinder->setData({NULL, arrows.size() * sizeof(rendering::ArrowInstanceData)});

    rendering::ArrowInstanceData *pArrowData = (rendering::ArrowInstanceData*)(void*)_bufferHead->map(
        0, 
        arrows.size() * sizeof(rendering::ArrowInstanceData), 
        Magnum::GL::Buffer::MapFlag::Write | Magnum::GL::Buffer::MapFlag::InvalidateBuffer
    );
    rendering::ArrowInstanceData *pCylinderData = (rendering::ArrowInstanceData*)(void*)_bufferCylinder->map(
        0, 
        arrows.size() * sizeof(rendering::ArrowInstanceData), 
        Magnum::GL::Buffer::MapFlag::Write | Magnum::GL::Buffer::MapFlag::InvalidateBuffer
    );

    int i = 0;
    fVector3 origin(_Engine.s.origin[0], _Engine.s.origin[1], _Engine.s.origin[2]);
    
    for (int aid = 0; aid < arrows.size(); aid++) {
        rendering::ArrowData *ad = arrows[aid];
        if(ad != NULL) {
            render_arrow(pArrowData, pCylinderData, i, ad, origin);
            i++;
        }
    }

    assert(i == arrows.size());

    Matrix4 tm = Matrix4::from(Matrix4::scaling(Vector3(0.125f)).rotationScaling(), {});
    OriginInstanceData originData[] {
        {tm, tm.normalMatrix(), Color4{0.1f, 0.5f, 0.5f, 1.f}}
    };
    _bufferOrigin->setData(originData);

    _bufferHead->unmap();
    _bufferCylinder->unmap();
    _bufferOrigin->unmap();
}

HRESULT rendering::OrientationRenderer::start(const std::vector<fVector4> &clipPlanes) {

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
    _bufferHead = Magnum::GL::Buffer();
    _bufferCylinder = Magnum::GL::Buffer();
    _bufferOrigin = Magnum::GL::Buffer();

    // create the meshes
    unsigned int numRingsHead = this->_arrowDetail;
    unsigned int numSegmentsHead = this->_arrowDetail;
    unsigned int numRingsCyl = this->_arrowDetail;
    unsigned int numSegmentsCyl = this->_arrowDetail;
    _meshHead = Magnum::MeshTools::compile(Magnum::Primitives::coneSolid(
        numRingsHead, 
        numSegmentsHead, 
        0.5 * _arrowHeadHeight / _arrowHeadRadius, 
        Magnum::Primitives::ConeFlag::CapEnd
    ));
    _meshCylinder = Magnum::MeshTools::compile(Magnum::Primitives::cylinderSolid(
        numRingsCyl, 
        numSegmentsCyl, 
        0.5 * _arrowCylHeight / _arrowCylRadius, 
        Magnum::Primitives::CylinderFlag::CapEnds
    ));
    _meshOrigin = MeshTools::compile(Primitives::icosphereSolid(2));

    _meshHead.addVertexBufferInstanced(_bufferHead, 1, 0, 
        Magnum::Shaders::Phong::TransformationMatrix{}, 
        Magnum::Shaders::Phong::NormalMatrix{}, 
        Magnum::Shaders::Phong::Color4{}
    );
    _meshCylinder.addVertexBufferInstanced(_bufferCylinder, 1, 0, 
        Magnum::Shaders::Phong::TransformationMatrix{}, 
        Magnum::Shaders::Phong::NormalMatrix{}, 
        Magnum::Shaders::Phong::Color4{}
    );
    _meshOrigin.addVertexBufferInstanced(_bufferOrigin, 1, 0, 
        Magnum::Shaders::Phong::TransformationMatrix{}, 
        Magnum::Shaders::Phong::NormalMatrix{}, 
        Magnum::Shaders::Phong::Color4{}
    );

    this->modelViewMat = fMatrix4::translation(-(Universe::dim() + engine_origin()) / 2.);

    this->arrowx = new rendering::ArrowData();
    this->arrowy = new rendering::ArrowData();
    this->arrowz = new rendering::ArrowData();

    this->arrowx->style.color = Color3::red();
    this->arrowy->style.color = Color3::green();
    this->arrowz->style.color = Color3::blue();

    this->arrowx->components = fVector3::xAxis();
    this->arrowy->components = fVector3::yAxis();
    this->arrowz->components = fVector3::zAxis();

    this->arrowx->scale = 0.5f;
    this->arrowy->scale = 0.5f;
    this->arrowz->scale = 0.5f;

    this->arrows = {this->arrowx, this->arrowy, this->arrowz};

    loadMesh(&_bufferHead, &_bufferCylinder, &_bufferOrigin, &_meshHead, &_meshCylinder, &_meshOrigin, this->arrows);

    return S_OK;
}

HRESULT rendering::OrientationRenderer::draw(TissueForge::rendering::ArcBallCamera *camera, const iVector2 &viewportSize, const fMatrix4 &modelViewMat) {
    TF_Log(LOG_DEBUG) << "";

    if(!this->_showAxes) 
        return S_OK;

    _shader
        .setProjectionMatrix(Matrix4())
        .setTransformationMatrix(
            this->staticTransformationMat * 
            Matrix4::from((camera->projectionMatrix() * camera->cameraMatrix() * this->modelViewMat).rotationScaling(), {})
        )
        .setNormalMatrix(camera->viewMatrix().normalMatrix());
    _shader.draw(_meshHead);
    _shader.draw(_meshCylinder);
    _shader.draw(_meshOrigin);

    return S_OK;
}

void rendering::OrientationRenderer::setAmbientColor(const Magnum::Color3& color) {
    _shader.setAmbientColor(color);
}

void rendering::OrientationRenderer::setDiffuseColor(const Magnum::Color3& color) {
    _shader.setDiffuseColor(color);
}

void rendering::OrientationRenderer::setSpecularColor(const Magnum::Color3& color) {
    _shader.setSpecularColor(color);
}

void rendering::OrientationRenderer::setShininess(float shininess) {
    _shader.setShininess(shininess);
}

void rendering::OrientationRenderer::setLightDirection(const fVector3& lightDir) {
    _shader.setLightPosition(lightDir);
}

void rendering::OrientationRenderer::setLightColor(const Magnum::Color3 &color) {
    _shader.setLightColor(color);
}

rendering::OrientationRenderer *rendering::OrientationRenderer::get() {
    auto *renderer = system::getRenderer();
    return (rendering::OrientationRenderer*)renderer->getSubRenderer(rendering::SubRendererFlag::SUBRENDERER_ORIENTATION);
}
