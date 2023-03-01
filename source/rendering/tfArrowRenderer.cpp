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

#include "tfArrowRenderer.h"

#include <tfEngine.h>
#include <tfLogger.h>
#include <tf_system.h>
#include "tfUniverseRenderer.h"

#include <Magnum/MeshTools/Compile.h>
#include <Magnum/MeshTools/Transform.h>
#include <Magnum/Primitives/Cone.h>
#include <Magnum/Primitives/Cylinder.h>
#include <Magnum/Shaders/Phong.h>
#include <Magnum/Trade/MeshData.h>

#include <limits>
#include <string>

// todo: make arrow geometry settable

static float _arrowHeadHeight = 0.5;
static float _arrowHeadRadius = 0.25;
static float _arrowCylHeight = 0.5;
static float _arrowCylRadius = 0.1;


using namespace TissueForge;


fMatrix3 rendering::vectorFrameRotation(const fVector3 &vec) {
    fVector3 u1, u2, u3, p;

    float vec_len = vec.length();
    if(vec_len == 0.0) {
        tf_error(E_FAIL, "Cannot pass zero vector");
        return fMatrix3();
    }

    u2 = vec / vec_len;

    if(vec[0] != 0) {
        p[1] = 1.0;
        p[2] = 1.0;
        p[0] = - (vec[1] + vec[2]) / vec[0];
    }
    else if (vec[1] != 0) {
        p[0] = 1.0;
        p[2] = 1.0;
        p[1] = - (vec[0] + vec[2]) / vec[1];
    }
    else {
        p[0] = 1.0;
        p[1] = 1.0;
        p[2] = - (vec[0] + vec[1]) / vec[2];
    }
    u3 = p.normalized();

    u1 = Magnum::Math::cross(u2, u3);

    fMatrix3 result(u1, u2, u3);

    return result;
}

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

rendering::ArrowRenderer::ArrowRenderer() {
    this->nr_arrows = 0;
}

rendering::ArrowRenderer::ArrowRenderer(const rendering::ArrowRenderer &other) {
    this->arrows = other.arrows;
    this->nr_arrows = other.nr_arrows;

    this->_arrowDetail = other._arrowDetail;
}

rendering::ArrowRenderer::~ArrowRenderer() {
    this->arrows.clear();
}

HRESULT rendering::ArrowRenderer::start(const std::vector<fVector4> &clipPlanes) {

    // create the shaders
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
    _bufferHead = Magnum::GL::Buffer();
    _bufferCylinder = Magnum::GL::Buffer();

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

    _meshHead.addVertexBufferInstanced(_bufferHead, 1, 0, 
        shaders::Phong::TransformationMatrix{}, 
        shaders::Phong::NormalMatrix{}, 
        shaders::Phong::Color4{}
    );
    _meshCylinder.addVertexBufferInstanced(_bufferCylinder, 1, 0, 
        shaders::Phong::TransformationMatrix{}, 
        shaders::Phong::NormalMatrix{}, 
        shaders::Phong::Color4{}
    );
    _meshHead.setInstanceCount(0);
    _meshCylinder.setInstanceCount(0);

    return S_OK;
}

HRESULT rendering::ArrowRenderer::draw(rendering::ArcBallCamera *camera, const iVector2& viewportSize, const fMatrix4 &modelViewMat) {
    TF_Log(LOG_DEBUG) << "";

    _meshHead.setInstanceCount(this->nr_arrows);
    _meshCylinder.setInstanceCount(this->nr_arrows);

    _bufferHead.setData({NULL, this->nr_arrows * sizeof(rendering::ArrowInstanceData)}, Magnum::GL::BufferUsage::DynamicDraw);
    _bufferCylinder.setData({NULL, this->nr_arrows * sizeof(rendering::ArrowInstanceData)}, Magnum::GL::BufferUsage::DynamicDraw);

    rendering::ArrowInstanceData *pArrowData = (rendering::ArrowInstanceData*)(void*)_bufferHead.map(
        0, 
        this->nr_arrows * sizeof(rendering::ArrowInstanceData), 
        Magnum::GL::Buffer::MapFlag::Write | Magnum::GL::Buffer::MapFlag::InvalidateBuffer
    );
    rendering::ArrowInstanceData *pCylinderData = (rendering::ArrowInstanceData*)(void*)_bufferCylinder.map(
        0, 
        this->nr_arrows * sizeof(rendering::ArrowInstanceData), 
        Magnum::GL::Buffer::MapFlag::Write | Magnum::GL::Buffer::MapFlag::InvalidateBuffer
    );

    int i = 0;
    fVector3 origin(_Engine.s.origin[0], _Engine.s.origin[1], _Engine.s.origin[2]);
    
    for (int aid = 0; aid < this->arrows.size(); aid++) {
        rendering::ArrowData *ad = this->arrows[aid];
        if(ad != NULL) {
            render_arrow(pArrowData, pCylinderData, i, ad, origin);
            i++;
        }
    }

    assert(i == this->nr_arrows);

    _bufferHead.unmap();
    _bufferCylinder.unmap();

    _shader
        .setProjectionMatrix(camera->projectionMatrix())
        .setTransformationMatrix(camera->cameraMatrix() * modelViewMat)
        .setNormalMatrix(camera->viewMatrix().normalMatrix());
    _shader.draw(_meshHead);
    _shader.draw(_meshCylinder);

    return S_OK;
}

const unsigned rendering::ArrowRenderer::addClipPlaneEquation(const Magnum::Vector4& pe) {
    unsigned int id = _clipPlanes.size();
    _clipPlanes.push_back(pe);
    _shader.setclipPlaneEquation(id, pe);
    return id;
}

const unsigned rendering::ArrowRenderer::removeClipPlaneEquation(const unsigned int &id) {
    _clipPlanes.erase(_clipPlanes.begin() + id);

    for(unsigned int i = id; i < _clipPlanes.size(); i++) {
        _shader.setclipPlaneEquation(i, _clipPlanes[i]);
    }

    return _clipPlanes.size();
}

void rendering::ArrowRenderer::setClipPlaneEquation(unsigned id, const Magnum::Vector4& pe) {
    if(id > _shader.clipPlaneCount()) tf_exp(std::invalid_argument("invalid id for clip plane"));

    _shader.setclipPlaneEquation(id, pe);
    _clipPlanes[id] = pe;
}

void rendering::ArrowRenderer::setAmbientColor(const Magnum::Color3& color) {
    _shader.setAmbientColor(color);
}

void rendering::ArrowRenderer::setDiffuseColor(const Magnum::Color3& color) {
    _shader.setDiffuseColor(color);
}

void rendering::ArrowRenderer::setSpecularColor(const Magnum::Color3& color) {
    _shader.setSpecularColor(color);
}

void rendering::ArrowRenderer::setShininess(float shininess) {
    _shader.setShininess(shininess);
}

void rendering::ArrowRenderer::setLightDirection(const fVector3& lightDir) {
    _shader.setLightPosition(lightDir);
}

void rendering::ArrowRenderer::setLightColor(const Magnum::Color3 &color) {
    _shader.setLightColor(color);
}

int rendering::ArrowRenderer::nextDataId() {
    if(this->nr_arrows == this->arrows.size()) return this->nr_arrows;

    for(int i = 0; i < this->arrows.size(); ++i)
        if(this->arrows[i] == NULL)
            return i;

    tf_error(E_FAIL, "Could not identify new arrow id");
    return -1;
}

int rendering::ArrowRenderer::addArrow(rendering::ArrowData *arrow) {
    int arrowId = this->nextDataId();
    arrow->id = arrowId;
    
    if(arrowId == this->arrows.size()) this->arrows.push_back(arrow);
    else this->arrows[arrowId] = arrow;

    this->nr_arrows++;

    TF_Log(LOG_DEBUG) << arrowId;

    return arrowId;
}

std::pair<int, rendering::ArrowData*> rendering::ArrowRenderer::addArrow(
    const fVector3 &position, 
    const fVector3 &components, 
    const rendering::Style &style, 
    const float &scale) 
{
    rendering::ArrowData *arrow = new rendering::ArrowData();
    arrow->position = position;
    arrow->components = components;
    arrow->style = style;
    arrow->scale = scale;

    int aid = this->addArrow(arrow);
    return std::make_pair(aid, arrow);
}

HRESULT rendering::ArrowRenderer::removeArrow(const int &arrowId) {
    this->nr_arrows--;
    this->arrows[arrowId] = NULL;
    return S_OK;
}

rendering::ArrowData *rendering::ArrowRenderer::getArrow(const int &arrowId) {
    return this->arrows[arrowId];
}

rendering::ArrowRenderer *rendering::ArrowRenderer::get() {
    auto renderer = system::getRenderer();
    return (rendering::ArrowRenderer*)renderer->getSubRenderer(rendering::SubRendererFlag::SUBRENDERER_ARROW);
}
