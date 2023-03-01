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

/**
 Derived from Magnum, with the following notice:

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

#include <tfSimulator.h>

#include <Corrade/Utility/Assert.h>
#include <Corrade/Containers/ArrayView.h>
#include <Corrade/Containers/GrowableArray.h>
#include "tfUniverseRenderer.h"
#include <Magnum/GL/Renderer.h>
#include <Magnum/Math/Functions.h>
#include <Magnum/Shaders/Generic.h>
#include <Magnum/SceneGraph/Drawable.h>
#include <Magnum/Trade/MeshData.h>

#include <Magnum/Animation/Easing.h>
#include <Magnum/GL/DefaultFramebuffer.h>
#include <Magnum/GL/Renderer.h>
#include <Magnum/GL/PixelFormat.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Image.h>
#include <Magnum/GL/Context.h>
#include <Magnum/GL/Version.h>
#include <Magnum/SceneGraph/Scene.h>
#include <Magnum/SceneGraph/Camera.h>
#include <Magnum/SceneGraph/Drawable.h>
#include <Magnum/Math/FunctionsBatch.h>

#include <Magnum/MeshTools/Compile.h>
#include <Magnum/Primitives/Cube.h>
#include <Magnum/Primitives/Grid.h>
#include <Magnum/Primitives/Icosphere.h>

#include <Magnum/Math/Vector4.h>

#include "tfStyle.h"
#include "tfAngleRenderer.h"
#include "tfAngleRenderer3D.h"
#include "tfArrowRenderer.h"
#include "tfBondRenderer.h"
#include "tfBondRenderer3D.h"
#include "tfDihedralRenderer.h"
#include "tfDihedralRenderer3D.h"
#include "tfOrientationRenderer.h"

#include <tf_util.h>
#include <tfLogger.h>
#include <tfError.h>
#include <tfTaskScheduler.h>
#include <tfEngine.h>

#include <assert.h>
#include <iostream>
#include <stdexcept>


using namespace Magnum::Math::Literals;
using namespace TissueForge;


static GL::Renderer::Feature clipDistanceGLLabels[] = {
    GL::Renderer::Feature::ClipDistance0, 
    GL::Renderer::Feature::ClipDistance1, 
    GL::Renderer::Feature::ClipDistance2, 
    GL::Renderer::Feature::ClipDistance3, 
    GL::Renderer::Feature::ClipDistance4, 
    GL::Renderer::Feature::ClipDistance5, 
    GL::Renderer::Feature::ClipDistance6, 
    GL::Renderer::Feature::ClipDistance7
};
static const unsigned int numClipDistanceGLLabels = 8;

struct discretizationGridData{
    Matrix4 transformationMatrix;
    Color3 color;
};

static inline void render_discretization_grid(
    TissueForge::uiVector3 nr_cells, 
    fVector3 grid_dim, 
    GL::Buffer *discretizationGridBuffer, 
    GL::Mesh *discretizationGridMesh, 
    Color3 discretizationGridColor) 
{

    float cell_dim_x = grid_dim.x() / nr_cells.x();
    float cell_dim_y = grid_dim.y() / nr_cells.y();
    float cell_dim_z = grid_dim.z() / nr_cells.z();
    Vector3 cell_dim{cell_dim_x, cell_dim_y, cell_dim_z};
    Vector3 cell_hdim = cell_dim * 0.5;

    Containers::Array<discretizationGridData> _discretizationGridData;
    Corrade::Containers::arrayResize(_discretizationGridData, 0);
    for(unsigned int i = 0; i < nr_cells.x(); i++) {
        float ox = cell_dim_x * i;
        for(unsigned int j = 0; j < nr_cells.y(); j++) {
            float oy = cell_dim_y * j;
            for(unsigned int k = 0; k < nr_cells.z(); k++) {
                float oz = cell_dim_z * k;
                Vector3 cell_origin{ox, oy, oz};
                Vector3 cell_center = cell_origin + cell_hdim;
                Matrix4 tm = Matrix4::translation(cell_center) * Matrix4::scaling(cell_hdim);
                Corrade::Containers::arrayAppend(_discretizationGridData, Corrade::Containers::InPlaceInit, tm, discretizationGridColor);
            }
        }
    }
    discretizationGridBuffer->setData(_discretizationGridData, GL::BufferUsage::DynamicDraw);
    discretizationGridMesh->setInstanceCount(_discretizationGridData.size());
}

static inline const bool cameraZoom(rendering::ArcBallCamera *camera, const float &delta) 
{
    if(Math::abs(delta) < 1.0e-2f) return false;

    const float distance = camera->viewDistance() * delta;
    camera->zoom(distance);
    return true;
}

static inline int render_largeparticle(rendering::SphereInstanceData* pData, int i, Particle *p) {

    rendering::Style *style = p->style ? p->style : (&_Engine.types[p->typeId])->style;

    float radius = style->flags & STYLE_VISIBLE && !(p->flags & PARTICLE_CLUSTER) ? p->radius : 0;
    pData[i].transformationMatrix =
        Matrix4::translation(p->position) * Matrix4::scaling(Vector3{radius});
    pData[i].normalMatrix =
        pData[i].transformationMatrix.normalMatrix();
    pData[i].color = style->map_color(p);

    return 1;
}

static inline int render_cell_particles(rendering::SphereInstanceData* pData, int cid) {

    int buffId = 0;
    for(int i = 0; i < cid; i++) 
        buffId += _Engine.s.cells[i].count;

    space_cell *c = &_Engine.s.cells[cid];
    rendering::SphereInstanceData *pData_buff = &pData[buffId];

    int count = 0;
    fVector3 origin = FVector3::from(c->origin);

    for(int pid = 0; pid < c->count; pid++) {
        Particle *p = &c->parts[pid];
        fVector4 px = FVector4::from(p->x);

        rendering::Style *style = p->style ? p->style : (&_Engine.types[p->typeId])->style;

        bool isvisible = style->flags & STYLE_VISIBLE && !(p->flags & PARTICLE_CLUSTER);
        float radius = isvisible ? p->radius : 0;
        count += isvisible;
    
        Magnum::Vector3 position = {
            origin[0] + px[0],
            origin[1] + px[1],
            origin[2] + px[2]
        };
        
        pData_buff[pid].transformationMatrix = Matrix4::translation(position) * Matrix4::scaling(Vector3{radius});
        pData_buff[pid].normalMatrix = pData_buff[pid].transformationMatrix.normalMatrix();
        pData_buff[pid].color = style->map_color(p);
    }

    return count;
}

rendering::UniverseRenderer::UniverseRenderer(const Simulator::Config &conf, rendering::Window *win):
    window{win}, 
    _zoomRate(0.05), 
    _spinRate{0.01*M_PI}, 
    _moveRate{0.01},
    _lagging{0.85f}
{
    TF_Log(LOG_DEBUG) << "Creating UniverseRenderer";

    //GL::Renderer::enable(GL::Renderer::Feature::FaceCulling);
    GL::Renderer::enable(GL::Renderer::Feature::DepthTest);

    TF_Log(LOG_DEBUG) << "clip planes: " << conf.clipPlanes.size();
    
    if(conf.clipPlanes.size() > 0) {
        GL::Renderer::enable(GL::Renderer::Feature::ClipDistance0);
    }
    if(conf.clipPlanes.size() > 1) {
        GL::Renderer::enable(GL::Renderer::Feature::ClipDistance1);
    }
    if(conf.clipPlanes.size() > 2) {
        GL::Renderer::enable(GL::Renderer::Feature::ClipDistance2);
    }
    if(conf.clipPlanes.size() > 3) {
        GL::Renderer::enable(GL::Renderer::Feature::ClipDistance3);
    }
    if(conf.clipPlanes.size() > 4) {
        GL::Renderer::enable(GL::Renderer::Feature::ClipDistance4);
    }
    if(conf.clipPlanes.size() > 5) {
        GL::Renderer::enable(GL::Renderer::Feature::ClipDistance5);
    }
    if(conf.clipPlanes.size() > 6) {
        GL::Renderer::enable(GL::Renderer::Feature::ClipDistance6);
    }
    if(conf.clipPlanes.size() > 7) {
        GL::Renderer::enable(GL::Renderer::Feature::ClipDistance7);
    }
    if(conf.clipPlanes.size() > 8) {
        tf_exp(std::invalid_argument("only up to 8 clip planes supported"));
    }
    
    GL::Renderer::setDepthFunction(GL::Renderer::StencilFunction::Less);
    
    GL::Renderer::enable(GL::Renderer::Feature::Blending);
    
    GL::Renderer::setBlendFunction(
       GL::Renderer::BlendFunction::SourceAlpha, /* or SourceAlpha for non-premultiplied */
       GL::Renderer::BlendFunction::OneMinusSourceAlpha);

    /* Loop at 60 Hz max */
    glfwSwapInterval(1);

    FVector3 origin = engine_origin();
    FVector3 dim = Universe::dim();

    center = (dim + origin) / 2.;

    sideLength = dim.max();
    
    Vector3i size = {(int)std::ceil(dim[0]), (int)std::ceil(dim[1]), (int)std::ceil(dim[2])};

    /* Set up the camera */
    {
        /* Setup the arcball after the camera objects */
        const fVector3 eye = fVector3(0.5f * sideLength, -2.2f * sideLength, 1.1f * sideLength);
        const fVector3 center{0.f, 0.f, -0.1f * sideLength};
        const fVector3 up = fVector3::zAxis();
        
        _eye = eye;
        _center = center;
        _up = up;
        const Vector2i viewportSize = win->windowSize();

        _arcball = new rendering::ArcBallCamera(eye, center, up, 45.0_degf,
            viewportSize, win->framebuffer().viewport().size());
    }

    /* Setup ground grid */
    
    // makes a grid and scene box. Both of these get made with extent
    // of {-1, 1}, thus, have a size of 2x2x2, so the transform for these
    // needs to cut them in half.
    gridMesh = MeshTools::compile(Primitives::grid3DWireframe({9, 9}));
    sceneBox = MeshTools::compile(Primitives::cubeWireframe());
    gridModelView = Matrix4::scaling({size[0]/2.f, size[1]/2.f, size[2]/2.f});

    setModelViewTransform(Matrix4::translation(-center));
    
    // set up the sphere rendering...
    sphereShader = shaders::Phong {
        shaders::Phong::Flag::VertexColor |
        shaders::Phong::Flag::InstancedTransformation,
        1,                                                // light count
        numClipDistanceGLLabels                           // clip plane count
    };
    
    sphereInstanceBuffer = GL::Buffer{};

    largeSphereInstanceBuffer = GL::Buffer{};
    
    wireframeShader = Shaders::Flat3D{};
    
    sphereMesh = MeshTools::compile(Primitives::icosphereSolid(2));

    largeSphereMesh = MeshTools::compile(Primitives::icosphereSolid(4));
    
    sphereMesh.addVertexBufferInstanced(sphereInstanceBuffer, 1, 0,
        shaders::Phong::TransformationMatrix{},
        shaders::Phong::NormalMatrix{},
        shaders::Phong::Color4{});
    
    largeSphereMesh.addVertexBufferInstanced(largeSphereInstanceBuffer, 1, 0,
        shaders::Phong::TransformationMatrix{},
        shaders::Phong::NormalMatrix{},
        shaders::Phong::Color4{});
    
    // we resize instances all the time.
    sphereMesh.setInstanceCount(0);
    largeSphereMesh.setInstanceCount(0);

    // setup optional discretization grid

    discretizationGridBuffer = GL::Buffer{};
    discretizationGridMesh = MeshTools::compile(Primitives::cubeWireframe());
    discretizationGridMesh.addVertexBufferInstanced(
        discretizationGridBuffer, 1, 0, 
        Shaders::Flat3D::TransformationMatrix{}, 
        Shaders::Flat3D::Color3{}
    );
    render_discretization_grid(uiVector3(conf.universeConfig.spaceGridSize), dim, &discretizationGridBuffer, &discretizationGridMesh, _discretizationGridColor);

    // Set up subrenderers and finish

    subRenderers = {
        new rendering::AngleRenderer(), 
        new rendering::ArrowRenderer(), 
        new rendering::BondRenderer(), 
        new rendering::DihedralRenderer(), 
        new rendering::OrientationRenderer()
    };
    std::vector<fVector4> clipPlanes;
    for(auto &e : conf.clipPlanes) clipPlanes.push_back(e);
    for(auto &s : subRenderers) 
        s->start(clipPlanes);
    
    for(int i = 0; i < clipPlanes.size(); ++i) {
        TF_Log(LOG_DEBUG) << "clip plane " << i << ": " << clipPlanes[i];
        addClipPlaneEquation(clipPlanes[i]);
    }

    this->setLightDirection(lightDirection())
        .setLightColor(lightColor())
        .setShininess(shininess())
        .setAmbientColor(ambientColor())
        .setDiffuseColor(diffuseColor())
        .setSpecularColor(specularColor()) 
        .setBackgroundColor(backgroundColor());
}

template<typename T>
rendering::UniverseRenderer& rendering::UniverseRenderer::draw(T& camera, const iVector2& viewportSize) {
    
    util::WallTime wt;
    
    util::PerformanceTimer t1(engine_timer_render);
    util::PerformanceTimer t2(engine_timer_render_total);
    
    _dirty = false;

    int nr_parts = _Engine.s.nr_parts - _Engine.s.largeparts.count;

    sphereMesh.setInstanceCount(nr_parts);
    largeSphereMesh.setInstanceCount(_Engine.s.largeparts.count);

    // invalidate / resize the buffer
    sphereInstanceBuffer.setData(
        {NULL, nr_parts * sizeof(SphereInstanceData)},
        GL::BufferUsage::DynamicDraw
    );

    largeSphereInstanceBuffer.setData(
        {NULL, _Engine.s.largeparts.count * sizeof(SphereInstanceData)},
        GL::BufferUsage::DynamicDraw
    );
    
    // get pointer to data
    SphereInstanceData* pData = (SphereInstanceData*)(void*)sphereInstanceBuffer.map(
        0,
        nr_parts * sizeof(SphereInstanceData),
        GL::Buffer::MapFlag::Write|GL::Buffer::MapFlag::InvalidateBuffer
    );

    parallel_for(_Engine.s.nr_cells, [&pData](int _cid) -> void {render_cell_particles(pData, _cid);});
    sphereInstanceBuffer.unmap();


    // get pointer to data
    SphereInstanceData* pLargeData = (SphereInstanceData*)(void*)largeSphereInstanceBuffer.map(
        0,
        _Engine.s.largeparts.count * sizeof(SphereInstanceData),
        GL::Buffer::MapFlag::Write|GL::Buffer::MapFlag::InvalidateBuffer
    );

    auto func_render_large_particles = [&pLargeData](int _pid) -> void {
        Particle *p = &_Engine.s.largeparts.parts[_pid];
        render_largeparticle(pLargeData, _pid, p);
    };
    parallel_for(_Engine.s.largeparts.count, func_render_large_particles);
    largeSphereInstanceBuffer.unmap();

    
    if(_decorateScene) {
        wireframeShader.setColor(_gridColor)
            .setTransformationProjectionMatrix(
                camera->projectionMatrix() *
                camera->cameraMatrix() *
                gridModelView)
            .draw(gridMesh);
        
        wireframeShader.setColor(_sceneBoxColor)
            .draw(sceneBox);
    }

    sphereShader
        .setProjectionMatrix(camera->projectionMatrix())
        .setTransformationMatrix(camera->cameraMatrix() * modelViewMat)
        .setNormalMatrix(camera->viewMatrix().normalMatrix());
    
    sphereShader.draw(sphereMesh);
    sphereShader.draw(largeSphereMesh);

    if(_showDiscretizationGrid) 
        sphereShader.draw(discretizationGridMesh);

    for(auto &s : subRenderers) 
        s->draw(camera, viewportSize, modelViewMat);
    
    return *this;
}

const Float rendering::UniverseRenderer::lineWidth() {
    GLfloat lw;
    glGetFloatv(GL_LINE_WIDTH, &lw);
    return lw;
}

rendering::UniverseRenderer& rendering::UniverseRenderer::setLineWidth(const Float &lw) {
    Magnum::GL::Renderer::setLineWidth(lw);
    return *this;
}

const Float rendering::UniverseRenderer::lineWidthMin() {
    auto lwr = Magnum::GL::Renderer::lineWidthRange();
    return lwr.min();
}

const Float rendering::UniverseRenderer::lineWidthMax() {
    auto lwr = Magnum::GL::Renderer::lineWidthRange();
    return lwr.max();
}

rendering::UniverseRenderer& rendering::UniverseRenderer::setAmbientColor(const Color3& color) {
    sphereShader.setAmbientColor(color);

    for(auto &s : subRenderers) 
        s->setAmbientColor(color);
    
    _ambientColor = color;
    return *this;
}

rendering::UniverseRenderer& rendering::UniverseRenderer::setDiffuseColor(const Color3& color) {
    sphereShader.setDiffuseColor(color);

    for(auto &s : subRenderers) 
        s->setDiffuseColor(color);

    _diffuseColor = color;
    return *this;
}

rendering::UniverseRenderer& rendering::UniverseRenderer::setSpecularColor(const Color3& color) {
    sphereShader.setSpecularColor(color);

    for(auto &s : subRenderers) 
        s->setSpecularColor(color);

    _specularColor = color;
    return *this;
}

rendering::UniverseRenderer& rendering::UniverseRenderer::setShininess(float shininess) {
    sphereShader.setShininess(shininess);

    for(auto &s : subRenderers) 
        s->setShininess(shininess);

    _shininess = shininess;
    return *this;
}

rendering::UniverseRenderer& rendering::UniverseRenderer::setLightDirection(const fVector3& lightDir) {
    sphereShader.setLightPositions({fVector4{lightDir, 0}});

    for(auto &s : subRenderers) 
        s->setLightDirection(lightDir);

    _lightDir = lightDir;
    return *this;
}

rendering::UniverseRenderer& rendering::UniverseRenderer::setLightColor(const Color3 &color) {
    sphereShader.setLightColor(Magnum::Color4(color));

    for(auto &s : subRenderers) 
        s->setLightColor(color);

    _lightColor = color;
    return *this;
}

rendering::UniverseRenderer& rendering::UniverseRenderer::setBackgroundColor(const Color3 &color) {
    GL::Renderer::setClearColor(color);

    _clearColor = color;
    return *this;
}

rendering::UniverseRenderer& rendering::UniverseRenderer::setDiscretizationGridColor(const Color3 &color) {
    _discretizationGridColor = color;
    render_discretization_grid(
        uiVector3(iVector3::from(_Engine.s.cdim)), 
        fVector3(FVector3::from(_Engine.s.dim)), 
        &discretizationGridBuffer, 
        &discretizationGridMesh, 
        _discretizationGridColor
    );
    return *this;
}

void rendering::UniverseRenderer::setupCallbacks() {
    TF_NOTIMPLEMENTED_NORET
}

rendering::UniverseRenderer::~UniverseRenderer() {
    TF_Log(LOG_TRACE);
}

void rendering::UniverseRenderer::onCursorMove(double xpos, double ypos)
{
}

void rendering::UniverseRenderer::decorateScene(const bool &decorate) {
    _decorateScene = decorate;
    rendering::OrientationRenderer::get()->showAxes(decorate);
}

bool rendering::UniverseRenderer::sceneDecorated() const {
    return _decorateScene;
}

fVector3 rendering::UniverseRenderer::unproject(const iVector2& windowPosition, float depth) const {
    /* We have to take window size, not framebuffer size, since the position is
       in window coordinates and the two can be different on HiDPI systems */
    const Vector2i viewSize = window->windowSize();
    const Vector2i viewPosition = Vector2i{windowPosition.x(), viewSize.y() - windowPosition.y() - 1};
    const fVector3 in{2.0f*Vector2{viewPosition}/Vector2{viewSize} - fVector2{1.0f}, depth*2.0f - 1.0f};

    return in;
}

void rendering::UniverseRenderer::onCursorEnter(int entered)
{
}

void rendering::UniverseRenderer::onRedraw()
{
}

void rendering::UniverseRenderer::onWindowMove(int x, int y)
{
}

void rendering::UniverseRenderer::onWindowSizeChange(int x, int y)
{
}

void rendering::UniverseRenderer::onFramebufferSizeChange(int x, int y)
{
}

void rendering::UniverseRenderer::draw() {
    
    TF_Log(LOG_TRACE);
    
    window->framebuffer().clear(GL::FramebufferClear::Color | GL::FramebufferClear::Depth);

    // Call arcball update in every frame. This will do nothing if the camera
    //   has not been changed. computes new transform.
    _arcball->updateTransformation();

    /* Trigger drawable object to update the particles to the GPU */
    setDirty();
    
    /* Draw particles */
    draw(_arcball, window->framebuffer().viewport().size());
}

void rendering::UniverseRenderer::viewportEvent(const int w, const int h) {
    /* Resize the main framebuffer */
    window->framebuffer().setViewport({{}, window->windowSize()});
}

void rendering::UniverseRenderer::onMouseButton(int button, int action, int mods)
{
}

void rendering::UniverseRenderer::viewportEvent(Platform::GlfwApplication::ViewportEvent& event) {
    window->framebuffer().setViewport({{}, event.framebufferSize()});

    _arcball->reshape(event.windowSize(), event.framebufferSize());
}

void rendering::UniverseRenderer::cameraTranslateDown() {
    _arcball->translateDelta({0, _moveRate * sideLength, 0});
}

void rendering::UniverseRenderer::cameraTranslateUp() {
    _arcball->translateDelta({0, -_moveRate * sideLength, 0});
}

void rendering::UniverseRenderer::cameraTranslateRight() {
    _arcball->translateDelta({-_moveRate * sideLength, 0, 0});
}

void rendering::UniverseRenderer::cameraTranslateLeft() {
    _arcball->translateDelta({_moveRate * sideLength, 0, 0});
}

void rendering::UniverseRenderer::cameraTranslateForward() {
    _arcball->translateDelta({0, 0, _moveRate * sideLength});
}

void rendering::UniverseRenderer::cameraTranslateBackward() {
    _arcball->translateDelta({0, 0, -_moveRate * sideLength});
}

void rendering::UniverseRenderer::cameraRotateDown() {
    _arcball->rotateDelta(&_spinRate, NULL, NULL);
}

void rendering::UniverseRenderer::cameraRotateUp() {
    const float _ang(-_spinRate);
    _arcball->rotateDelta(&_ang, NULL, NULL);
}

void rendering::UniverseRenderer::cameraRotateLeft() {
    const float _ang(-_spinRate);
    _arcball->rotateDelta(NULL, &_ang, NULL);
}

void rendering::UniverseRenderer::cameraRotateRight() {
    _arcball->rotateDelta(NULL, &_spinRate, NULL);
}

void rendering::UniverseRenderer::cameraRollLeft() {
    _arcball->rotateDelta(NULL, NULL, &_spinRate);
}

void rendering::UniverseRenderer::cameraRollRight() {
    const float _ang(-_spinRate);
    _arcball->rotateDelta(NULL, NULL, &_ang);
}

void rendering::UniverseRenderer::cameraZoomIn() {
    cameraZoom(_arcball, _zoomRate);
}

void rendering::UniverseRenderer::cameraZoomOut() {
    cameraZoom(_arcball, - _zoomRate);
}

void rendering::UniverseRenderer::keyPressEvent(Platform::GlfwApplication::KeyEvent& event) {
    switch(event.key()) {
        case Platform::GlfwApplication::KeyEvent::Key::B: {
            if(event.modifiers() & Platform::GlfwApplication::MouseMoveEvent::Modifier::Shift) {
                _arcball->viewBottom(2 * sideLength);
                _arcball->translateToOrigin();
            }

            }
            break;

        case Platform::GlfwApplication::KeyEvent::Key::D: {
            if(event.modifiers() & Platform::GlfwApplication::MouseMoveEvent::Modifier::Ctrl) {
                showDiscretizationGrid(!showingDiscretizationGrid());
            }
            else {
                decorateScene(!sceneDecorated());
            }

            }
            break;

        case Platform::GlfwApplication::KeyEvent::Key::F: {
            if(event.modifiers() & Platform::GlfwApplication::MouseMoveEvent::Modifier::Shift) {
                _arcball->viewFront(2 * sideLength);
                _arcball->translateToOrigin();
            }

            }
            break;

        case Platform::GlfwApplication::KeyEvent::Key::K: {
            if(event.modifiers() & Platform::GlfwApplication::MouseMoveEvent::Modifier::Shift) {
                _arcball->viewBack(2 * sideLength);
                _arcball->translateToOrigin();
            }

            }
            break;

        case Platform::GlfwApplication::KeyEvent::Key::L: {
            if(event.modifiers() & Platform::GlfwApplication::MouseMoveEvent::Modifier::Shift) {
                _arcball->viewLeft(2 * sideLength);
                _arcball->translateToOrigin();
            }
            else {
                toggleLagging();
            }

            }
            break;

        case Platform::GlfwApplication::KeyEvent::Key::R: {
            if(event.modifiers() & Platform::GlfwApplication::MouseMoveEvent::Modifier::Shift) {
                _arcball->viewRight(2 * sideLength);
                _arcball->translateToOrigin();
            }
            else {
                _arcball->reset();
            }

            }
            break;
            
        case Platform::GlfwApplication::KeyEvent::Key::T: {
            if(event.modifiers() & Platform::GlfwApplication::MouseMoveEvent::Modifier::Shift) {
                _arcball->viewTop(2 * sideLength);
                _arcball->translateToOrigin();
            }

            }
            break;
            
        case Platform::GlfwApplication::KeyEvent::Key::Down: {
            if(event.modifiers() & Platform::GlfwApplication::MouseMoveEvent::Modifier::Ctrl) {
                if(event.modifiers() & Platform::GlfwApplication::MouseMoveEvent::Modifier::Shift) {
                    this->cameraTranslateBackward();
                }
                else {
                    this->cameraZoomOut();
                }
            }
            else if(event.modifiers() & Platform::GlfwApplication::MouseMoveEvent::Modifier::Shift) {
                this->cameraRotateDown();
            }
            else {
                this->cameraTranslateDown();
            }
            
            }
            break;
            
        case Platform::GlfwApplication::KeyEvent::Key::Left: {
            if(event.modifiers() & Platform::GlfwApplication::MouseMoveEvent::Modifier::Ctrl) {
                this->cameraRollLeft();
            }
            else if(event.modifiers() & Platform::GlfwApplication::MouseMoveEvent::Modifier::Shift) {
                this->cameraRotateLeft();
            }
            else {
                this->cameraTranslateLeft();
            }
            
            }
            break;
            
        case Platform::GlfwApplication::KeyEvent::Key::Right: {
            if(event.modifiers() & Platform::GlfwApplication::MouseMoveEvent::Modifier::Ctrl) {
                this->cameraRollRight();
            }
            else if(event.modifiers() & Platform::GlfwApplication::MouseMoveEvent::Modifier::Shift) {
                this->cameraRotateRight();
            }
            else {
                this->cameraTranslateRight();
            }
            
            }
            break;
            
        case Platform::GlfwApplication::KeyEvent::Key::Up: {
            if(event.modifiers() & Platform::GlfwApplication::MouseMoveEvent::Modifier::Ctrl) {
                if(event.modifiers() & Platform::GlfwApplication::MouseMoveEvent::Modifier::Shift) {
                    this->cameraTranslateForward();
                }
                else {
                    this->cameraZoomIn();
                }
            }
            else if(event.modifiers() & Platform::GlfwApplication::MouseMoveEvent::Modifier::Shift) {
                this->cameraRotateUp();
            }
            else {
                this->cameraTranslateUp();
            }
            
            }
            break;

        default: return;
    }

    event.setAccepted();
    window->redraw();
}

void rendering::UniverseRenderer::mousePressEvent(Platform::GlfwApplication::MouseEvent& event) {
    /* Enable mouse capture so the mouse can drag outside of the window */
    /** @todo replace once https://github.com/mosra/magnum/pull/419 is in */
    //SDL_CaptureMouse(SDL_TRUE);

    _arcball->initTransformation(event.position());

    event.setAccepted();
    window->redraw(); /* camera has changed, redraw! */

}

void rendering::UniverseRenderer::mouseReleaseEvent(Platform::GlfwApplication::MouseEvent& event) {

}

void rendering::UniverseRenderer::mouseMoveEvent(Platform::GlfwApplication::MouseMoveEvent& event) {
    if(!event.buttons()) return;

    if(event.modifiers() & Platform::GlfwApplication::MouseMoveEvent::Modifier::Shift) {
        _arcball->translate(event.position());
    }
    else if(event.modifiers() & Platform::GlfwApplication::MouseEvent::Modifier::Ctrl) {
        if(!cameraZoom(_arcball, - _zoomRate * event.relativePosition().y() * 0.1)) return;
    }
    else {
        _arcball->rotate(event.position());
    }

    event.setAccepted();
    window->redraw(); /* camera has changed, redraw! */
}

void rendering::UniverseRenderer::mouseScrollEvent(Platform::GlfwApplication::MouseScrollEvent& event) {
    if(!cameraZoom(_arcball, _zoomRate * event.offset().y())) return;

    event.setAccepted();
    window->redraw(); /* camera has changed, redraw! */
}

rendering::SubRenderer *rendering::UniverseRenderer::getSubRenderer(const rendering::SubRendererFlag &flag) {
    if(subRenderers.size() == 0) return 0;

    switch(flag) {
        case SUBRENDERER_ANGLE: {
            return subRenderers[0];
            }
            break;
        case SUBRENDERER_ARROW: {
            return subRenderers[1];
            }
            break;
        case SUBRENDERER_BOND: {
            return subRenderers[2];
            }
            break;
        case SUBRENDERER_DIHEDRAL: {
            return subRenderers[3];
            }
            break;
        case SUBRENDERER_ORIENTATION: {
            return subRenderers[4];
            }
            break;
        default: {
            TF_Log(LOG_DEBUG) << "No renderer for flag " << (unsigned int)flag;
            return 0;
        }
    }
}

HRESULT rendering::UniverseRenderer::registerSubRenderer(rendering::SubRenderer *subrenderer) {
    std::vector<fVector4> cps(_clipPlanes.begin(), _clipPlanes.end());
    if(subrenderer->start(cps) != S_OK) 
        return E_FAIL;
    for(auto &cp : _clipPlanes) 
        subrenderer->addClipPlaneEquation(cp);
    subRenderers.push_back(subrenderer);
    return S_OK;
}

int rendering::UniverseRenderer::clipPlaneCount() const {
    return _clipPlanes.size();
}

int rendering::UniverseRenderer::maxClipPlaneCount() {
    return numClipDistanceGLLabels;
}

const unsigned rendering::UniverseRenderer::addClipPlaneEquation(const Magnum::Vector4& pe) {
    if(_clipPlanes.size() == numClipDistanceGLLabels) {
        tf_exp(std::invalid_argument("only up to 8 clip planes supported"));
    }

    GL::Renderer::enable(clipDistanceGLLabels[_clipPlanes.size()]);

    unsigned int id = _clipPlanes.size();

    _clipPlanes.push_back(pe);

    sphereShader.setclipPlaneEquation(id, pe);

    for(auto &s : subRenderers) 
        s->addClipPlaneEquation(pe);

    return id;
}

const unsigned rendering::UniverseRenderer::removeClipPlaneEquation(const unsigned int &id) {
    if(id >= _clipPlanes.size()) {
        tf_exp(std::invalid_argument("invalid id for clip plane"));
    }

    _clipPlanes.erase(_clipPlanes.begin() + id);

    GL::Renderer::disable(clipDistanceGLLabels[_clipPlanes.size()]);

    for(unsigned int i = id; i < _clipPlanes.size(); i++) {
        sphereShader.setclipPlaneEquation(i, _clipPlanes[i]);
    }

    for(auto &s : subRenderers) 
        s->removeClipPlaneEquation(id);

    return _clipPlanes.size();
}

void rendering::UniverseRenderer::setClipPlaneEquation(unsigned id, const Magnum::Vector4& pe) {
    if(id > sphereShader.clipPlaneCount()) {
        tf_exp(std::invalid_argument("invalid id for clip plane"));
    }
    
    sphereShader.setclipPlaneEquation(id, pe);

    for(auto &s : subRenderers) 
        s->setClipPlaneEquation(id, pe);

    _clipPlanes[id] = pe;
}

const Magnum::Vector4& rendering::UniverseRenderer::getClipPlaneEquation(unsigned id) {
    return _clipPlanes[id];
}

const float rendering::UniverseRenderer::getZoomRate() {
    return _zoomRate;
}

void rendering::UniverseRenderer::setZoomRate(const float &zoomRate) {
    if(zoomRate <= 0.0 || zoomRate >= 1.0) {
        tf_exp(std::invalid_argument("invalid zoom rate (0, 1.0)"));
    }
    _zoomRate = zoomRate;
}

const float rendering::UniverseRenderer::getSpinRate() {
    return _spinRate;
}

void rendering::UniverseRenderer::setSpinRate(const float &spinRate) {
    _spinRate = spinRate;
}

const float rendering::UniverseRenderer::getMoveRate() {
    return _moveRate;
}

void rendering::UniverseRenderer::setMoveRate(const float &moveRate) {
    _moveRate = moveRate;
}

const bool rendering::UniverseRenderer::isLagging() const {
    return getLagging() > 0;
}

void rendering::UniverseRenderer::enableLagging() {
    TF_Log(LOG_INFORMATION) << "Lagging enabled";

    setLagging(_lagging);
}

void rendering::UniverseRenderer::disableLagging() {
    TF_Log(LOG_INFORMATION) << "Lagging disabled";

    setLagging(0.f);
}

void rendering::UniverseRenderer::toggleLagging() {
    if(isLagging()) 
        disableLagging();
    else 
        enableLagging();
}

const float rendering::UniverseRenderer::getLagging() const {
    return _arcball->lagging();
}

void rendering::UniverseRenderer::setLagging(const float &lagging) {
    if(lagging < 0 || lagging >= 1.0) 
        TF_Log(LOG_ERROR) << "Invalid input: lagging must be in [0, 1)";
    else 
        _arcball->setLagging(lagging);
}

const bool rendering::UniverseRenderer::getRendering3DBonds() const {
    return _bonds3d_flags[0];
}

void rendering::UniverseRenderer::setRendering3DBonds(const bool &_flag) {
    if(_flag == _bonds3d_flags[0]) 
        return;
    _bonds3d_flags[0] = _flag;

    std::vector<FVector4> clipPlanes;
    for(auto &cp : _clipPlanes) 
        clipPlanes.push_back(cp);

    delete subRenderers[2];
    SubRenderer *renderer;
    if(_flag) renderer = new BondRenderer3D();
    else renderer = new BondRenderer();

    renderer->start(clipPlanes);
    subRenderers[2] = renderer;
}

void rendering::UniverseRenderer::toggleRendering3DBonds() {
    setRendering3DBonds(!getRendering3DBonds());
}

const bool rendering::UniverseRenderer::getRendering3DAngles() const {
    return _bonds3d_flags[1];
}

void rendering::UniverseRenderer::setRendering3DAngles(const bool &_flag) {
    if(_flag == _bonds3d_flags[1]) 
        return;
    _bonds3d_flags[1] = _flag;

    std::vector<FVector4> clipPlanes;
    for(auto &cp : _clipPlanes) 
        clipPlanes.push_back(cp);

    delete subRenderers[0];
    SubRenderer *renderer;
    if(_flag) renderer = new AngleRenderer3D();
    else renderer = new AngleRenderer();

    renderer->start(clipPlanes);
    subRenderers[0] = renderer;
}

void rendering::UniverseRenderer::toggleRendering3DAngles() {
    setRendering3DAngles(!getRendering3DAngles());
}

const bool rendering::UniverseRenderer::getRendering3DDihedrals() const {
    return _bonds3d_flags[2];
}

void rendering::UniverseRenderer::setRendering3DDihedrals(const bool &_flag) {
    if(_flag == _bonds3d_flags[2]) 
        return;
    _bonds3d_flags[2] = _flag;

    std::vector<FVector4> clipPlanes;
    for(auto &cp : _clipPlanes) 
        clipPlanes.push_back(cp);

    delete subRenderers[3];
    SubRenderer *renderer;
    if(_flag) renderer = new DihedralRenderer3D();
    else renderer = new DihedralRenderer();

    renderer->start(clipPlanes);
    subRenderers[3] = renderer;
}

void rendering::UniverseRenderer::toggleRendering3DDihedrals() {
    setRendering3DDihedrals(!getRendering3DDihedrals());
}

void rendering::UniverseRenderer::setRendering3DAll(const bool &_flag) {
    setRendering3DBonds(_flag);
    setRendering3DAngles(_flag);
    setRendering3DDihedrals(_flag);
}

void rendering::UniverseRenderer::toggleRendering3DAll() {
    toggleRendering3DBonds();
    toggleRendering3DAngles();
    toggleRendering3DDihedrals();
}
