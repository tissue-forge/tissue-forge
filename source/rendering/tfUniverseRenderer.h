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

/**
 * @file tfUniverseRenderer.h
 * 
 */

#pragma once

#include <vector>

#include <Magnum/GL/Buffer.h>
#include <Magnum/GL/Mesh.h>

#include <tfUniverse.h>
#include <tfSimulator.h>
#include "tfRenderer.h"
#include "tfGlfwWindow.h"
#include <shaders/tfParticleSphereShader.h>

#include <shaders/tfPhong.h>

#include <Corrade/Containers/Pointer.h>

#include <Corrade/Containers/Pointer.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Primitives/Cube.h>
#include <Magnum/Platform/GlfwApplication.h>
#include <Magnum/SceneGraph/MatrixTransformation3D.h>
#include <Magnum/Timeline.h>

#include <Magnum/Shaders/Phong.h>
#include <Magnum/Shaders/Flat.h>

#include "tfGlfwWindow.h"

#include <Magnum/Platform/GlfwApplication.h>

#include "tfWindow.h"
#include "tfArcBallCamera.h"

#include "tfSubRenderer.h"


namespace TissueForge {


    struct Simulator;


    namespace rendering {


        class WireframeGrid;
        class WireframeBox;

        struct SphereInstanceData {
            Magnum::Matrix4 transformationMatrix;
            Magnum::Matrix3x3 normalMatrix;
            Magnum::Color4 color;
        };

        struct BondsInstanceData {
            Magnum::Vector3 position;
            Magnum::Color4 color;
        };


        typedef enum SubRendererFlag {
            SUBRENDERER_ANGLE           = 1 << 0,
            SUBRENDERER_ARROW           = 1 << 1,
            SUBRENDERER_BOND            = 1 << 2,
            SUBRENDERER_DIHEDRAL        = 1 << 3, 
            SUBRENDERER_ORIENTATION     = 1 << 4
        } SubRendererFlag;


        struct UniverseRenderer : Renderer {


            // TODO, implement the event system instead of hard coding window events.
            UniverseRenderer(const Simulator::Config &conf, Window *win);

            template<typename T>
            UniverseRenderer& draw(T& camera, const iVector2& viewportSize);

            bool& isDirty() { return _dirty; }

            UniverseRenderer& setDirty() {
                _dirty = true;
                return *this;
            }

            shaders::ParticleSphereShader::ColorMode& colorMode() { return _colorMode; }

            UniverseRenderer& setColorMode(shaders::ParticleSphereShader::ColorMode colorMode) {
                _colorMode = colorMode;
                return *this;
            }

            const Float lineWidth();

            UniverseRenderer& setLineWidth(const Float &lw);

            const Float lineWidthMin();

            const Float lineWidthMax();

            Color3& ambientColor() { return _ambientColor; }

            UniverseRenderer& setAmbientColor(const Color3& color);

            Color3& diffuseColor() { return _diffuseColor; }

            UniverseRenderer& setDiffuseColor(const Color3& color);

            Color3& specularColor() { return _specularColor; }

            UniverseRenderer& setSpecularColor(const Color3& color);

            Float& shininess() { return _shininess; }

            UniverseRenderer& setShininess(float shininess);

            Color3& gridColor() { return _gridColor; }

            UniverseRenderer& setGridColor(const Color3 &color) {
                _gridColor = color;
                return *this;
            }

            Color3& sceneBoxColor() { return _sceneBoxColor; }

            UniverseRenderer& setSceneBoxColor(const Color3 &color) {
                _sceneBoxColor = color;
                return *this;
            }

            fVector3& lightDirection() { return _lightDir; }

            UniverseRenderer& setLightDirection(const fVector3& lightDir);

            Color3& lightColor() { return _lightColor; }

            UniverseRenderer& setLightColor(const Color3 &color);

            Color3& backgroundColor() { return _clearColor; }

            UniverseRenderer& setBackgroundColor(const Color3 &color);

            UniverseRenderer& setModelViewTransform(const Magnum::Matrix4& mat) {
                modelViewMat = mat;
                return *this;
            }

            UniverseRenderer& setProjectionTransform(const Magnum::Matrix4& mat) {
                projMat = mat;
                return *this;
            }

            const bool showingDiscretizationGrid() const {
                return _showDiscretizationGrid;
            }

            UniverseRenderer& showDiscretizationGrid(const bool &show) {
                _showDiscretizationGrid = show;
                return *this;
            }

            Color3& discretizationGridColor() {
                return _discretizationGridColor;
            }

            UniverseRenderer& setDiscretizationGridColor(const Color3 &color);
            
            const fVector3& defaultEye() const {
                return _eye;
            }
            
            const fVector3& defaultCenter() const {
                return _center;
            }
            
            const fVector3& defaultUp() const {
                return _up;
            }

            bool renderUniverse = true;


            void onCursorMove(double xpos, double ypos);

            void onCursorEnter(int entered);

            void onMouseButton(int button, int action, int mods);

            void onRedraw();

            void onWindowMove(int x, int y);

            void onWindowSizeChange(int x, int y);

            void onFramebufferSizeChange( int x, int y);

            void viewportEvent(const int w, const int h);

            void draw();
            
            int clipPlaneCount() const;

            static int maxClipPlaneCount();
            
            const unsigned addClipPlaneEquation(const Magnum::Vector4& pe);
            
            const unsigned removeClipPlaneEquation(const unsigned int &id);
            
            void setClipPlaneEquation(unsigned id, const Magnum::Vector4& pe);
            
            const Magnum::Vector4& getClipPlaneEquation(unsigned id);

            const float getZoomRate();

            void setZoomRate(const float &zoomRate);

            const float getSpinRate();

            void setSpinRate(const float &spinRate);

            const float getMoveRate();

            void setMoveRate(const float &moveRate);

            /** Test whether the camera is lagging */
            const bool isLagging() const;

            /** Enable camera lagging */
            void enableLagging();

            /** Disable camera lagging */
            void disableLagging();

            /** Toggle camera lagging */
            void toggleLagging();

            /** Get the camera lagging */
            const float getLagging() const;

            /** Set the camera lagging. Value must be in [0, 1) */
            void setLagging(const float &lagging);

            /** Get whether bonds are renderered with 3D objects */
            const bool getRendering3DBonds() const;

            /** Set whether bonds are renderered with 3D objects */
            void setRendering3DBonds(const bool &_flag);

            /** Toggle whether bonds are renderered with 3D objects */
            void toggleRendering3DBonds();

            /** Get whether angles are renderered with 3D objects */
            const bool getRendering3DAngles() const;

            /** Set whether angles are renderered with 3D objects */
            void setRendering3DAngles(const bool &_flag);

            /** Toggle whether angles are renderered with 3D objects */
            void toggleRendering3DAngles();

            /** Get whether dihedrals are renderered with 3D objects */
            const bool getRendering3DDihedrals() const;

            /** Set whether dihedrals are renderered with 3D objects */
            void setRendering3DDihedrals(const bool &_flag);

            /** Toggle whether dihedrals are renderered with 3D objects */
            void toggleRendering3DDihedrals();

            /** Set whether bonds, angle and dihedrals are renderered with 3D objects */
            void setRendering3DAll(const bool &_flag);

            /** Toggle whether bonds, angle and dihedrals are renderered with 3D objects */
            void toggleRendering3DAll();

            void viewportEvent(Platform::GlfwApplication::ViewportEvent& event);

            void cameraTranslateDown();

            void cameraTranslateUp();

            void cameraTranslateRight();

            void cameraTranslateLeft();

            void cameraTranslateForward();

            void cameraTranslateBackward();

            void cameraRotateDown();

            void cameraRotateUp();

            void cameraRotateLeft();

            void cameraRotateRight();

            void cameraRollLeft();

            void cameraRollRight();

            void cameraZoomIn();

            void cameraZoomOut();

            /**
            * @brief Key press event handling. 
            * 
            * @details When a key is pressed, actions are as follows by key and modifier: 
            * 
            * - D: toggle scene decorations
            * - L: toggle lagging
            * - R: reset camera
            * - Arrow down: translate camera down
            * - Arrow left: translate camera left
            * - Arrow right: translate camera right
            * - Arrow up: translate camera up
            * - Ctrl + D: toggle discretization rendering
            * - Ctrl + arrow down: zoom camera out
            * - Ctrl + arrow left: roll camera left
            * - Ctrl + arrow right: roll camera right
            * - Ctrl + arrow up: zoom camera in
            * - Shift + B: bottom view
            * - Shift + F: front view
            * - Shift + K: back view
            * - Shift + L: left view
            * - Shift + R: right view
            * - Shift + T: top view
            * - Shift + arrow down: rotate camera down
            * - Shift + arrow left: rotate camera left
            * - Shift + arrow right: rotate camera right
            * - Shift + arrow up: rotate camera up
            * - Shift + Ctrl + arrow down: translate backward
            * - Shift + Ctrl + arrow up: translate forward
            * 
            * @param event 
            */
            void keyPressEvent(Platform::GlfwApplication::KeyEvent& event);
            void mousePressEvent(Platform::GlfwApplication::MouseEvent& event);
            void mouseReleaseEvent(Platform::GlfwApplication::MouseEvent& event);

            /**
            * @brief Mouse move event handling. 
            * 
            * @details When a mouse button is pressed, actions are as follows by modifier:
            *  - Shift: translates the camera
            *  - Ctrl: zooms the camera
            *  - None: rotates the camera
            *  .
            * 
            * @param event 
            */
            void mouseMoveEvent(Platform::GlfwApplication::MouseMoveEvent& event);
            void mouseScrollEvent(Platform::GlfwApplication::MouseScrollEvent& event);

            SubRenderer *getSubRenderer(const SubRendererFlag &flag);
            HRESULT registerSubRenderer(SubRenderer *subrenderer);

            bool _dirty = false;
            bool _decorateScene = true;
            bool _showDiscretizationGrid = false;
            shaders::ParticleSphereShader::ColorMode _colorMode = shaders::ParticleSphereShader::ColorMode::ConsistentRandom;
            Color3 _ambientColor{0.4f};
            Color3 _diffuseColor{1.f};
            Color3 _specularColor{0.2f};
            Color3 _gridColor = {1.f, 1.f, 1.f};
            Color3 _sceneBoxColor = {1.f, 1.f, 0.f};
            Float _shininess = 100.0f;
            fVector3 _lightDir{1.0f, 1.0f, 2.0f};
            Color3 _lightColor = {0.9, 0.9, 0.9};
            Color3 _clearColor{0.35f};
            Color3 _discretizationGridColor{0.1, 0.1, 0.8};
            
            fVector3 _eye, _center, _up;
            
            std::vector<Magnum::Vector4> _clipPlanes;
            
            /**
            * Only set a single combined matrix in the shader, this way,
            * the shader only performs a single matrix multiply of the vertices, update the
            * shader matrix whenever any of these change.
            *
            * multiplication order is the reverse of the pipeline.
            * Therefore you do totalmat = proj * view * model.
            */
            Magnum::Matrix4 modelViewMat = Matrix4{Math::IdentityInit};
            Magnum::Matrix4 projMat =  Matrix4{Math::IdentityInit};

            iVector2 _prevMousePosition;
            fVector3  _rotationPoint, _translationPoint;
            Float _lastDepth;
            
            float sideLength;
            float _zoomRate;
            float _spinRate;
            float _moveRate;

            ArcBallCamera *_arcball;
            
            /* ground grid */
            GL::Mesh gridMesh{NoCreate};
            Magnum::Matrix4 gridModelView;
            
            GL::Mesh sceneBox{NoCreate};

            
            /* Spheres rendering */
            
            shaders::Phong sphereShader{NoCreate};
            
            Shaders::Flat3D wireframeShader{NoCreate};
            
            GL::Buffer sphereInstanceBuffer{NoCreate};
            
            GL::Buffer largeSphereInstanceBuffer{NoCreate};

            GL::Mesh sphereMesh{NoCreate};

            GL::Mesh largeSphereMesh{NoCreate};

            GL::Mesh discretizationGridMesh{NoCreate};

            GL::Buffer discretizationGridBuffer{NoCreate};

            std::vector<SubRenderer*> subRenderers;

            fVector3 center;

            Window *window;

            /**
            * @brief Set flag to draw/not draw scene decorators (e.g., grid)
            * 
            * @param decorate flag; true says to decorate
            */
            void decorateScene(const bool &decorate);

            /**
            * @brief Get scene decorator flag value
            * 
            * @return true 
            * @return false 
            */
            bool sceneDecorated() const;

            fVector3 unproject(const iVector2& windowPosition, float depth) const;

            // todo: implement UniverseRenderer::setupCallbacks
            void setupCallbacks();
            
            ~UniverseRenderer();

        protected:

            float _lagging;
            bool _bonds3d_flags[3] = {false, false, false}; // bonds, angles, dihedrals
        };

}};
