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
        2020 — Nghia Truong <nghiatruong.vn@gmail.com>

    This is free and unencumbered software released into the public domain.

    Anyone is free to copy, modify, publish, use, compile, sell, or distribute
    this software, either in source code form or as a compiled binary, for any
    purpose, commercial or non-commercial, and by any means.

    In jurisdictions that recognize copyright laws, the author or authors of
    this software dedicate any and all copyright interest in the software to
    the public domain. We make this dedication for the benefit of the public
    at large and to the detriment of our heirs and successors. We intend this
    dedication to be an overt act of relinquishment in perpetuity of all
    present and future rights to this software under copyright law.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
    IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
    CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef Magnum_Examples_ArcBallCamera_h
#define Magnum_Examples_ArcBallCamera_h

#include <Corrade/Containers/EnumSet.h>
#include <Corrade/Containers/LinkedList.h>

#include <Magnum/Magnum.h>
#include <limits>

#include "tfArcBall.h"


namespace TissueForge::rendering { 


    /* Arcball camera implementation integrated into the SceneGraph */
    class ArcBallCamera : public ArcBall {
    public:
        
        enum class AspectRatioPolicy : UnsignedByte {
            NotPreserved,   /**< Don't preserve aspect ratio (default) */
            Extend,         /**< Extend on larger side of view */
            Clip            /**< Clip on smaller side of view */
        };
        
        ArcBallCamera(
            const Vector3& cameraPosition, const Vector3& viewCenter,
            const Vector3& upDir, Deg fov, const Magnum::Vector2i& windowSize,
            const Magnum::Vector2i& viewportSize,
            float nearClip = 0.01f, float farClip = std::numeric_limits<float>::infinity()
        ) :
            ArcBall{cameraPosition, viewCenter, upDir, fov, windowSize}
        {
            setAspectRatioPolicy(AspectRatioPolicy::Extend);
            setProjectionMatrix(Matrix4::perspectiveProjection(fov, Vector2{windowSize}.aspectRatio(), nearClip, farClip));
            setViewport(viewportSize);
        }
        
        /* Update screen and viewport size after the window has been resized */
        void reshape(const Magnum::Vector2i& windowSize, const Magnum::Vector2i& viewportSize) {
            _windowSize = windowSize;
            setViewport(viewportSize);
        }
        
        auto cameraMatrix() {
            return viewMatrix();
        }
        
        auto projectionMatrix() {
            return _projectionMatrix;
        }

        void rotateDelta(const float *deltaX=NULL, const float *deltaY=NULL, const float *deltaZ=NULL) {
            if(deltaZ) _targetQRotation = Magnum::Quaternion::rotation(Magnum::Rad(*deltaZ), Magnum::Vector3::zAxis()) * _targetQRotation;
            if(deltaY) _targetQRotation = Magnum::Quaternion::rotation(Magnum::Rad(*deltaY), Magnum::Vector3::yAxis()) * _targetQRotation;
            if(deltaX) _targetQRotation = Magnum::Quaternion::rotation(Magnum::Rad(*deltaX), Magnum::Vector3::xAxis()) * _targetQRotation;
        }

        void translateDelta(const Vector3 &deltaPos, const bool &absolute=false) {
            if(absolute) _targetPosition += deltaPos;
            else _targetPosition += _targetQRotation.inverted().transformVector(deltaPos);
        }

        void translateToOrigin() {
            _targetPosition = Vector3(0.0);
        }

        void viewBottom(const float &viewDistance) {
            rotateToAxis(Vector3::xAxis(-1.0f), viewDistance);
            rotateByEulerAngles({0, 0, M_PI});
        }

        void viewTop(const float &viewDistance) {
            rotateToAxis(Vector3::xAxis(), viewDistance);
        }

        void viewLeft(const float &viewDistance) {
            viewTop(viewDistance);
            rotateByEulerAngles({-0.5f*M_PI, 0, 0.5f*M_PI});
        }

        void viewRight(const float &viewDistance) {
            viewLeft(viewDistance);
            rotateByEulerAngles({0, 0, M_PI});
        }

        void viewBack(const float &viewDistance) {
            viewTop(viewDistance);
            rotateByEulerAngles({-0.5f*M_PI, 0, M_PI});
        }

        void viewFront(const float &viewDistance) {
            viewBack(viewDistance);
            rotateByEulerAngles({0, 0, M_PI});
        }

        fVector3 cposition() {
            return _currentPosition;
        }

        fQuaternion crotation() {
            return _currentQRotation;
        }

        float czoom() {
            return _currentZooming;
        }

        void setViewParameters(const fVector3 &position, const fQuaternion &rotation, const float &zoom) {
            _targetPosition = position;
            _targetQRotation = rotation;
            _targetZooming = zoom;
        }
        
    private:
        
        Magnum::Matrix4 _projectionMatrix;
        
        Magnum::Matrix4 _rawProjectionMatrix;
        
        AspectRatioPolicy _aspectRatioPolicy;
        
        Magnum::Vector2i _viewport;
        
        void setViewport(const Magnum::Vector2i& size) {
            _viewport = size;
            fixAspectRatio();
        }
        
        void setProjectionMatrix(const Matrix4& matrix) {
            _rawProjectionMatrix = matrix;
            fixAspectRatio();
        }
        
        void setAspectRatioPolicy(AspectRatioPolicy policy) {
            _aspectRatioPolicy = policy;
            fixAspectRatio();
        }
        
        static Matrix4 aspectRatioFix(AspectRatioPolicy aspectRatioPolicy, const Vector2& projectionScale, const Magnum::Vector2i& viewport) {
            
            /* Don't divide by zero / don't preserve anything */
            if(projectionScale.x() == 0 || projectionScale.y() == 0 || viewport.x() == 0 || viewport.y() == 0 || aspectRatioPolicy == AspectRatioPolicy::NotPreserved)
                return {};
            
            CORRADE_INTERNAL_ASSERT((projectionScale > Vector2(0)).all() && (viewport > Magnum::Vector2i(0)).all());
            
            Vector2 relativeAspectRatio = Vector2(viewport)*projectionScale;
            
            /* Extend on larger side = scale larger side down
            Clip on smaller side = scale smaller side up */
            return Matrix4::scaling(Vector3::pad(
                                                (relativeAspectRatio.x() > relativeAspectRatio.y()) == (aspectRatioPolicy == AspectRatioPolicy::Extend) ?
                                                Vector2(relativeAspectRatio.y()/relativeAspectRatio.x(), float(1)) :
                                                Vector2(float(1), relativeAspectRatio.x()/relativeAspectRatio.y()), float(1)));
            
        };
        
        void fixAspectRatio() {
            _projectionMatrix = aspectRatioFix(_aspectRatioPolicy, {Math::abs(_rawProjectionMatrix[0].x()), Math::abs(_rawProjectionMatrix[1].y())}, _viewport)*_rawProjectionMatrix;
        }
        
    };

}

#endif
