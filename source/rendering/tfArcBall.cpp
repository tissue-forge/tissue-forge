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

/*
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

#include "tfArcBall.h"

#include <Magnum/Math/Matrix3.h>


using namespace Magnum;


namespace TissueForge::rendering {

namespace {

/* Project a point in NDC onto the arcball sphere */
Quaternion ndcToArcBall(const Vector2& p) {
    const Float dist = Math::dot(p, p);

    /* Point is on sphere */
    if(dist <= 1.0f)
        return {{p.x(), p.y(), Math::sqrt(1.0f - dist)}, 0.0f};

    /* Point is outside sphere */
    else {
        const Vector2 proj = p.normalized();
        return {{proj.x(), proj.y(), 0.0f}, 0.0f};
    }
}

}

ArcBall::ArcBall(const Vector3& eye, const Vector3& viewCenter,
    const Vector3& upDir, Deg fov, const Vector2i& windowSize) : 
    _fov{fov}, _windowSize{windowSize}
{
    setViewParameters(eye, viewCenter, upDir);
}

void ArcBall::setViewParameters(const Vector3& eye, const Vector3& viewCenter,
    const Vector3& upDir)
{
    const Vector3 dir = viewCenter - eye;
    Vector3 zAxis = dir.normalized();
    Vector3 xAxis = (Math::cross(zAxis, upDir.normalized())).normalized();
    Vector3 yAxis = (Math::cross(xAxis, zAxis)).normalized();
    xAxis = (Math::cross(zAxis, yAxis)).normalized();

    _targetPosition = -viewCenter;
    _targetZooming = -dir.length();
    _targetQRotation = Quaternion::fromMatrix(
        Matrix3x3{xAxis, yAxis, -zAxis}.transposed()).normalized();

    _positionT0  = _currentPosition = _targetPosition;
    _zoomingT0 = _currentZooming = _targetZooming;
    _qRotationT0 = _currentQRotation = _targetQRotation;

    updateInternalTransformations();
}
    
/*
 * Set the camera view parameters: eye position, view center, up
 * direction, only rotates the view to the given eye position.
 */
void ArcBall::rotateToAxis(const Vector3& axis, float distance) {
    
    
    Vector3 zAxis;
    Vector3 xAxis;
    Vector3 yAxis;
        
    Vector3 naxis = axis.normalized();
    float dotx = Math::dot(naxis, Vector3::xAxis());
    float doty = Math::dot(naxis, Vector3::yAxis());
    float dotz = Math::dot(naxis, Vector3::zAxis());
        
    if(dotx > 0) {
        xAxis = Vector3::xAxis();
        yAxis = Vector3::yAxis();
        zAxis = Vector3::zAxis();
    }
    else if(dotx < 0) {
        xAxis = -Vector3::xAxis();
        yAxis = Vector3::yAxis();
        zAxis = Vector3::zAxis();
    }
    else {

    }
    
    // original position
    _targetPosition = _positionT0;
    _targetZooming = -distance;
    _targetQRotation = Quaternion::fromMatrix(
        Matrix3x3{xAxis, yAxis, -zAxis}.transposed()).normalized();
        
}

void ArcBall::reset() {
    _targetPosition = _positionT0;
    _targetZooming = _zoomingT0;
    _targetQRotation = _qRotationT0;
}

void ArcBall::setLagging(const Float lagging) {
    CORRADE_INTERNAL_ASSERT(lagging >= 0.0f && lagging < 1.0f);
    _lagging = lagging;
}

void ArcBall::initTransformation(const Vector2i& mousePos) {
    _prevMousePosNDC = screenCoordToNDC(mousePos);
}

void ArcBall::rotate(const Vector2i& mousePos) {
    const Vector2 mousePosNDC = screenCoordToNDC(mousePos);
    const Quaternion currentQRotation = ndcToArcBall(mousePosNDC);
    const Quaternion prevQRotation = ndcToArcBall(_prevMousePosNDC);
    _prevMousePosNDC = mousePosNDC;
    _targetQRotation =
        (currentQRotation*prevQRotation*_targetQRotation).normalized();
}

void ArcBall::translate(const Vector2i& mousePos) {
    const Vector2 mousePosNDC = screenCoordToNDC(mousePos);
    const Vector2 translationNDC = mousePosNDC - _prevMousePosNDC;
    _prevMousePosNDC = mousePosNDC;
    translateDelta(translationNDC);
}

void ArcBall::translateDelta(const Vector2& translationNDC) {
    /* Half size of the screen viewport at the view center and perpendicular
       with the viewDir */
    const Float hh = Math::abs(_targetZooming)*Math::tan(_fov*0.5f);
    const Float hw = hh*Vector2{_windowSize}.aspectRatio();

    _targetPosition += _inverseView.transformVector(
        {translationNDC.x()*hw, translationNDC.y()*hh, 0.0f});
}

void ArcBall::zoom(const Float delta) {
    _targetZooming += delta;
}

void ArcBall::zoomTo(const Float distance) {
    _targetZooming = distance;
}


bool ArcBall::updateTransformation() {
    const Vector3 diffViewCenter = _targetPosition - _currentPosition;
    const Quaternion diffRotation = _targetQRotation - _currentQRotation;
    const Float diffZooming = _targetZooming - _currentZooming;

    const Float dViewCenter = Math::dot(diffViewCenter, diffViewCenter);
    const Float dRotation = Math::dot(diffRotation, diffRotation);
    const Float dZooming = diffZooming * diffZooming;

    /* Nothing change */
    if(dViewCenter < 1.0e-10f &&
       dRotation < 1.0e-10f &&
       dZooming < 1.0e-10f) {
        return false;
    }

    /* Nearly done: just jump directly to the target */
    if(dViewCenter < 1.0e-6f &&
       dRotation < 1.0e-6f &&
       dZooming < 1.0e-6f) {
        _currentPosition  = _targetPosition;
        _currentQRotation = _targetQRotation;
        _currentZooming   = _targetZooming;

    /* Interpolate between the current transformation and the target
       transformation */
    } else {
        const Float t = 1 - _lagging;
        _currentPosition = Math::lerp(_currentPosition, _targetPosition, t);
        _currentZooming  = Math::lerp(_currentZooming, _targetZooming, t);
        _currentQRotation = Math::slerpShortestPath(
            _currentQRotation, _targetQRotation, t);
    }

    updateInternalTransformations();
    return true;
}

void ArcBall::updateInternalTransformations() {
    _view = DualQuaternion::translation(Vector3::zAxis(_currentZooming))*
            DualQuaternion{_currentQRotation}*
            DualQuaternion::translation(_currentPosition);
    _inverseView = _view.inverted();
}

Vector2 ArcBall::screenCoordToNDC(const Vector2i& mousePos) const {
    return {mousePos.x()*2.0f/_windowSize.x() - 1.0f,
            1.0f - 2.0f*mousePos.y()/ _windowSize.y()};
}


void ArcBall::rotateByEulerAngles(const Vector3& eulerAngles) {
    Magnum::Quaternion qx = Magnum::Quaternion::rotation(Magnum::Rad(eulerAngles[0]), Magnum::Vector3::xAxis());
    Magnum::Quaternion qy = Magnum::Quaternion::rotation(Magnum::Rad(eulerAngles[1]), Magnum::Vector3::yAxis());
    Magnum::Quaternion qz = Magnum::Quaternion::rotation(Magnum::Rad(eulerAngles[2]), Magnum::Vector3::zAxis());
    _targetQRotation = (_targetQRotation * qx * qy * qz).normalized();
    
}

void ArcBall::rotateToEulerAngles(const Vector3& eulerAngles) {
    Magnum::Quaternion qx = Magnum::Quaternion::rotation(Magnum::Rad(eulerAngles[0]), Magnum::Vector3::xAxis());
    Magnum::Quaternion qy = Magnum::Quaternion::rotation(Magnum::Rad(eulerAngles[1]), Magnum::Vector3::yAxis());
    Magnum::Quaternion qz = Magnum::Quaternion::rotation(Magnum::Rad(eulerAngles[2]), Magnum::Vector3::zAxis());
    _targetQRotation =  (qx * qy * qz).normalized();
    
}
 
}
