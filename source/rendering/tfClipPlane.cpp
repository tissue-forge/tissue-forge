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

#include "tfClipPlane.h"
#include <Magnum/Math/Distance.h>
#include <tf_system.h>
#include "tfUniverseRenderer.h"
#include <tfError.h>
#include <tfLogger.h>


using namespace TissueForge;


rendering::ClipPlanes _clipPlanesObj;

rendering::ClipPlane::ClipPlane(int i) : index(i) {}

std::vector<fVector4> TissueForge::parsePlaneEquation(const std::vector<std::tuple<fVector3, fVector3> > &clipPlanes) {
    std::vector<fVector4> result(clipPlanes.size());
    fVector3 point;
    fVector3 normal;

    for(unsigned int i = 0; i < clipPlanes.size(); ++i) {
        std::tie(point, normal) = clipPlanes[i];
        result[i] = planeEquation(normal, point);
    }
    return result;
}

fVector3 rendering::ClipPlane::getPoint() {
    auto eq = system::getRenderer()->getClipPlaneEquation(this->index);
    return std::get<1>(planeEquation(eq));
}

fVector3 rendering::ClipPlane::getNormal() {
    auto eq = system::getRenderer()->getClipPlaneEquation(this->index);
    return std::get<0>(planeEquation(eq));
}

fVector4 rendering::ClipPlane::getEquation() {
    return system::getRenderer()->getClipPlaneEquation(this->index);
}

HRESULT rendering::ClipPlane::setEquation(const fVector4 &pe) {
    system::getRenderer()->setClipPlaneEquation(this->index, pe);
    return S_OK;
}

HRESULT rendering::ClipPlane::setEquation(const fVector3 &point, const fVector3 &normal) {
    return this->setEquation(planeEquation(normal, point));
}

HRESULT rendering::ClipPlane::destroy() {
    if(this->index < 0) {
        TF_Log(LOG_CRITICAL) << "Clip plane no longer valid";
        return E_FAIL;
    }

    system::getRenderer()->removeClipPlaneEquation(this->index);
    this->index = -1;
    return S_OK;
}

int rendering::ClipPlanes::len() {
    return system::getRenderer()->clipPlaneCount();
}

const fVector4 &rendering::ClipPlanes::getClipPlaneEquation(const unsigned int &index) {
    try {
        rendering::UniverseRenderer *renderer = system::getRenderer();
        
        if(index > renderer->clipPlaneCount()) tf_exp(std::range_error("index out of bounds"));
        return fVector4::from(renderer->getClipPlaneEquation(index).data());
    }
    catch(const std::exception &e) {
        tf_error(E_FAIL, e.what());
        fVector4 *result = new fVector4();
        return *result;
    }
}

HRESULT rendering::ClipPlanes::setClipPlaneEquation(const unsigned int &index, const fVector4 &pe) {
    try {
        rendering::UniverseRenderer *renderer = system::getRenderer();
        if(index > renderer->clipPlaneCount()) tf_exp(std::range_error("index out of bounds"));
        renderer->setClipPlaneEquation(index, pe);
        return S_OK;
    }
    catch(const std::exception &e) {
        tf_error(E_FAIL, e.what());
        return -1;
    }
}

rendering::ClipPlane rendering::ClipPlanes::item(const unsigned int &index) {
    return rendering::ClipPlane(index);
}

rendering::ClipPlane rendering::ClipPlanes::create(const fVector4 &pe) {
    rendering::ClipPlane result(rendering::ClipPlanes::len());
    system::getRenderer()->addClipPlaneEquation(pe);
    return result;
}

rendering::ClipPlane rendering::ClipPlanes::create(const fVector3 &point, const fVector3 &normal) {
    return rendering::ClipPlanes::create(planeEquation(normal, point));
}

rendering::ClipPlanes *rendering::getClipPlanes() {
    return &_clipPlanesObj;
}
