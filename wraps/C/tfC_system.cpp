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

#include "tfC_system.h"

#include "TissueForge_c_private.h"

#include <tf_system.h>


using namespace TissueForge;


HRESULT tfSystem_imageData(char **imgData, size_t *imgSize) {
    TFC_PTRCHECK(imgData);
    TFC_PTRCHECK(imgSize);
    char *_imgData;
    std::tie(_imgData, *imgSize) = system::imageData();
    *imgData = _imgData;
    return S_OK;
}

HRESULT tfSystem_screenshot(const char *filePath) {
    return tfSystem_screenshot(filePath);
}

HRESULT tfSystem_screenshotS(const char *filePath, bool decorate, float *bgcolor) {
    if(!bgcolor) 
        return E_FAIL;
    return system::screenshot(filePath, decorate, fVector3::from(bgcolor));
}

HRESULT tfSystem_contextHasCurrent(bool *current) {
    TFC_PTRCHECK(current);
    *current = system::contextHasCurrent();
    return S_OK;
}

HRESULT tfSystem_contextMakeCurrent() {
    return system::contextMakeCurrent();
}

HRESULT tfSystem_contextRelease() {
    return system::contextRelease();
}

bool tfSystem_cameraIsLagging() {
    return system::cameraIsLagging();
}

HRESULT tfSystem_cameraEnableLagging() {
    return system::cameraEnableLagging();
}

HRESULT tfSystem_cameraDisableLagging() {
    return system::cameraDisableLagging();
}

HRESULT tfSystem_cameraToggleLagging() {
    return system::cameraToggleLagging();
}

float tfSystem_cameraGetLagging() {
    return system::cameraGetLagging();
}

HRESULT tfSystem_cameraSetLagging(float lagging) {
    return system::cameraSetLagging(lagging);
}

HRESULT tfSystem_cameraMoveTo(float *eye, float *center, float *up) {
    if(!eye || !center || !up) 
        return E_FAIL;
    return system::cameraMoveTo(fVector3::from(eye), fVector3::from(center), fVector3::from(up));
}

HRESULT tfSystem_cameraMoveToR(float *center, float *rotation, float zoom) {
    if(!center || !rotation) 
        return E_FAIL;
    return system::cameraMoveTo(fVector3::from(center), fQuaternion{fVector3(rotation[0], rotation[1], rotation[2]), rotation[3]}, zoom);
}

HRESULT tfSystem_cameraViewBottom() {
    return system::cameraViewBottom();
}

HRESULT tfSystem_cameraViewTop() {
    return system::cameraViewTop();
}

HRESULT tfSystem_cameraViewLeft() {
    return system::cameraViewLeft();
}

HRESULT tfSystem_cameraViewRight() {
    return system::cameraViewRight();
}

HRESULT tfSystem_cameraViewBack() {
    return system::cameraViewBack();
}

HRESULT tfSystem_cameraViewFront() {
    return system::cameraViewFront();
}

HRESULT tfSystem_cameraReset() {
    return system::cameraReset();
}

HRESULT tfSystem_cameraRotateMouse(int x, int y) {
    return system::cameraRotateMouse({x, y});
}

HRESULT tfSystem_cameraTranslateMouse(int x, int y) {
    return system::cameraTranslateMouse({x, y});
}

HRESULT tfSystem_cameraTranslateDown() {
    return system::cameraTranslateDown();
}

HRESULT tfSystem_cameraTranslateUp() {
    return system::cameraTranslateUp();
}

HRESULT tfSystem_cameraTranslateRight() {
    return system::cameraTranslateRight();
}

HRESULT tfSystem_cameraTranslateLeft() {
    return system::cameraTranslateLeft();
}

HRESULT tfSystem_cameraTranslateForward() {
    return system::cameraTranslateForward();
}

HRESULT tfSystem_cameraTranslateBackward() {
    return system::cameraTranslateBackward();
}

HRESULT tfSystem_cameraRotateDown() {
    return system::cameraRotateDown();
}

HRESULT tfSystem_cameraRotateUp() {
    return system::cameraRotateUp();
}

HRESULT tfSystem_cameraRotateLeft() {
    return system::cameraRotateLeft();
}

HRESULT tfSystem_cameraRotateRight() {
    return system::cameraRotateRight();
}

HRESULT tfSystem_cameraRollLeft() {
    return system::cameraRollLeft();
}

HRESULT tfSystem_cameraRollRight() {
    return system::cameraRollRight();
}

HRESULT tfSystem_cameraZoomIn() {
    return system::cameraZoomIn();
}

HRESULT tfSystem_cameraZoomOut() {
    return system::cameraZoomOut();
}

HRESULT tfSystem_cameraInitMouse(int x, int y) {
    return system::cameraInitMouse({x, y});
}

HRESULT tfSystem_cameraTranslateBy(float x, float y) {
    return system::cameraTranslateBy({x, y});
}

HRESULT tfSystem_cameraZoomBy(float delta) {
    return system::cameraZoomBy(delta);
}

HRESULT tfSystem_cameraZoomTo(float distance) {
    return system::cameraZoomTo(distance);
}

HRESULT tfSystem_cameraRotateToAxis(float *axis, float distance) {
    if(!axis) 
        return E_FAIL;
    return system::cameraRotateToAxis(fVector3::from(axis), distance);
}

HRESULT tfSystem_cameraRotateToEulerAngle(float *angles) {
    if(!angles) 
        return E_FAIL;
    return system::cameraRotateToEulerAngle(fVector3::from(angles));
}

HRESULT tfSystem_cameraRotateByEulerAngle(float *angles) {
    if(!angles) 
        return E_FAIL;
    return system::cameraRotateByEulerAngle(fVector3::from(angles));
}

HRESULT tfSystem_cameraCenter(float **center) {
    if(!center) 
        return E_FAIL;
    auto _center = system::cameraCenter();
    TFC_VECTOR3_COPYFROM(_center, (*center));
    return S_OK;
}

HRESULT tfSystem_cameraRotation(float **rotation) {
    if(!rotation) 
        return E_FAIL;
    auto _rotation = system::cameraRotation();
    auto axis = _rotation.axis();
    TFC_VECTOR3_COPYFROM(axis, (*rotation));
    *rotation[3] = _rotation.angle();
    return S_OK;
}

HRESULT tfSystem_cameraZoom(float *zoom) {
    TFC_PTRCHECK(zoom);
    *zoom = system::cameraZoom();
    return S_OK;
}

HRESULT tfSystem_getRendering3DBonds(bool *flag) {
    TFC_PTRCHECK(flag);
    *flag = system::getRendering3DBonds();
    return S_OK;
}

HRESULT tfSystem_setRendering3DBonds(bool flag) {
    system::setRendering3DBonds(flag);
    return S_OK;
}

HRESULT tfSystem_toggleRendering3DBonds() {
    system::toggleRendering3DBonds();
    return S_OK;
}

HRESULT tfSystem_getRendering3DAngles(bool *flag) {
    *flag = system::getRendering3DAngles();
    return S_OK;
}

HRESULT tfSystem_setRendering3DAngles(bool flag) {
    system::setRendering3DAngles(flag);
    return S_OK;
}

HRESULT tfSystem_toggleRendering3DAngles() {
    system::toggleRendering3DAngles();
    return S_OK;
}

HRESULT tfSystem_getRendering3DDihedrals(bool *flag) {
    TFC_PTRCHECK(flag);
    *flag = system::getRendering3DDihedrals();
    return S_OK;
}

HRESULT tfSystem_setRendering3DDihedrals(bool flag) {
    system::setRendering3DDihedrals(flag);
    return S_OK;
}

HRESULT tfSystem_toggleRendering3DDihedrals() {
    system::toggleRendering3DDihedrals();
    return S_OK;
}

HRESULT tfSystem_setRendering3DAll(bool flag) {
    system::setRendering3DAll(flag);
    return S_OK;
}

HRESULT tfSystem_toggleRendering3DAll() {
    system::toggleRendering3DAll();
    return S_OK;
}

HRESULT tfSystem_getLineWidth(tfFloatP_t *lineWidth) {
    TFC_PTRCHECK(lineWidth);
    *lineWidth = system::getLineWidth();
    return S_OK;
}

HRESULT tfSystem_setLineWidth(tfFloatP_t lineWidth) {
    return system::setLineWidth(lineWidth);
}

HRESULT tfSystem_getLineWidthMin(tfFloatP_t *lineWidth) {
    TFC_PTRCHECK(lineWidth);
    *lineWidth = system::getLineWidthMin();
    return S_OK;
}

HRESULT tfSystem_getLineWidthMax(tfFloatP_t *lineWidth) {
    TFC_PTRCHECK(lineWidth);
    *lineWidth = system::getLineWidthMax();
    return S_OK;
}

HRESULT tfSystem_getAmbientColor(float **color) {
    TFC_PTRCHECK(color);
    auto _color = system::getAmbientColor();
    TFC_VECTOR3_COPYFROM(_color, (*color));
    return S_OK;
}

HRESULT tfSystem_setAmbientColor(float *color) {
    if(!color) 
        return E_FAIL;
    return system::setAmbientColor(fVector3::from(color));
}

HRESULT tfSystem_getDiffuseColor(float **color) {
    if(!color) 
        return E_FAIL;
    auto _color = system::getDiffuseColor();
    TFC_VECTOR3_COPYFROM(_color, (*color));
    return S_OK;
}

HRESULT tfSystem_setDiffuseColor(float *color) {
    if(!color) 
        return E_FAIL;
    return system::setDiffuseColor(fVector3::from(color));
}

HRESULT tfSystem_getSpecularColor(float **color) {
    if(!color) 
        return E_FAIL;
    auto _color = system::getSpecularColor();
    TFC_VECTOR3_COPYFROM(_color, (*color));
    return S_OK;
}

HRESULT tfSystem_setSpecularColor(float *color) {
    if(!color) 
        return E_FAIL;
    return system::setSpecularColor(fVector3::from(color));
}

HRESULT tfSystem_getShininess(float *shininess) {
    TFC_PTRCHECK(shininess);
    *shininess = system::getShininess();
    return S_OK;
}

HRESULT tfSystem_setShininess(float shininess) {
    return system::setShininess(shininess);
}

HRESULT tfSystem_getGridColor(float **color) {
    if(!color) 
        return E_FAIL;
    auto _color = system::getGridColor();
    TFC_VECTOR3_COPYFROM(_color, (*color));
    return S_OK;
}

HRESULT tfSystem_setGridColor(float *color) {
    if(!color) 
        return E_FAIL;
    return system::setGridColor(fVector3::from(color));
}

HRESULT tfSystem_getSceneBoxColor(float **color) {
    if(!color) 
        return E_FAIL;
    auto _color = system::getSceneBoxColor();
    TFC_VECTOR3_COPYFROM(_color, (*color));
    return S_OK;
}

HRESULT tfSystem_setSceneBoxColor(float *color) {
    if(!color) 
        return E_FAIL;
    return system::setSceneBoxColor(fVector3::from(color));
}

HRESULT tfSystem_getLightDirection(float **lightDir) {
    if(!lightDir) 
        return E_FAIL;
    auto _lightDir = system::getLightDirection();
    TFC_VECTOR3_COPYFROM(_lightDir, (*lightDir));
    return S_OK;
}

HRESULT tfSystem_setLightDirection(float *lightDir) {
    if(!lightDir) 
        return E_FAIL;
    return system::setLightDirection(fVector3::from(lightDir));
}

HRESULT tfSystem_getLightColor(float **color) {
    if(!color) 
        return E_FAIL;
    auto _color = system::getLightColor();
    TFC_VECTOR3_COPYFROM(_color, (*color));
    return S_OK;
}

HRESULT tfSystem_setLightColor(float *color) {
    if(!color) 
        return E_FAIL;
    return system::setLightColor(fVector3::from(color));
}

HRESULT tfSystem_getBackgroundColor(float **color) {
    if(!color) 
        return E_FAIL;
    auto _color = system::getBackgroundColor();
    TFC_VECTOR3_COPYFROM(_color, (*color));
    return S_OK;
}

HRESULT tfSystem_setBackgroundColor(float *color) {
    if(!color) 
        return E_FAIL;
    return system::setBackgroundColor(fVector3::from(color));
}

HRESULT tfSystem_decorated(bool *decorated) {
    TFC_PTRCHECK(decorated);
    *decorated = system::decorated();
    return S_OK;
}

HRESULT tfSystem_decorateScene(bool decorate) {
    return system::decorateScene(decorate);
}

HRESULT tfSystem_showingDiscretization(bool *showing) {
    TFC_PTRCHECK(showing);
    *showing = system::showingDiscretization();
    return S_OK;
}

HRESULT tfSystem_showDiscretization(bool show) {
    return system::showDiscretization(show);
}

HRESULT tfSystem_getDiscretizationColor(float **color) {
    if(!color) 
        return E_FAIL;
    auto _color = system::getDiscretizationColor();
    TFC_VECTOR3_COPYFROM(_color, (*color));
    return S_OK;
}

HRESULT tfSystem_setDiscretizationColor(float *color) {
    if(!color) 
        return E_FAIL;
    return system::setDiscretizationColor(fVector3::from(color));
}

HRESULT tfSystem_viewReshape(int sizex, int sizey) {
    return system::viewReshape({sizex, sizey});
}

HRESULT tfSystem_getCPUInfo(char ***names, bool **flags, unsigned int *numNames) {
    TFC_PTRCHECK(names);
    TFC_PTRCHECK(flags);
    TFC_PTRCHECK(numNames);
    auto cpu_info = system::cpu_info();
    *numNames = cpu_info.size();
    if(*numNames > 0) {
        char **_names = (char**)malloc(*numNames * sizeof(char*));
        bool *_flags = (bool*)malloc(*numNames * sizeof(bool));
        if(!_names || !_flags) 
            return E_OUTOFMEMORY;
        unsigned int i = 0;
        for(auto &itr : cpu_info) {
            char *_c = new char[itr.first.size() + 1];
            std::strcpy(_c, itr.first.c_str());
            _names[i] = _c;
            _flags[i] = itr.second;
            i++;
        }
        *names = _names;
        *flags = _flags;
    }
    return S_OK;
}

HRESULT tfSystem_getCompileFlags(char ***flags, unsigned int *numFlags) {
    TFC_PTRCHECK(flags);
    TFC_PTRCHECK(numFlags);
    auto compile_flags = system::compile_flags();
    *numFlags = compile_flags.size();
    if(*numFlags > 0) {
        char **_flags = (char**)malloc(*numFlags * sizeof(char*));
        if(!_flags) 
            return E_OUTOFMEMORY;
        unsigned int i = 0;
        for(std::string &_s : compile_flags) {
            char *_c = new char[_s.size() + 1];
            std::strcpy(_c, _s.c_str());
            _flags[i] = _c;
            i++;
        }
        *flags = _flags;
    }
    return S_OK;
}

HRESULT tfSystem_getGLInfo(char ***names, char ***values, unsigned int *numNames) {
    TFC_PTRCHECK(names);
    TFC_PTRCHECK(values);
    TFC_PTRCHECK(numNames);
    auto gl_info = system::gl_info();
    *numNames = gl_info.size();
    if(*numNames > 0) {
        char **_names = (char**)malloc(*numNames * sizeof(char*));
        char **_values = (char**)malloc(*numNames * sizeof(char*));
        if(!_names || !_values) 
            return E_OUTOFMEMORY;
        unsigned int i = 0;
        for(auto &itr : gl_info) {
            char *_sn = new char[itr.first.size() + 1];
            char *_sv = new char[itr.second.size() + 1];
            std::strcpy(_sn, itr.first.c_str());
            std::strcpy(_sv, itr.second.c_str());
            i++;
        }
        *names = _names;
        *values = _values;
    }
    return S_OK;
}

HRESULT tfSystem_getEGLInfo(char **info, unsigned int *numChars) {
    TFC_PTRCHECK(info);
    TFC_PTRCHECK(numChars);
    return TissueForge::capi::str2Char(system::egl_info(), info, numChars);
}
