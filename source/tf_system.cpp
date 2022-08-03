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

#include "tf_system.h"
#include "tfSimulator.h"
#include "rendering/tfWindowlessApplication.h"
#include "rendering/tfWindowless.h"
#include "rendering/tfApplication.h"
#include "rendering/tfGlfwApplication.h"
#include "rendering/tfClipPlane.h"
#include "tfLogger.h"
#include "tfError.h"
#include "tf_test.h"
#include <sstream>


using namespace TissueForge;


static double ms(ticks tks)
{
    return (double)tks / (_Engine.time * CLOCKS_PER_SEC);
}

std::tuple<char*, size_t> system::imageData() {
    return rendering::framebufferImageData();
}

HRESULT system::screenshot(const std::string &filePath) {

    try {
        return rendering::screenshot(filePath);
        
    }
    catch(const std::exception &e) {
        TF_RETURN_EXP(e);
    }
}

HRESULT system::screenshot(const std::string &filePath, const bool &decorate, const FVector3 &bgcolor) {

    try {
        Simulator *sim = Simulator::get();
        
        rendering::UniverseRenderer *renderer = sim->app->getRenderer();
        
        bool _decorate = renderer->sceneDecorated();
        FVector3 _bgcolor = renderer->backgroundColor();

        renderer->decorateScene(decorate);
        renderer->setBackgroundColor(fVector3(bgcolor));

        HRESULT result = screenshot(filePath);

        renderer->decorateScene(_decorate);
        renderer->setBackgroundColor(fVector3(_bgcolor));
    
        return result;
        
    }
    catch(const std::exception &e) {
        TF_RETURN_EXP(e);
    }
}

bool system::contextHasCurrent() {
    try {
        std::thread::id id = std::this_thread::get_id();
        TF_Log(LOG_INFORMATION)  << ", thread id: " << id ;
        
        Simulator *sim = Simulator::get();
        
        return sim->app->contextHasCurrent();
        
    }
    catch(const std::exception &e) {
        TF_RETURN_EXP(e);
    }
}

HRESULT system::contextMakeCurrent() {
    try {
        std::thread::id id = std::this_thread::get_id();
        TF_Log(LOG_INFORMATION)  << ", thread id: " << id ;
        
        Simulator *sim = Simulator::get();
        sim->app->contextMakeCurrent();

        return S_OK;
    }
    catch(const std::exception &e) {
        tf_exp(e);
        return E_FAIL;
    }
}

HRESULT system::contextRelease() {
    try {
        std::thread::id id = std::this_thread::get_id();
        TF_Log(LOG_INFORMATION)  << ", thread id: " << id ;
        
        Simulator *sim = Simulator::get();
        sim->app->contextRelease();
        
        return S_OK;
    }
    catch(const std::exception &e) {
        tf_exp(e);
        return E_FAIL;
    }
}

HRESULT system::cameraMoveTo(const FVector3 &eye, const FVector3 &center, const FVector3 &up) {
    try {
        Simulator *sim = Simulator::get();
        
        rendering::UniverseRenderer *renderer = sim->app->getRenderer();
        
        rendering::ArcBall *ab = renderer->_arcball;
        
        ab->setViewParameters(fVector3(eye), fVector3(center), fVector3(up));

        return S_OK;
    }
    catch(const std::exception &e) {
        tf_exp(e);
        return E_FAIL;
    }
}

HRESULT system::cameraMoveTo(const FVector3 &center, const FQuaternion &rotation, const float &zoom) {
    try {
        Simulator *sim = Simulator::get();
        
        rendering::UniverseRenderer *renderer = sim->app->getRenderer();
        
        rendering::ArcBallCamera *ab = renderer->_arcball;
        
        ab->setViewParameters(center, rotation, zoom);

        return S_OK;
    }
    catch(const std::exception &e) {
        tf_exp(e);
        return E_FAIL;
    }
}

HRESULT system::cameraViewBottom() {
    try {
        Simulator *sim = Simulator::get();
        
        rendering::UniverseRenderer *renderer = sim->app->getRenderer();
        
        rendering::ArcBallCamera *ab = renderer->_arcball;
        
        ab->viewBottom(2.0 * renderer->sideLength);
        ab->translateToOrigin();

        return S_OK;
    }
    catch(const std::exception &e) {
        tf_exp(e);
        return E_FAIL;
    }
}

HRESULT system::cameraViewTop() {
    try {
        Simulator *sim = Simulator::get();
        
        rendering::UniverseRenderer *renderer = sim->app->getRenderer();
        
        rendering::ArcBallCamera *ab = renderer->_arcball;
        
        ab->viewTop(2.0 * renderer->sideLength);
        ab->translateToOrigin();

        return S_OK;
    }
    catch(const std::exception &e) {
        tf_exp(e);
        return E_FAIL;
    }
}

HRESULT system::cameraViewLeft() {
    try {
        Simulator *sim = Simulator::get();
        
        rendering::UniverseRenderer *renderer = sim->app->getRenderer();
        
        rendering::ArcBallCamera *ab = renderer->_arcball;
        
        ab->viewLeft(2.0 * renderer->sideLength);
        ab->translateToOrigin();

        return S_OK;
    }
    catch(const std::exception &e) {
        tf_exp(e);
        return E_FAIL;
    }
}

HRESULT system::cameraViewRight() {
    try {
        Simulator *sim = Simulator::get();
        
        rendering::UniverseRenderer *renderer = sim->app->getRenderer();
        
        rendering::ArcBallCamera *ab = renderer->_arcball;
        
        ab->viewRight(2.0 * renderer->sideLength);
        ab->translateToOrigin();

        return S_OK;
    }
    catch(const std::exception &e) {
        tf_exp(e);
        return E_FAIL;
    }
}

HRESULT system::cameraViewBack() {
    try {
        Simulator *sim = Simulator::get();
        
        rendering::UniverseRenderer *renderer = sim->app->getRenderer();
        
        rendering::ArcBallCamera *ab = renderer->_arcball;
        
        ab->viewBack(2.0 * renderer->sideLength);
        ab->translateToOrigin();

        return S_OK;
    }
    catch(const std::exception &e) {
        tf_exp(e);
        return E_FAIL;
    }
}

HRESULT system::cameraViewFront() {
    try {
        Simulator *sim = Simulator::get();
        
        rendering::UniverseRenderer *renderer = sim->app->getRenderer();
        
        rendering::ArcBallCamera *ab = renderer->_arcball;
        
        ab->viewFront(2.0 * renderer->sideLength);
        ab->translateToOrigin();

        return S_OK;
    }
    catch(const std::exception &e) {
        tf_exp(e);
        return E_FAIL;
    }
}

HRESULT system::cameraReset() {
    try {
        Simulator *sim = Simulator::get();
        
        rendering::UniverseRenderer *renderer = sim->app->getRenderer();
        
        rendering::ArcBall *ab = renderer->_arcball;
            
        ab->reset();

        return S_OK;
    }
    catch(const std::exception &e) {
        tf_exp(e);
        return E_FAIL;
    }
}

HRESULT system::cameraRotateMouse(const iVector2 &mousePos) {
    try {
        TF_Log(LOG_TRACE);
        
        Simulator *sim = Simulator::get();
        
        rendering::UniverseRenderer *renderer = sim->app->getRenderer();
        
        rendering::ArcBall *ab = renderer->_arcball;
        
        ab->rotate(mousePos);
        
        ab->updateTransformation();

        return S_OK;
    }
    catch(const std::exception &e) {
        tf_exp(e);
        return E_FAIL;
    }
}

HRESULT system::cameraTranslateMouse(const iVector2 &mousePos) {
    try {
        TF_Log(LOG_TRACE);
        
        Simulator *sim = Simulator::get();
        
        rendering::UniverseRenderer *renderer = sim->app->getRenderer();
        
        rendering::ArcBall *ab = renderer->_arcball;
        
        ab->translate(mousePos);
        
        ab->updateTransformation();
        
        return S_OK;
    }
    catch(const std::exception &e) {
        tf_exp(e);
        return E_FAIL;
    }
}

HRESULT system::cameraTranslateDown() {
    try {
        TF_Log(LOG_TRACE);
        
        Simulator *sim = Simulator::get();
        
        rendering::UniverseRenderer *renderer = sim->app->getRenderer();
        
        renderer->cameraTranslateDown();
        
        return S_OK;
    }
    catch(const std::exception &e) {
        tf_exp(e);
        return E_FAIL;
    }
}

HRESULT system::cameraTranslateUp() {
    try {
        TF_Log(LOG_TRACE);
        
        Simulator *sim = Simulator::get();
        
        rendering::UniverseRenderer *renderer = sim->app->getRenderer();
        
        renderer->cameraTranslateUp();
        
        return S_OK;
    }
    catch(const std::exception &e) {
        tf_exp(e);
        return E_FAIL;
    }
}

HRESULT system::cameraTranslateRight() {
    try {
        TF_Log(LOG_TRACE);
        
        Simulator *sim = Simulator::get();
        
        rendering::UniverseRenderer *renderer = sim->app->getRenderer();
        
        renderer->cameraTranslateRight();
        
        return S_OK;
    }
    catch(const std::exception &e) {
        tf_exp(e);
        return E_FAIL;
    }
}

HRESULT system::cameraTranslateLeft() {
    try {
        TF_Log(LOG_TRACE);
        
        Simulator *sim = Simulator::get();
        
        rendering::UniverseRenderer *renderer = sim->app->getRenderer();
        
        renderer->cameraTranslateLeft();
        
        return S_OK;
    }
    catch(const std::exception &e) {
        tf_exp(e);
        return E_FAIL;
    }
}

HRESULT system::cameraTranslateForward() {
    try {
        TF_Log(LOG_TRACE);
        
        Simulator *sim = Simulator::get();
        
        rendering::UniverseRenderer *renderer = sim->app->getRenderer();
        
        renderer->cameraTranslateForward();
        
        return S_OK;
    }
    catch(const std::exception &e) {
        tf_exp(e);
        return E_FAIL;
    }
}

HRESULT system::cameraTranslateBackward() {
    try {
        TF_Log(LOG_TRACE);
        
        Simulator *sim = Simulator::get();
        
        rendering::UniverseRenderer *renderer = sim->app->getRenderer();
        
        renderer->cameraTranslateBackward();
        
        return S_OK;
    }
    catch(const std::exception &e) {
        tf_exp(e);
        return E_FAIL;
    }
}

HRESULT system::cameraRotateDown() {
    try {
        TF_Log(LOG_TRACE);
        
        Simulator *sim = Simulator::get();
        
        rendering::UniverseRenderer *renderer = sim->app->getRenderer();
        
        renderer->cameraRotateDown();
        
        return S_OK;
    }
    catch(const std::exception &e) {
        tf_exp(e);
        return E_FAIL;
    }
}

HRESULT system::cameraRotateUp() {
    try {
        TF_Log(LOG_TRACE);
        
        Simulator *sim = Simulator::get();
        
        rendering::UniverseRenderer *renderer = sim->app->getRenderer();
        
        renderer->cameraRotateUp();
        
        return S_OK;
    }
    catch(const std::exception &e) {
        tf_exp(e);
        return E_FAIL;
    }
}

HRESULT system::cameraRotateLeft() {
    try {
        TF_Log(LOG_TRACE);
        
        Simulator *sim = Simulator::get();
        
        rendering::UniverseRenderer *renderer = sim->app->getRenderer();
        
        renderer->cameraRotateLeft();
        
        return S_OK;
    }
    catch(const std::exception &e) {
        tf_exp(e);
        return E_FAIL;
    }
}

HRESULT system::cameraRotateRight() {
    try {
        TF_Log(LOG_TRACE);
        
        Simulator *sim = Simulator::get();
        
        rendering::UniverseRenderer *renderer = sim->app->getRenderer();
        
        renderer->cameraRotateRight();
        
        return S_OK;
    }
    catch(const std::exception &e) {
        tf_exp(e);
        return E_FAIL;
    }
}

HRESULT system::cameraRollLeft() {
    try {
        TF_Log(LOG_TRACE);
        
        Simulator *sim = Simulator::get();
        
        rendering::UniverseRenderer *renderer = sim->app->getRenderer();
        
        renderer->cameraRollLeft();
        
        return S_OK;
    }
    catch(const std::exception &e) {
        tf_exp(e);
        return E_FAIL;
    }
}

HRESULT system::cameraRollRight() {
    try {
        TF_Log(LOG_TRACE);
        
        Simulator *sim = Simulator::get();
        
        rendering::UniverseRenderer *renderer = sim->app->getRenderer();
        
        renderer->cameraRollRight();
        
        return S_OK;
    }
    catch(const std::exception &e) {
        tf_exp(e);
        return E_FAIL;
    }
}

HRESULT system::cameraZoomIn() {
    try {
        TF_Log(LOG_TRACE);
        
        Simulator *sim = Simulator::get();
        
        rendering::UniverseRenderer *renderer = sim->app->getRenderer();
        
        renderer->cameraZoomIn();
        
        return S_OK;
    }
    catch(const std::exception &e) {
        tf_exp(e);
        return E_FAIL;
    }
}

HRESULT system::cameraZoomOut() {
    try {
        TF_Log(LOG_TRACE);
        
        Simulator *sim = Simulator::get();
        
        rendering::UniverseRenderer *renderer = sim->app->getRenderer();
        
        renderer->cameraZoomOut();
        
        return S_OK;
    }
    catch(const std::exception &e) {
        tf_exp(e);
        return E_FAIL;
    }
}

HRESULT system::cameraInitMouse(const iVector2 &mousePos) {
    try {
        Simulator *sim = Simulator::get();
        
        rendering::UniverseRenderer *renderer = sim->app->getRenderer();
        
        rendering::ArcBall *ab = renderer->_arcball;
        
        ab->initTransformation(mousePos);
        
        Simulator::get()->redraw();

        return S_OK;
    }
    catch(const std::exception &e) {
        tf_exp(e);
        return E_FAIL;
    }
}

HRESULT system::cameraTranslateBy(const FVector2 &trans) {
    try {
        Simulator *sim = Simulator::get();
        
        rendering::UniverseRenderer *renderer = sim->app->getRenderer();
        
        rendering::ArcBall *ab = renderer->_arcball;
        
        ab->translateDelta(fVector2(trans));

        return S_OK;
    }
    catch(const std::exception &e) {
        tf_exp(e);
        return E_FAIL;
    }
}

HRESULT system::cameraZoomBy(const float &delta) {
    try {
        Simulator *sim = Simulator::get();
        
        rendering::UniverseRenderer *renderer = sim->app->getRenderer();
        
        rendering::ArcBall *ab = renderer->_arcball;
        
        ab->zoom(delta);
        
        ab->updateTransformation();

        return S_OK;
    }
    catch(const std::exception &e) {
        tf_exp(e);
        return E_FAIL;
    }
}

HRESULT system::cameraZoomTo(const float &distance) {
    try {
        Simulator *sim = Simulator::get();
        
        rendering::UniverseRenderer *renderer = sim->app->getRenderer();
        
        rendering::ArcBall *ab = renderer->_arcball;
        
        ab->zoomTo(distance);

        return S_OK;
    }
    catch(const std::exception &e) {
        tf_exp(e);
        return E_FAIL;
    }
}

HRESULT system::cameraRotateToAxis(const FVector3 &axis, const float &distance) {
    try {
        Simulator *sim = Simulator::get();
        
        rendering::UniverseRenderer *renderer = sim->app->getRenderer();
        
        rendering::ArcBall *ab = renderer->_arcball;
        
        ab->rotateToAxis(fVector3(axis), distance);
        
        return S_OK;
    }
    catch(const std::exception &e) {
        tf_exp(e);
        return E_FAIL;
    }
}

HRESULT system::cameraRotateToEulerAngle(const FVector3 &angles) {
    try {
        Simulator *sim = Simulator::get();
        
        rendering::UniverseRenderer *renderer = sim->app->getRenderer();
        
        rendering::ArcBall *ab = renderer->_arcball;
        
        ab->rotateToEulerAngles(fVector3(angles));
        
        return S_OK;
    }
    catch(const std::exception &e) {
        tf_exp(e);
        return E_FAIL;
    }
}

HRESULT system::cameraRotateByEulerAngle(const FVector3 &angles) {
    try {
        Simulator *sim = Simulator::get();
        
        rendering::UniverseRenderer *renderer = sim->app->getRenderer();
        
        rendering::ArcBall *ab = renderer->_arcball;
        
        ab->rotateByEulerAngles(fVector3(angles));
        
        Simulator::get()->redraw();

        return S_OK;
    }
    catch(const std::exception &e) {
        tf_exp(e);
        return E_FAIL;
    }
}

FVector3 system::cameraCenter() {
    try {
        Simulator *sim = Simulator::get();
        
        rendering::UniverseRenderer *renderer = sim->app->getRenderer();
        
        rendering::ArcBallCamera *ab = renderer->_arcball;
        
        return ab->cposition();
    }
    catch(const std::exception &e) {
        tf_exp(e);
        return FVector3();
    }
}

FQuaternion system::cameraRotation() {
    try {
        Simulator *sim = Simulator::get();
        
        rendering::UniverseRenderer *renderer = sim->app->getRenderer();
        
        rendering::ArcBallCamera *ab = renderer->_arcball;
        
        return FQuaternion(ab->crotation());
    }
    catch(const std::exception &e) {
        TF_RETURN_EXP(e);
    }
}

float system::cameraZoom() {
    try {
        Simulator *sim = Simulator::get();
        
        rendering::UniverseRenderer *renderer = sim->app->getRenderer();
        
        rendering::ArcBallCamera *ab = renderer->_arcball;
        
        return ab->czoom();
    }
    catch(const std::exception &e) {
        TF_RETURN_EXP(e);
    }
}

struct rendering::UniverseRenderer* system::getRenderer() {
    try {
        Simulator *sim = Simulator::get();
        
        return sim->app->getRenderer();
    }
    catch(const std::exception &e) {
        TF_RETURN_EXP(e);
    }
}

FVector3 system::getAmbientColor() {
    try {
        return system::getRenderer()->ambientColor();
    }
    catch(const std::exception &e) {
        tf_exp(e);
        return FVector3();
    }
}

HRESULT system::setAmbientColor(const FVector3 &color) {
    try {
        system::getRenderer()->setAmbientColor(fVector3(color));
        
        return S_OK;
    }
    catch(const std::exception &e) {
        tf_exp(e);
        return E_FAIL;
    }
}

HRESULT system::setAmbientColor(const FVector3 &color, const unsigned int &srFlag) {
    try {
        system::getRenderer()->getSubRenderer((rendering::SubRendererFlag)srFlag)->setAmbientColor(fVector3(color));
        
        return S_OK;
    }
    catch(const std::exception &e) {
        tf_exp(e);
        return E_FAIL;
    }
}

FVector3 system::getDiffuseColor() {
    try {
        return system::getRenderer()->diffuseColor();
    }
    catch(const std::exception &e) {
        tf_exp(e);
        return FVector3();
    }
}

HRESULT system::setDiffuseColor(const FVector3 &color) {
    try {
        system::getRenderer()->setDiffuseColor(fVector3(color));
        
        return S_OK;
    }
    catch(const std::exception &e) {
        tf_exp(e);
        return E_FAIL;
    }
}

HRESULT system::setDiffuseColor(const FVector3 &color, const unsigned int &srFlag) {
    try {
        system::getRenderer()->getSubRenderer((rendering::SubRendererFlag)srFlag)->setDiffuseColor(fVector3(color));
        
        return S_OK;
    }
    catch(const std::exception &e) {
        tf_exp(e);
        return E_FAIL;
    }
}

FVector3 system::getSpecularColor() {
    try {
        return system::getRenderer()->specularColor();
    }
    catch(const std::exception &e) {
        tf_exp(e);
        return FVector3();
    }
}

HRESULT system::setSpecularColor(const FVector3 &color) {
    try {
        system::getRenderer()->setSpecularColor(fVector3(color));
        
        return S_OK;
    }
    catch(const std::exception &e) {
        tf_exp(e);
        return E_FAIL;
    }
}

HRESULT system::setSpecularColor(const FVector3 &color, const unsigned int &srFlag) {
    try {
        system::getRenderer()->getSubRenderer((rendering::SubRendererFlag)srFlag)->setSpecularColor(fVector3(color));
        
        return S_OK;
    }
    catch(const std::exception &e) {
        tf_exp(e);
        return E_FAIL;
    }
}

float system::getShininess() {
    try {
        return system::getRenderer()->shininess();
    }
    catch(const std::exception &e) {
        TF_RETURN_EXP(e);
    }
}

HRESULT system::setShininess(const float &shininess) {
    try {
        system::getRenderer()->setShininess(shininess);
        
        return S_OK;
    }
    catch(const std::exception &e) {
        tf_exp(e);
        return E_FAIL;
    }
}

HRESULT system::setShininess(const float &shininess, const unsigned int &srFlag) {
    try {
        system::getRenderer()->getSubRenderer((rendering::SubRendererFlag)srFlag)->setShininess(shininess);
        
        return S_OK;
    }
    catch(const std::exception &e) {
        tf_exp(e);
        return E_FAIL;
    }
}

FVector3 system::getGridColor() {
    try {
        return system::getRenderer()->gridColor();
    }
    catch(const std::exception &e) {
        tf_exp(e);
        return FVector3();
    }
}

HRESULT system::setGridColor(const FVector3 &color) {
    try {
        system::getRenderer()->setGridColor(fVector3(color));
        
        return S_OK;
    }
    catch(const std::exception &e) {
        tf_exp(e);
        return E_FAIL;
    }
}

FVector3 system::getSceneBoxColor() {
    try {
        return system::getRenderer()->sceneBoxColor();
    }
    catch(const std::exception &e) {
        tf_exp(e);
        return FVector3();
    }
}

HRESULT system::setSceneBoxColor(const FVector3 &color) {
    try {
        system::getRenderer()->setSceneBoxColor(fVector3(color));
        
        return S_OK;
    }
    catch(const std::exception &e) {
        tf_exp(e);
        return E_FAIL;
    }
}

FVector3 system::getLightDirection() {
    try {
        return system::getRenderer()->lightDirection();
    }
    catch(const std::exception &e) {
        tf_exp(e);
        return FVector3();
    }
}

HRESULT system::setLightDirection(const FVector3& lightDir) {
    try {
        system::getRenderer()->setLightDirection(lightDir);
        
        return S_OK;
    }
    catch(const std::exception &e) {
        tf_exp(e);
        return E_FAIL;
    }
}

HRESULT system::setLightDirection(const FVector3& lightDir, const unsigned int &srFlag) {
    try {
        system::getRenderer()->getSubRenderer((rendering::SubRendererFlag)srFlag)->setLightDirection(lightDir);
        
        return S_OK;
    }
    catch(const std::exception &e) {
        tf_exp(e);
        return E_FAIL;
    }
}

FVector3 system::getLightColor() {
    try {
        return FVector3(system::getRenderer()->lightColor());
    }
    catch(const std::exception &e) {
        tf_exp(e);
        return FVector3();
    }
}

HRESULT system::setLightColor(const FVector3 &color) {
    try {
        system::getRenderer()->setLightColor(fVector3(color));
        
        return S_OK;
    }
    catch(const std::exception &e) {
        tf_exp(e);
        return E_FAIL;
    }
}

HRESULT system::setLightColor(const FVector3 &color, const unsigned int &srFlag) {
    try {
        system::getRenderer()->getSubRenderer((rendering::SubRendererFlag)srFlag)->setLightColor(fVector3(color));
        
        return S_OK;
    }
    catch(const std::exception &e) {
        tf_exp(e);
        return E_FAIL;
    }
}

FVector3 system::getBackgroundColor() {
    try {
        return system::getRenderer()->backgroundColor();
    }
    catch(const std::exception &e) {
        tf_exp(e);
        return FVector3();
    }
}

HRESULT system::setBackgroundColor(const FVector3 &color) {
    try {
        system::getRenderer()->setBackgroundColor(fVector3(color));
        
        return S_OK;
    }
    catch(const std::exception &e) {
        tf_exp(e);
        return E_FAIL;
    }
}

bool system::decorated() {
    return system::getRenderer()->sceneDecorated();
}

HRESULT system::decorateScene(const bool &decorate) {
    try {
        system::getRenderer()->decorateScene(decorate);
        
        return S_OK;
    }
    catch(const std::exception &e) {
        tf_exp(e);
        return E_FAIL;
    }
}

bool system::showingDiscretization() {
    return system::getRenderer()->showingDiscretizationGrid();
}

HRESULT system::showDiscretization(const bool &show) {
    try {
        system::getRenderer()->showDiscretizationGrid(show);
        
        return S_OK;
    }
    catch(const std::exception &e) {
        tf_exp(e);
        return E_FAIL;
    }
}

FVector3 system::getDiscretizationColor() {
    try {
        return system::getRenderer()->discretizationGridColor();
    }
    catch(const std::exception &e) {
        tf_exp(e);
        return FVector3();
    }
}

HRESULT system::setDiscretizationColor(const FVector3 &color) {
    try {
        system::getRenderer()->setDiscretizationGridColor(fVector3(color));

        return S_OK;
    }
    catch(const std::exception &e) {
        tf_exp(e);
        return E_FAIL;
    }
}

HRESULT system::viewReshape(const iVector2 &windowSize) {
    try {
        Simulator *sim = Simulator::get();

        rendering::UniverseRenderer *renderer = sim->app->getRenderer();
        
        rendering::ArcBall *ab = renderer->_arcball;
        
        ab->reshape(windowSize);
        
        return S_OK;
    }
    catch(const std::exception &e) {
        tf_exp(e);
        return E_FAIL;
    }
}

std::string system::performanceCounters() {
    std::stringstream ss;
    
    ss << "performance_timers : { " << std::endl;
    ss << "\t name: " << Universe::getName() << "," << std::endl;
    ss << "\t wall_time: " << util::wallTime() << "," << std::endl;
    ss << "\t cpu_time: " << util::CPUTime() << "," << std::endl;
    ss << "\t fps: " << engine_steps_per_second() << "," << std::endl;
    ss << "\t kinetic energy: " << engine_kinetic_energy(&_Engine) << "," << std::endl;
    ss << "\t kinetic: " << ms(_Engine.timers[engine_timer_kinetic]) << "," << std::endl;
    ss << "\t prepare: " << ms(_Engine.timers[engine_timer_prepare]) << "," << std::endl;
    ss << "\t verlet: " << ms(_Engine.timers[engine_timer_verlet]) << "," << std::endl;
    ss << "\t shuffle: " << ms(_Engine.timers[engine_timer_shuffle]) << "," << std::endl;
    ss << "\t step: " << ms(_Engine.timers[engine_timer_step]) << "," << std::endl;
    ss << "\t nonbond: " << ms(_Engine.timers[engine_timer_nonbond]) << "," << std::endl;
    ss << "\t bonded: " << ms(_Engine.timers[engine_timer_bonded]) << "," << std::endl;
    ss << "\t advance: " << ms(_Engine.timers[engine_timer_advance]) << "," << std::endl;
    ss << "\t rendering: " << ms(_Engine.timers[engine_timer_render]) << "," << std::endl;
    ss << "\t total: " << ms(_Engine.timers[engine_timer_render] + _Engine.timers[engine_timer_step]) << "," << std::endl;
    #ifdef HAVE_CUDA
    if(_Engine.flags & engine_flag_cuda) {
        ss << "\t cuda load: " << ms(_Engine.timers[engine_timer_cuda_load]) << "," << std::endl;
        ss << "\t cuda pairs: " << ms(_Engine.timers[engine_timer_cuda_dopairs]) << "," << std::endl;
        ss << "\t cuda unload: " << ms(_Engine.timers[engine_timer_cuda_unload]) << "," << std::endl;
    }
    #endif
    ss << "\t time_steps: " << _Engine.time  << std::endl;
    ss << "}" << std::endl;
    
    return ss.str();
}

std::unordered_map<std::string, bool> system::cpu_info() {
    return util::getFeaturesMap();
}

std::list<std::string> system::compile_flags() {
    return util::CompileFlags().getFlags();
}

std::unordered_map<std::string, std::string> system::gl_info() {
    return rendering::glInfo();
}

std::string system::egl_info() {
    return rendering::eglInfo();
}

test::testHeadless_t test::testHeadless() {
    return system::gl_info();
}

void system::printPerformanceCounters() {
    LoggingBuffer log(LOG_NOTICE, NULL, NULL, -1);
    log.stream() << system::performanceCounters();
}

static Magnum::Debug *magnum_debug = NULL;
static Magnum::Warning *magnum_warning = NULL;
static Magnum::Error *magnum_error = NULL;

HRESULT system::LoggerCallbackImpl(LogEvent, std::ostream *os) {
    TF_Log(LOG_TRACE);
    
    delete magnum_debug; magnum_debug = NULL;
    delete magnum_warning; magnum_warning = NULL;
    delete magnum_error; magnum_error = NULL;
    
    if(Logger::getLevel() >= LOG_ERROR) {
        TF_Log(LOG_DEBUG) << "setting Magnum::Error to Tissue Forge log output";
        magnum_error = new Magnum::Error(os);
    }
    else {
        magnum_error = new Magnum::Error(NULL);
    }
    
    if(Logger::getLevel() >= LOG_WARNING) {
        TF_Log(LOG_DEBUG) << "setting Magnum::Warning to Tissue Forge log output";
        magnum_warning = new Magnum::Warning(os);
    }
    else {
        magnum_warning = new Magnum::Warning(NULL);
    }
    
    if(Logger::getLevel() >= LOG_DEBUG) {
        TF_Log(LOG_DEBUG) << "setting Magnum::Debug to Tissue Forge log output";
        magnum_debug = new Magnum::Debug(os);
    }
    else {
        magnum_debug = new Magnum::Debug(NULL);
    }
    
    return S_OK;
}

std::vector<std::string> system::colorMapperNames() {
    return rendering::ColorMapper::getNames();
}


#define TF_SYSTEM_GETARROWRENDERER(renderer, __VA_ARGS__...) rendering::ArrowRenderer *renderer = rendering::ArrowRenderer::get(); if(!renderer) return __VA_ARGS__;


int system::addRenderArrow(rendering::ArrowData *arrow) {
    TF_SYSTEM_GETARROWRENDERER(renderer, -1);
    return renderer->addArrow(arrow);
}

std::pair<int, rendering::ArrowData*> system::addRenderArrow(
    const FVector3 &position, 
    const FVector3 &components, 
    const rendering::Style &style, 
    const float &scale) 
{
    TF_SYSTEM_GETARROWRENDERER(renderer, {-1, 0});
    return renderer->addArrow(position, components, style, scale);
}

HRESULT system::removeRenderArrow(const int &arrowId) {
    TF_SYSTEM_GETARROWRENDERER(renderer, E_FAIL);
    return renderer->removeArrow(arrowId);
}

rendering::ArrowData *system::getRenderArrow(const int &arrowId) {
    TF_SYSTEM_GETARROWRENDERER(renderer, 0);
    return renderer->getArrow(arrowId);
}
