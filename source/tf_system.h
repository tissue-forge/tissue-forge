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
 * @file tf_system.h
 * 
 */

#ifndef _SOURCE_TF_SYSTEM_H_
#define _SOURCE_TF_SYSTEM_H_

#include "TissueForge_private.h"
#include "tf_util.h"
#include "tfLogger.h"
#include "rendering/tfGlInfo.h"
#include "rendering/tfEglInfo.h"
#include "rendering/tfUniverseRenderer.h"
#include "rendering/tfArrowRenderer.h"


namespace TissueForge::system {

    using CallbackVoidOutput = void(*)();
    using CallbackInputInt = void(*)(const int&);
    using CallbackInputFloat = void(*)(const float&);
    using CallbackInputDouble = void(*)(const double&);
    using CallbackInputString = void(*)(const std::string&);

    using CallbackVoidOutput = void(*)();
    using CallbackInputInt = void(*)(const int&);
    using CallbackInputFloat = void(*)(const float&);
    using CallbackInputDouble = void(*)(const double&);
    using CallbackInputString = void(*)(const std::string&);


    CPPAPI_FUNC(void) printPerformanceCounters();

    HRESULT LoggerCallbackImpl(LogEvent, std::ostream *);

    CPPAPI_FUNC(std::tuple<char*, size_t>) imageData();

    /**
     * @brief Save a screenshot of the current scene. 
     * 
     * File formats currently supported are 
     * 
     * - Windows Bitmap (.bmp)
     * - Radiance HDR (.hdr)
     * - JPEG (.jpe, .jpg, .jpeg)
     * - PNG (.png)
     * - Truevision TGA (.tga)
     * 
     * @param filePath path of file to save
     * @return HRESULT 
     */
    CPPAPI_FUNC(HRESULT) screenshot(const std::string &filePath);

    /**
     * @brief Save a screenshot of the current scene. 
     * 
     * File formats currently supported are 
     * 
     * - Windows Bitmap (.bmp)
     * - Radiance HDR (.hdr)
     * - JPEG (.jpe, .jpg, .jpeg)
     * - PNG (.png)
     * - Truevision TGA (.tga)
     * 
     * @param filePath path of file to save
     * @param decorate flag to decorate the scene in the screenshot
     * @param bgcolor background color of the scene in the screenshot
     * @return HRESULT 
     */
    CPPAPI_FUNC(HRESULT) screenshot(const std::string &filePath, const bool &decorate, const FVector3 &bgcolor);

    CPPAPI_FUNC(bool) contextHasCurrent();
    CPPAPI_FUNC(HRESULT) contextMakeCurrent();
    CPPAPI_FUNC(HRESULT) contextRelease();

    /** Test whether the camera is lagging */
    CPPAPI_FUNC(bool) cameraIsLagging();

    /** Enable camera lagging */
    CPPAPI_FUNC(HRESULT) cameraEnableLagging();

    /** Disable camera lagging */
    CPPAPI_FUNC(HRESULT) cameraDisableLagging();

    /** Toggle camera lagging */
    CPPAPI_FUNC(HRESULT) cameraToggleLagging();

    /** Get the camera lagging */
    CPPAPI_FUNC(float) cameraGetLagging();

    /** Set the camera lagging. Value must be in [0, 1) */
    CPPAPI_FUNC(HRESULT) cameraSetLagging(const float &lagging);

    /**
     * @brief Set the camera view parameters
     * 
     * @param eye camera eye
     * @param center view center
     * @param up view upward direction
     * @return HRESULT 
     */
    CPPAPI_FUNC(HRESULT) cameraMoveTo(const FVector3 &eye, const FVector3 &center, const FVector3 &up);

    /**
     * @brief Set the camera view parameters
     * 
     * @param center target camera view center position
     * @param rotation target camera rotation
     * @param zoom target camera zoom
     * @return HRESULT 
     */
    CPPAPI_FUNC(HRESULT) cameraMoveTo(const FVector3 &center, const FQuaternion &rotation, const float &zoom);

    /**
     * @brief Move the camera to view the domain from the bottm
     * 
     * @return HRESULT 
     */
    CPPAPI_FUNC(HRESULT) cameraViewBottom();

    /**
     * @brief Move the camera to view the domain from the top
     * 
     * @return HRESULT 
     */
    CPPAPI_FUNC(HRESULT) cameraViewTop();

    /**
     * @brief Move the camera to view the domain from the left
     * 
     * @return HRESULT 
     */
    CPPAPI_FUNC(HRESULT) cameraViewLeft();

    /**
     * @brief Move the camera to view the domain from the right
     * 
     * @return HRESULT 
     */
    CPPAPI_FUNC(HRESULT) cameraViewRight();

    /**
     * @brief Move the camera to view the domain from the back
     * 
     * @return HRESULT 
     */
    CPPAPI_FUNC(HRESULT) cameraViewBack();

    /**
     * @brief Move the camera to view the domain from the front
     * 
     * @return HRESULT 
     */
    CPPAPI_FUNC(HRESULT) cameraViewFront();

    /**
     * @brief Reset the camera
     * 
     * @return HRESULT 
     */
    CPPAPI_FUNC(HRESULT) cameraReset();

    /* Rotate the camera from the previous (screen) mouse position to the
    current (screen) position */
    CPPAPI_FUNC(HRESULT) cameraRotateMouse(const iVector2 &mousePos);

    /* Translate the camera from the previous (screen) mouse position to
    the current (screen) mouse position */
    CPPAPI_FUNC(HRESULT) cameraTranslateMouse(const iVector2 &mousePos);

    /**
     * @brief Translate the camera down
     * 
     * @return HRESULT 
     */
    CPPAPI_FUNC(HRESULT) cameraTranslateDown();

    /**
     * @brief Translate the camera up
     * 
     * @return HRESULT 
     */
    CPPAPI_FUNC(HRESULT) cameraTranslateUp();

    /**
     * @brief Translate the camera right
     * 
     * @return HRESULT 
     */
    CPPAPI_FUNC(HRESULT) cameraTranslateRight();

    /**
     * @brief Translate the camera left
     * 
     * @return HRESULT 
     */
    CPPAPI_FUNC(HRESULT) cameraTranslateLeft();

    /**
     * @brief Translate the camera forward
     * 
     * @return HRESULT 
     */
    CPPAPI_FUNC(HRESULT) cameraTranslateForward();

    /**
     * @brief Translate the camera backward
     * 
     * @return HRESULT 
     */
    CPPAPI_FUNC(HRESULT) cameraTranslateBackward();

    /**
     * @brief Rotate the camera down
     * 
     * @return HRESULT 
     */
    CPPAPI_FUNC(HRESULT) cameraRotateDown();

    /**
     * @brief Rotate the camera up
     * 
     * @return HRESULT 
     */
    CPPAPI_FUNC(HRESULT) cameraRotateUp();

    /**
     * @brief Rotate the camera left
     * 
     * @return HRESULT 
     */
    CPPAPI_FUNC(HRESULT) cameraRotateLeft();

    /**
     * @brief Rotate the camera right
     * 
     * @return HRESULT 
     */
    CPPAPI_FUNC(HRESULT) cameraRotateRight();

    /**
     * @brief Roll the camera left
     * 
     * @return HRESULT 
     */
    CPPAPI_FUNC(HRESULT) cameraRollLeft();

    /**
     * @brief Rotate the camera right
     * 
     * @return HRESULT 
     */
    CPPAPI_FUNC(HRESULT) cameraRollRight();

    /**
     * @brief Zoom the camera in
     * 
     * @return HRESULT 
     */
    CPPAPI_FUNC(HRESULT) cameraZoomIn();

    /**
     * @brief Zoom the camera out
     * 
     * @return HRESULT 
     */
    CPPAPI_FUNC(HRESULT) cameraZoomOut();

    /* Rotate the camera from the previous (screen) mouse position to the
    current (screen) position */
    CPPAPI_FUNC(HRESULT) cameraInitMouse(const iVector2 &mousePos);

    /* Translate the camera by the delta amount of (NDC) mouse position.
    Note that NDC position must be in [-1, -1] to [1, 1]. */
    CPPAPI_FUNC(HRESULT) cameraTranslateBy(const FVector2 &trans);

    /**
     * @brief Zoom the camera by an increment in distance. 
     * 
     * Positive values zoom in. 
     * 
     * @param delta zoom increment
     * @return HRESULT 
     */
    CPPAPI_FUNC(HRESULT) cameraZoomBy(const float &delta);

    /**
     * @brief Zoom the camera to a distance. 
     * 
     * @param distance zoom distance
     * @return HRESULT 
     */
    CPPAPI_FUNC(HRESULT) cameraZoomTo(const float &distance);

    /**
     * @brief Rotate the camera to a point from the view center a distance along an axis. 
     * 
     * Only rotates the view to the given eye position.
     * 
     * @param axis axis from the view center
     * @param distance distance along the axis
     * @return HRESULT 
     */
    CPPAPI_FUNC(HRESULT) cameraRotateToAxis(const FVector3 &axis, const float &distance);

    /**
     * @brief Rotate the camera to a set of Euler angles. 
     * 
     * Rotations are Z-Y-X. 
     * 
     * @param angles 
     * @return HRESULT 
     */
    CPPAPI_FUNC(HRESULT) cameraRotateToEulerAngle(const FVector3 &angles);

    /**
     * @brief Rotate the camera by a set of Euler angles. 
     * 
     * Rotations are Z-Y-X. 
     * 
     * @param angles 
     * @return HRESULT 
     */
    CPPAPI_FUNC(HRESULT) cameraRotateByEulerAngle(const FVector3 &angles);

    /**
     * @brief Get the current camera view center position
     * 
     * @return FVector3 
     */
    CPPAPI_FUNC(FVector3) cameraCenter();

    /**
     * @brief Get the current camera rotation
     * 
     * @return FQuaternion 
     */
    CPPAPI_FUNC(FQuaternion) cameraRotation();

    /**
     * @brief Get the current camera zoom
     * 
     * @return float 
     */
    CPPAPI_FUNC(float) cameraZoom();
    
    /**
     * @brief Get the universe renderer
     * 
     * @return struct rendering::UniverseRenderer* 
     */
    CPPAPI_FUNC(struct rendering::UniverseRenderer*) getRenderer();

    /** Get whether bonds are renderered with 3D objects */
    CPPAPI_FUNC(const bool) getRendering3DBonds();

    /** Set whether bonds are renderered with 3D objects */
    CPPAPI_FUNC(void) setRendering3DBonds(const bool &_flag);

    /** Toggle whether bonds are renderered with 3D objects */
    CPPAPI_FUNC(void) toggleRendering3DBonds();

    /** Get whether angles are renderered with 3D objects */
    CPPAPI_FUNC(const bool) getRendering3DAngles();

    /** Set whether angles are renderered with 3D objects */
    CPPAPI_FUNC(void) setRendering3DAngles(const bool &_flag);

    /** Toggle whether angles are renderered with 3D objects */
    CPPAPI_FUNC(void) toggleRendering3DAngles();

    /** Get whether dihedrals are renderered with 3D objects */
    CPPAPI_FUNC(const bool) getRendering3DDihedrals();

    /** Set whether dihedrals are renderered with 3D objects */
    CPPAPI_FUNC(void) setRendering3DDihedrals(const bool &_flag);

    /** Toggle whether dihedrals are renderered with 3D objects */
    CPPAPI_FUNC(void) toggleRendering3DDihedrals();

    /** Set whether bonds, angle and dihedrals are renderered with 3D objects */
    CPPAPI_FUNC(void) setRendering3DAll(const bool &_flag);

    /** Toggle whether bonds, angle and dihedrals are renderered with 3D objects */
    CPPAPI_FUNC(void) toggleRendering3DAll();

    /** Get the line width */
    CPPAPI_FUNC(const FloatP_t) getLineWidth();

    /** Set the line width */
    CPPAPI_FUNC(HRESULT) setLineWidth(const FloatP_t &lineWidth);

    /** Get the minimum line width */
    CPPAPI_FUNC(const FloatP_t) getLineWidthMin();

    /** Get the maximum line width */
    CPPAPI_FUNC(const FloatP_t) getLineWidthMax();

    /**
     * @brief Get the ambient color
     * 
     * @return FVector3 
     */
    CPPAPI_FUNC(FVector3) getAmbientColor();

    /**
     * @brief Set the ambient color
     * 
     * @param color 
     * @return HRESULT 
     */
    CPPAPI_FUNC(HRESULT) setAmbientColor(const FVector3 &color);

    /**
     * @brief Set the ambient color of a subrenderer
     * 
     * @param color 
     * @param srFlag 
     * @return HRESULT 
     */
    CPPAPI_FUNC(HRESULT) setAmbientColor(const FVector3 &color, const unsigned int &srFlag);

    /**
     * @brief Get the diffuse color
     * 
     * @return FVector3 
     */
    CPPAPI_FUNC(FVector3) getDiffuseColor();

    /**
     * @brief Set the diffuse color
     * 
     * @param color 
     * @return HRESULT 
     */
    CPPAPI_FUNC(HRESULT) setDiffuseColor(const FVector3 &color);

    /**
     * @brief Set the diffuse color of a subrenderer
     * 
     * @param color 
     * @param srFlag 
     * @return HRESULT 
     */
    CPPAPI_FUNC(HRESULT) setDiffuseColor(const FVector3 &color, const unsigned int &srFlag);

    /**
     * @brief Get specular color
     * 
     * @return FVector3 
     */
    CPPAPI_FUNC(FVector3) getSpecularColor();

    /**
     * @brief Set the specular color
     * 
     * @param color 
     * @return HRESULT 
     */
    CPPAPI_FUNC(HRESULT) setSpecularColor(const FVector3 &color);

    /**
     * @brief Set the specular color of a subrenderer
     * 
     * @param color 
     * @param srFlag 
     * @return HRESULT 
     */
    CPPAPI_FUNC(HRESULT) setSpecularColor(const FVector3 &color, const unsigned int &srFlag);

    /**
     * @brief Get the shininess
     * 
     * @return float 
     */
    CPPAPI_FUNC(float) getShininess();

    /**
     * @brief Set the shininess
     * 
     * @param shininess 
     * @return HRESULT 
     */
    CPPAPI_FUNC(HRESULT) setShininess(const float &shininess);

    /**
     * @brief Set the shininess of a subrenderer
     * 
     * @param shininess 
     * @param srFlag 
     * @return HRESULT 
     */
    CPPAPI_FUNC(HRESULT) setShininess(const float &shininess, const unsigned int &srFlag);

    /**
     * @brief Get the grid color
     * 
     * @return FVector3 
     */
    CPPAPI_FUNC(FVector3) getGridColor();

    /**
     * @brief Set the grid color
     * 
     * @param color 
     * @return HRESULT 
     */
    CPPAPI_FUNC(HRESULT) setGridColor(const FVector3 &color);

    /**
     * @brief Get the scene box color
     * 
     * @return FVector3 
     */
    CPPAPI_FUNC(FVector3) getSceneBoxColor();

    /**
     * @brief Set the scene box color
     * 
     * @param color 
     * @return HRESULT 
     */
    CPPAPI_FUNC(HRESULT) setSceneBoxColor(const FVector3 &color);

    /**
     * @brief Get the light direction
     * 
     * @return FVector3 
     */
    CPPAPI_FUNC(FVector3) getLightDirection();

    /**
     * @brief Set the light direction
     * 
     * @param lightDir 
     * @return HRESULT 
     */
    CPPAPI_FUNC(HRESULT) setLightDirection(const FVector3& lightDir);

    /**
     * @brief Set the light direction of a subrenderer
     * 
     * @param lightDir 
     * @param srFlag 
     * @return HRESULT 
     */
    CPPAPI_FUNC(HRESULT) setLightDirection(const FVector3& lightDir, const unsigned int &srFlag);

    /**
     * @brief Get the light color
     * 
     * @return FVector3 
     */
    CPPAPI_FUNC(FVector3) getLightColor();

    /**
     * @brief Set the light color
     * 
     * @param color 
     * @return HRESULT 
     */
    CPPAPI_FUNC(HRESULT) setLightColor(const FVector3 &color);

    /**
     * @brief Set the light color of a subrenderer
     * 
     * @param color 
     * @param srFlag 
     * @return HRESULT 
     */
    CPPAPI_FUNC(HRESULT) setLightColor(const FVector3 &color, const unsigned int &srFlag);

    /**
     * @brief Get the background color
     * 
     * @return FVector3 
     */
    CPPAPI_FUNC(FVector3) getBackgroundColor();

    /**
     * @brief Set the background color
     * 
     * @param color 
     * @return HRESULT 
     */
    CPPAPI_FUNC(HRESULT) setBackgroundColor(const FVector3 &color);

    /**
     * @brief Test whether the rendered scene is decorated
     * 
     * @return true 
     * @return false 
     */
    CPPAPI_FUNC(bool) decorated();

    /**
     * @brief Set flag to draw/not draw scene decorators (e.g., grid)
     * 
     * @param decorate flag; true says to decorate
     * @return HRESULT 
     */
    CPPAPI_FUNC(HRESULT) decorateScene(const bool &decorate);

    /**
     * @brief Test whether discretization is current shown
     * 
     * @return true 
     * @return false 
     */
    CPPAPI_FUNC(bool) showingDiscretization();

    /**
     * @brief Set flag to draw/not draw discretization
     * 
     * @param show flag; true says to show
     * @return HRESULT 
     */
    CPPAPI_FUNC(HRESULT) showDiscretization(const bool &show);

    /**
     * @brief Get the current color of the discretization grid
     * 
     * @return FVector3 
     */
    CPPAPI_FUNC(FVector3) getDiscretizationColor();

    /**
     * @brief Set the color of the discretization grid
     * 
     * @param color 
     * @return HRESULT 
     */
    CPPAPI_FUNC(HRESULT) setDiscretizationColor(const FVector3 &color);

    /* Update screen size after the window has been resized */
    CPPAPI_FUNC(HRESULT) viewReshape(const iVector2 &windowSize);

    CPPAPI_FUNC(std::string) performanceCounters();

    /**
     * @brief Get CPU info
     * 
     * @return std::unordered_map<std::string, bool> 
     */
    CPPAPI_FUNC(std::unordered_map<std::string, bool>) cpu_info();

    /**
     * @brief Get compiler flags of this installation
     * 
     * @return std::list<std::string> 
     */
    CPPAPI_FUNC(std::list<std::string>) compile_flags();

    /**
     * @brief Get OpenGL info
     * 
     * @return std::unordered_map<std::string, std::string> 
     */
    CPPAPI_FUNC(std::unordered_map<std::string, std::string>) gl_info();

    /**
     * @brief Get EGL info
     * 
     * @return std::string 
     */
    CPPAPI_FUNC(std::string) egl_info();

    /**
     * @brief Get all available color mapper names
     */
    CPPAPI_FUNC(std::vector<std::string>) colorMapperNames();

    /**
     * @brief Adds a vector visualization specification. 
     * 
     * The passed pointer is borrowed. The client is 
     * responsible for maintaining the underlying data. 
     * The returned integer can be used to reference the 
     * arrow when doing subsequent operations with the 
     * renderer (e.g., removing an arrow from the scene). 
     * 
     * @param arrow pointer to visualization specs
     * @return id of arrow according to the renderer
     */
    CPPAPI_FUNC(int) addRenderArrow(rendering::ArrowData *arrow);

    /**
     * @brief Adds a vector visualization specification. 
     * 
     * The passed pointer is borrowed. The client is 
     * responsible for maintaining the underlying data. 
     * The returned integer can be used to reference the 
     * arrow when doing subsequent operations with the 
     * renderer (e.g., removing an arrow from the scene). 
     * 
     * @param position position of vector
     * @param components components of vector
     * @param style style of vector
     * @param scale scale of vector; defaults to 1.0
     * @return id of arrow according to the renderer and arrow
     */
    CPPAPI_FUNC(std::pair<int, rendering::ArrowData*>) addRenderArrow(
        const FVector3 &position, 
        const FVector3 &components, 
        const rendering::Style &style, 
        const float &scale=1.0
    );

    /**
     * @brief Removes a vector visualization specification. 
     * 
     * The removed pointer is only forgotten. The client is 
     * responsible for clearing the underlying data. 
     * 
     * @param arrowId id of arrow according to the renderer
     */
    CPPAPI_FUNC(HRESULT) removeRenderArrow(const int &arrowId);

    /**
     * @brief Gets a vector visualization specification. 
     * 
     * @param arrowId id of arrow according to the renderer
     */
    CPPAPI_FUNC(rendering::ArrowData*) getRenderArrow(const int &arrowId);


    CPPAPI_FUNC(int) addButton(CallbackVoidOutput& cb, const std::string& label);
    CPPAPI_FUNC(HRESULT) showTime();
    CPPAPI_FUNC(HRESULT) showParticleNumber();

    CPPAPI_FUNC(int) addOutputInt(const int& val, const std::string& label);
    CPPAPI_FUNC(int) addOutputFloat(const float& val, const std::string& label);
    CPPAPI_FUNC(int) addOutputDouble(const double& val, const std::string& label);
    CPPAPI_FUNC(int) addOutputString(const std::string& val, const std::string& label);

    CPPAPI_FUNC(int) addInputInt(CallbackInputInt& cb, const int& val, const std::string& label);
    CPPAPI_FUNC(int) addInputFloat(CallbackInputFloat& cb, const float& val, const std::string& label);
    CPPAPI_FUNC(int) addInputDouble(CallbackInputDouble& cb, const double& val, const std::string& label);
    CPPAPI_FUNC(int) addInputString(CallbackInputString& cb, const std::string& val, const std::string& label);

    CPPAPI_FUNC(HRESULT) setOutputInt(const unsigned int& idx, const int& val);
    CPPAPI_FUNC(HRESULT) setOutputFloat(const unsigned int& idx, const float& val);
    CPPAPI_FUNC(HRESULT) setOutputDouble(const unsigned int& idx, const double& val);
    CPPAPI_FUNC(HRESULT) setOutputString(const unsigned int& idx, const std::string& val);


    CPPAPI_FUNC(int) addWidgetButton(CallbackVoidOutput& cb, const std::string& label);
    CPPAPI_FUNC(HRESULT) showWidgetTime();
    CPPAPI_FUNC(HRESULT) showWidgetParticleNumber();
    CPPAPI_FUNC(HRESULT) showWidgetBondNumber();
    CPPAPI_FUNC(HRESULT) showWidgetDihedralNumber();
    CPPAPI_FUNC(HRESULT) showWidgetAngleNumber();

    CPPAPI_FUNC(int) addWidgetOutputInt(const int& val, const std::string& label);
    CPPAPI_FUNC(int) addWidgetOutputFloat(const float& val, const std::string& label);
    CPPAPI_FUNC(int) addWidgetOutputDouble(const double& val, const std::string& label);
    CPPAPI_FUNC(int) addWidgetOutputString(const std::string& val, const std::string& label);

    CPPAPI_FUNC(int) addWidgetInputInt(CallbackInputInt& cb, const int& val, const std::string& label);
    CPPAPI_FUNC(int) addWidgetInputFloat(CallbackInputFloat& cb, const float& val, const std::string& label);
    CPPAPI_FUNC(int) addWidgetInputDouble(CallbackInputDouble& cb, const double& val, const std::string& label);
    CPPAPI_FUNC(int) addWidgetInputString(CallbackInputString& cb, const std::string& val, const std::string& label);

    CPPAPI_FUNC(HRESULT) setWidgetOutputInt(const unsigned int& idx, const int& val);
    CPPAPI_FUNC(HRESULT) setWidgetOutputFloat(const unsigned int& idx, const float& val);
    CPPAPI_FUNC(HRESULT) setWidgetOutputDouble(const unsigned int& idx, const double& val);
    CPPAPI_FUNC(HRESULT) setWidgetOutputString(const unsigned int& idx, const std::string& val);

    CPPAPI_FUNC(HRESULT) setWidgetFontSize(const float size);
    CPPAPI_FUNC(HRESULT) setWidgetTextColor(const std::string& colorName);
    CPPAPI_FUNC(HRESULT) setWidgetBackgroundColor(const std::string& colorName);

    CPPAPI_FUNC(HRESULT) setWidgetTextColor(float r, float g, float b, float a = 1.0f);
    CPPAPI_FUNC(HRESULT) setWidgetTextColor(const TissueForge::FVector3& color);
    CPPAPI_FUNC(HRESULT) setWidgetTextColor(const TissueForge::FVector4& color);

    CPPAPI_FUNC(HRESULT) setWidgetBackgroundColor(float r, float g, float b, float a = 1.0f);
    CPPAPI_FUNC(HRESULT) setWidgetBackgroundColor(const TissueForge::FVector3& color);
    CPPAPI_FUNC(HRESULT) setWidgetBackgroundColor(const TissueForge::FVector4& color);

    CPPAPI_FUNC(float) getWidgetFontSize();
    CPPAPI_FUNC(TissueForge::FVector4) getWidgetTextColor();
    CPPAPI_FUNC(TissueForge::FVector4) getWidgetBackgroundColor();
};

#endif // _SOURCE_TF_SYSTEM_H_
