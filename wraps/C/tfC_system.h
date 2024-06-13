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
 * @file tfC_system.h
 * 
 */

#ifndef _WRAPS_C_TFC_SYSTEM_H_
#define _WRAPS_C_TFC_SYSTEM_H_

#include "tf_port_c.h"


//////////////////////
// Module functions //
//////////////////////


/**
 * @brief Get the current image data
 * 
 * @param imgData image data
 * @param imgSize image size
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfSystem_imageData(char **imgData, size_t *imgSize);

/**
* @brief Save a screenshot of the current scene
* 
* @param filePath path of file to save
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfSystem_screenshot(const char *filePath);

/**
* @brief Save a screenshot of the current scene
* 
* @param filePath path of file to save
* @param decorate flag to decorate the scene in the screenshot
* @param bgcolor background color of the scene in the screenshot
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfSystem_screenshotS(const char *filePath, bool decorate, float *bgcolor);

/**
 * @brief Test whether the context is current
 * 
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfSystem_contextHasCurrent(bool *current);

/**
 * @brief Make the context current
 * 
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfSystem_contextMakeCurrent();

/**
 * @brief Release the current context
 * 
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfSystem_contextRelease();

/**
 * @brief Test whether the camera is lagging
 * 
 * @return value of test
 */
CAPI_FUNC(bool) tfSystem_cameraIsLagging();

/**
 * @brief Enable camera lagging
 * 
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfSystem_cameraEnableLagging();

/**
 * @brief Disable camera lagging
 * 
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfSystem_cameraDisableLagging();

/**
 * @brief Toggle camera lagging
 * 
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfSystem_cameraToggleLagging();

/**
 * @brief Get the camera lagging
 * 
 * @return value of lagging
 */
CAPI_FUNC(float) tfSystem_cameraGetLagging();

/**
 * @brief Set the camera lagging. Value must be in [0, 1)
 * 
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfSystem_cameraSetLagging(float lagging);

/**
* @brief Set the camera view parameters
* 
* @param eye camera eye
* @param center view center
* @param up view upward direction
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfSystem_cameraMoveTo(float *eye, float *center, float *up);

/**
* @brief Set the camera view parameters
* 
* @param center target camera view center position
* @param rotation target camera rotation
* @param zoom target camera zoom
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfSystem_cameraMoveToR(float *center, float *rotation, float zoom);

/**
* @brief Move the camera to view the domain from the bottm
* 
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfSystem_cameraViewBottom();

/**
* @brief Move the camera to view the domain from the top
* 
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfSystem_cameraViewTop();

/**
* @brief Move the camera to view the domain from the left
* 
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfSystem_cameraViewLeft();

/**
* @brief Move the camera to view the domain from the right
* 
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfSystem_cameraViewRight();

/**
* @brief Move the camera to view the domain from the back
* 
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfSystem_cameraViewBack();

/**
* @brief Move the camera to view the domain from the front
* 
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfSystem_cameraViewFront();

/**
* @brief Reset the camera
* 
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfSystem_cameraReset();

/**
 * @brief Rotate the camera from the previous (screen) mouse position to the current (screen) position
 * 
 * @param x horizontal coordinate
 * @param y vertical coordinate
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfSystem_cameraRotateMouse(int x, int y);

/**
 * @brief Translate the camera from the previous (screen) mouse position to the current (screen) mouse position
 * 
 * @param x horizontal coordinate
 * @param y vertical coordintae
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfSystem_cameraTranslateMouse(int x, int y);

/**
* @brief Translate the camera down
* 
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfSystem_cameraTranslateDown();

/**
* @brief Translate the camera up
* 
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfSystem_cameraTranslateUp();

/**
* @brief Translate the camera right
* 
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfSystem_cameraTranslateRight();

/**
* @brief Translate the camera left
* 
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfSystem_cameraTranslateLeft();

/**
* @brief Translate the camera forward
* 
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfSystem_cameraTranslateForward();

/**
* @brief Translate the camera backward
* 
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfSystem_cameraTranslateBackward();

/**
* @brief Rotate the camera down
* 
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfSystem_cameraRotateDown();

/**
* @brief Rotate the camera up
* 
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfSystem_cameraRotateUp();

/**
* @brief Rotate the camera left
* 
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfSystem_cameraRotateLeft();

/**
* @brief Rotate the camera right
* 
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfSystem_cameraRotateRight();

/**
* @brief Roll the camera left
* 
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfSystem_cameraRollLeft();

/**
* @brief Rotate the camera right
* 
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfSystem_cameraRollRight();

/**
* @brief Zoom the camera in
* 
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfSystem_cameraZoomIn();

/**
* @brief Zoom the camera out
* 
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfSystem_cameraZoomOut();

/**
 * @brief Initialize the camera at a mouse position
 * 
 * @param x horizontal coordinate
 * @param y vertical coordinate
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfSystem_cameraInitMouse(int x, int y);

/**
 * @brief Translate the camera by the delta amount of (NDC) mouse position. 
 * 
 * Note that NDC position must be in [-1, -1] to [1, 1].
 * 
 * @param x horizontal delta
 * @param y vertical delta
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfSystem_cameraTranslateBy(float x, float y);

/**
* @brief Zoom the camera by an increment in distance. 
* 
* Positive values zoom in. 
* 
* @param delta zoom increment
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfSystem_cameraZoomBy(float delta);

/**
* @brief Zoom the camera to a distance. 
* 
* @param distance zoom distance
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfSystem_cameraZoomTo(float distance);

/**
* @brief Rotate the camera to a point from the view center a distance along an axis. 
* 
* Only rotates the view to the given eye position.
* 
* @param axis axis from the view center
* @param distance distance along the axis
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfSystem_cameraRotateToAxis(float *axis, float distance);

/**
* @brief Rotate the camera to a set of Euler angles. 
* 
* Rotations are Z-Y-X. 
* 
* @param angles 
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfSystem_cameraRotateToEulerAngle(float *angles);

/**
* @brief Rotate the camera by a set of Euler angles. 
* 
* Rotations are Z-Y-X. 
* 
* @param angles 
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfSystem_cameraRotateByEulerAngle(float *angles);

/**
 * @brief Get the current camera view center position
 * 
 * @param center camera center
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfSystem_cameraCenter(float **center);

/**
 * @brief Get the current camera rotation
 * 
 * @param rotation camera rotation
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfSystem_cameraRotation(float **rotation);

/**
 * @brief Get the current camera zoom
 * 
 * @param zoom camera zoom
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfSystem_cameraZoom(float *zoom);

/** Get whether bonds are renderered with 3D objects */
CAPI_FUNC(HRESULT) tfSystem_getRendering3DBonds(bool *flag);

/** Set whether bonds are renderered with 3D objects */
CAPI_FUNC(HRESULT) tfSystem_setRendering3DBonds(bool flag);

/** Toggle whether bonds are renderered with 3D objects */
CAPI_FUNC(HRESULT) tfSystem_toggleRendering3DBonds();

/** Get whether angles are renderered with 3D objects */
CAPI_FUNC(HRESULT) tfSystem_getRendering3DAngles(bool *flag);

/** Set whether angles are renderered with 3D objects */
CAPI_FUNC(HRESULT) tfSystem_setRendering3DAngles(bool flag);

/** Toggle whether angles are renderered with 3D objects */
CAPI_FUNC(HRESULT) tfSystem_toggleRendering3DAngles();

/** Get whether dihedrals are renderered with 3D objects */
CAPI_FUNC(HRESULT) tfSystem_getRendering3DDihedrals(bool *flag);

/** Set whether dihedrals are renderered with 3D objects */
CAPI_FUNC(HRESULT) tfSystem_setRendering3DDihedrals(bool flag);

/** Toggle whether dihedrals are renderered with 3D objects */
CAPI_FUNC(HRESULT) tfSystem_toggleRendering3DDihedrals();

/** Set whether bonds, angle and dihedrals are renderered with 3D objects */
CAPI_FUNC(HRESULT) tfSystem_setRendering3DAll(bool flag);

/** Toggle whether bonds, angle and dihedrals are renderered with 3D objects */
CAPI_FUNC(HRESULT) tfSystem_toggleRendering3DAll();

/** Get the line width */
CAPI_FUNC(HRESULT) tfSystem_getLineWidth(tfFloatP_t *lineWidth);

/** Set the line width */
CAPI_FUNC(HRESULT) tfSystem_setLineWidth(tfFloatP_t lineWidth);

/** Get the minimum line width */
CAPI_FUNC(HRESULT) tfSystem_getLineWidthMin(tfFloatP_t *lineWidth);

/** Get the maximum line width */
CAPI_FUNC(HRESULT) tfSystem_getLineWidthMax(tfFloatP_t *lineWidth);

/**
 * @brief Get the ambient color
 * 
 * @param color 
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfSystem_getAmbientColor(float **color);

/**
 * @brief Set the ambient color
 * 
 * @param color 
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfSystem_setAmbientColor(float *color);

/**
 * @brief Get the diffuse color
 * 
 * @param color 
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfSystem_getDiffuseColor(float **color);

/**
* @brief Set the diffuse color
* 
* @param color 
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfSystem_setDiffuseColor(float *color);

/**
 * @brief Get specular color
 * 
 * @param color 
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfSystem_getSpecularColor(float **color);

/**
* @brief Set the specular color
* 
* @param color 
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfSystem_setSpecularColor(float *color);

/**
 * @brief Get the shininess
 * 
 * @param shininess 
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfSystem_getShininess(float *shininess);

/**
* @brief Set the shininess
* 
* @param shininess 
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfSystem_setShininess(float shininess);

/**
 * @brief Get the grid color
 * 
 * @param color 
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfSystem_getGridColor(float **color);

/**
* @brief Set the grid color
* 
* @param color 
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfSystem_setGridColor(float *color);

/**
 * @brief Get the scene box color
 * 
 * @param color 
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfSystem_getSceneBoxColor(float **color);

/**
* @brief Set the scene box color
* 
* @param color 
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfSystem_setSceneBoxColor(float *color);

/**
 * @brief Get the light direction
 * 
 * @param lightDir 
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfSystem_getLightDirection(float **lightDir);

/**
* @brief Set the light direction
* 
* @param lightDir 
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfSystem_setLightDirection(float *lightDir);

/**
 * @brief Get the light color
 * 
 * @param color 
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfSystem_getLightColor(float **color);

/**
* @brief Set the light color
* 
* @param color 
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfSystem_setLightColor(float *color);

/**
 * @brief Get the background color
 * 
 * @param color 
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfSystem_getBackgroundColor(float **color);

/**
* @brief Set the background color
* 
* @param color 
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfSystem_setBackgroundColor(float *color);

/**
 * @brief Test whether the rendered scene is decorated
 * 
 * @param decorated 
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfSystem_decorated(bool *decorated);

/**
* @brief Set flag to draw/not draw scene decorators (e.g., grid)
* 
* @param decorate flag; true says to decorate
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfSystem_decorateScene(bool decorate);

/**
 * @brief Test whether discretization is currently shown
 * 
 * @param showing 
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfSystem_showingDiscretization(bool *showing);

/**
* @brief Set flag to draw/not draw discretization
* 
* @param show flag; true says to show
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfSystem_showDiscretization(bool show);

/**
 * @brief Get the current color of the discretization grid
 * 
 * @param color 
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfSystem_getDiscretizationColor(float **color);

/**
* @brief Set the color of the discretization grid
* 
* @param color 
* @return S_OK on success 
*/
CAPI_FUNC(HRESULT) tfSystem_setDiscretizationColor(float *color);

/**
 * @brief Update screen size after the window has been resized
 * 
 * @param sizex horizontal size
 * @param sizey vertical size
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfSystem_viewReshape(int sizex, int sizey);

/**
 * @brief Get CPU info
 * 
 * @param names entry name
 * @param flags entry flag
 * @param numNames number of entries
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfSystem_getCPUInfo(char ***names, bool **flags, unsigned int *numNames);

/**
 * @brief Get compiler flags of this installation
 * 
 * @param flags compiler flags
 * @param numFlags number of flags
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfSystem_getCompileFlags(char ***flags, unsigned int *numFlags);

/**
 * @brief Get OpenGL info
 * 
 * @param names 
 * @param values 
 * @param numNames 
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfSystem_getGLInfo(char ***names, char ***values, unsigned int *numNames);

/**
 * @brief Get EGL info
 * 
 * @param info 
 * @param numChars
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfSystem_getEGLInfo(char **info, unsigned int *numChars);

#endif // _WRAPS_C_TFC_SYSTEM_H_