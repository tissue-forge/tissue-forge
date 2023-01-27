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

#include "tf_systemPy.h"
#include <tf_system.h>
#include <rendering/tfApplication.h>


using namespace TissueForge;


void py::print_performance_counters() { system::printPerformanceCounters(); }

HRESULT py::screenshot(const std::string &filePath) { return system::screenshot(filePath); }

HRESULT py::screenshot(const std::string &filePath, const bool &decorate, const FVector3 &bgcolor) { return system::screenshot(filePath, decorate, bgcolor); }

bool py::camera_is_lagging() { return system::cameraIsLagging(); }

HRESULT py::camera_enable_lagging() { return system::cameraEnableLagging(); }

HRESULT py::camera_disable_lagging() { return system::cameraDisableLagging(); }

HRESULT py::camera_toggle_lagging() { return system::cameraToggleLagging(); }

float py::camera_get_lagging() { return system::cameraGetLagging(); }

HRESULT py::camera_set_lagging(const float &lagging) { return system::cameraSetLagging(lagging); }

bool py::context_has_current() { return system::contextHasCurrent(); }

HRESULT py::context_make_current() { return system::contextMakeCurrent(); }

HRESULT py::context_release() { return system::contextRelease(); }

HRESULT py::camera_move_to(const FVector3 &eye, const FVector3 &center, const FVector3 &up) { return system::cameraMoveTo(eye, center, up); }

HRESULT py::camera_move_to(const FVector3 &center, const FQuaternion &rotation, const float &zoom) { return system::cameraMoveTo(center, rotation, zoom); }

HRESULT py::camera_view_bottom() { return system::cameraViewBottom(); }

HRESULT py::camera_view_top() { return system::cameraViewTop(); }

HRESULT py::camera_view_left() { return system::cameraViewLeft(); }

HRESULT py::camera_view_right() { return system::cameraViewRight(); }

HRESULT py::camera_view_back() {  return system::cameraViewBack(); }

HRESULT py::camera_view_front() { return system::cameraViewFront(); }

HRESULT py::camera_reset() { return system::cameraReset(); }

HRESULT py::camera_rotate_mouse(const iVector2 &mousePos) { return system::cameraRotateMouse(mousePos); }

HRESULT py::camera_translate_mouse(const iVector2 &mousePos) { return system::cameraTranslateMouse(mousePos); }

HRESULT py::camera_translate_down() { return system::cameraTranslateDown(); }

HRESULT py::camera_translate_up() { return system::cameraTranslateUp(); }

HRESULT py::camera_translate_right() { return system::cameraTranslateRight(); }

HRESULT py::camera_translate_left() { return system::cameraTranslateLeft(); }

HRESULT py::camera_translate_forward() { return system::cameraTranslateForward(); }

HRESULT py::camera_translate_backward() { return system::cameraTranslateBackward(); }

HRESULT py::camera_rotate_down() { return system::cameraRotateDown(); }

HRESULT py::camera_rotate_up() { return system::cameraRotateUp(); }

HRESULT py::camera_rotate_left() { return system::cameraRotateLeft(); }

HRESULT py::camera_rotate_right() { return system::cameraRotateRight(); }

HRESULT py::camera_roll_left() { return system::cameraRollLeft(); }

HRESULT py::camera_roll_right() { return system::cameraRollRight(); }

HRESULT py::camera_zoom_in() { return system::cameraZoomIn(); }

HRESULT py::camera_zoom_out() { return system::cameraZoomOut(); }

HRESULT py::camera_init_mouse(const iVector2 &mousePos) { return system::cameraInitMouse(mousePos); }

HRESULT py::camera_translate_by(const FVector2 &trans) { return system::cameraTranslateBy(trans); }

HRESULT py::camera_zoom_by(const float &delta) { return system::cameraZoomBy(delta); }

HRESULT py::camera_zoom_to(const float &distance) { return system::cameraZoomTo(distance); }

HRESULT py::camera_rotate_to_axis(const FVector3 &axis, const float &distance) { return system::cameraRotateToAxis(axis, distance); }

HRESULT py::camera_rotate_to_euler_angle(const FVector3 &angles) { return system::cameraRotateToEulerAngle(angles); }

HRESULT py::camera_rotate_by_euler_angle(const FVector3 &angles) { return system::cameraRotateByEulerAngle(angles); }

FVector3 py::camera_center() { return system::cameraCenter(); }

FQuaternion py::camera_rotation() { return system::cameraRotation(); }

float py::camera_zoom() { return system::cameraZoom(); }

struct rendering::UniverseRenderer* py::get_renderer() { return system::getRenderer(); }

const bool py::get_rendering_3d_bonds() { return system::getRendering3DBonds(); }

void py::set_rendering_3d_bonds(const bool &_flag) { return system::setRendering3DBonds(_flag); }

void py::toggle_rendering_3d_bonds() { return system::toggleRendering3DBonds(); }

const bool py::get_rendering_3d_angles() { return system::getRendering3DAngles(); }

void py::set_rendering_3d_angles(const bool &_flag) { return system::setRendering3DAngles(_flag); }

void py::toggle_rendering_3d_angles() { return system::toggleRendering3DAngles(); }

const bool py::get_rendering_3d_dihedrals() { return system::getRendering3DDihedrals(); }

void py::set_rendering_3d_dihedrals(const bool &_flag) { return system::setRendering3DDihedrals(_flag); }

void py::toggle_rendering_3d_dihedrals() { return system::toggleRendering3DDihedrals(); }

void py::set_rendering_3d_all(const bool &_flag) { return system::setRendering3DAll(_flag); }

void py::toggle_rendering_3d_all() { return system::toggleRendering3DAll(); }

FloatP_t py::get_line_width() { return system::getLineWidth(); }

HRESULT py::set_line_width(const FloatP_t &lineWidth) { return system::setLineWidth(lineWidth); }

FloatP_t py::get_line_width_min() { return system::getLineWidthMin(); }

FloatP_t py::get_line_width_max() { return system::getLineWidthMax(); }

FVector3 py::get_ambient_color() { return system::getAmbientColor(); }

HRESULT py::set_ambient_color(const FVector3 &color) { return system::setAmbientColor(color); }

HRESULT py::set_ambient_color(const FVector3 &color, const unsigned int &srFlag) { return system::setAmbientColor(color, srFlag); }

FVector3 py::get_diffuse_color() { return system::getDiffuseColor(); }

HRESULT py::set_diffuse_color(const FVector3 &color) { return system::setDiffuseColor(color); }

HRESULT py::set_diffuse_color(const FVector3 &color, const unsigned int &srFlag) { return system::setDiffuseColor(color, srFlag); }

FVector3 py::get_specular_color() { return system::getSpecularColor(); }

HRESULT py::set_specular_color(const FVector3 &color) { return system::setSpecularColor(color); }

HRESULT py::set_specular_color(const FVector3 &color, const unsigned int &srFlag) { return system::setSpecularColor(color, srFlag); }

float py::get_shininess() { return system::getShininess(); }

HRESULT py::set_shininess(const float &shininess) { return system::setShininess(shininess); }

HRESULT py::set_shininess(const float &shininess, const unsigned int &srFlag) { return system::setShininess(shininess, srFlag); }

FVector3 py::get_grid_color() { return system::getGridColor(); }

HRESULT py::set_grid_color(const FVector3 &color) { return system::setGridColor(color); }

FVector3 py::get_scene_box_color() { return system::getSceneBoxColor(); }

HRESULT py::set_scene_box_color(const FVector3 &color) { return system::setSceneBoxColor(color); }

FVector3 py::get_light_direction() { return system::getLightDirection(); }

HRESULT py::set_light_direction(const FVector3& lightDir) { return system::setLightDirection(lightDir); }

HRESULT py::set_light_direction(const FVector3& lightDir, const unsigned int &srFlag) { return system::setLightDirection(lightDir, srFlag); }

FVector3 py::get_light_color() { return system::getLightColor(); }

HRESULT py::set_light_color(const FVector3 &color) { return system::setLightColor(color); }

HRESULT py::set_light_color(const FVector3 &color, const unsigned int &srFlag) { return system::setLightColor(color, srFlag); }

FVector3 py::get_background_color() { return system::getBackgroundColor(); }

HRESULT py::set_background_color(const FVector3 &color) { return system::setBackgroundColor(color); }

bool py::decorated() { return system::decorated(); }

HRESULT py::decorate_scene(const bool &decorate) { return system::decorateScene(decorate); }

bool py::showing_discretization() { return system::showingDiscretization(); }

HRESULT py::show_discretization(const bool &show) { return system::showDiscretization(show); }

FVector3 py::get_discretizationColor() { return system::getDiscretizationColor(); }

HRESULT py::set_discretizationColor(const FVector3 &color) { return system::setDiscretizationColor(color); }

HRESULT py::view_reshape(const iVector2 &windowSize) { return system::viewReshape(windowSize); }

std::string py::performance_counters() { return system::performanceCounters(); }

std::unordered_map<std::string, bool> py::cpu_info() { return system::cpu_info(); }

std::unordered_map<std::string, bool> py::compile_flags() {
    std::unordered_map<std::string, bool> result;
    util::CompileFlags cfs;
    for(auto &s : system::compile_flags()) 
        result[s] = cfs.getFlag(s);
    return result;
}

std::unordered_map<std::string, std::string> py::gl_info() { return system::gl_info(); }

std::string py::egl_info() { return system::egl_info(); }

PyObject *py::image_data() {
    TF_Log(LOG_TRACE);
    
    auto jpegData = rendering::JpegImageData();
    char *data = jpegData.data();
    size_t size = jpegData.size();
    
    return PyBytes_FromStringAndSize(data, size);
}

bool py::is_terminal_interactive() { return py::terminalInteractiveShell(); }

bool py::is_jupyter_notebook() { return py::ZMQInteractiveShell(); }

PyObject *py::jwidget_init(PyObject *args, PyObject *kwargs) {
    
    PyObject* moduleString = PyUnicode_FromString((char*)"tissue_forge.jwidget");
    
    if(!moduleString) {
        return NULL;
    }
    
    #if defined(__has_feature)
    #  if __has_feature(thread_sanitizer)
        std::cout << "thread sanitizer, returning NULL" << std::endl;
        return NULL;
    #  endif
    #endif
    
    PyObject* module = PyImport_Import(moduleString);
    if(!module) {
        tf_error(E_FAIL, "could not import tissue_forge.jwidget package");
        return NULL;
    }
    
    // Then getting a reference to your function :

    PyObject* init = PyObject_GetAttrString(module,(char*)"init");
    
    if(!init) {
        tf_error(E_FAIL, "tissue_forge.jwidget package does not have an init function");
        return NULL;
    }

    PyObject* result = PyObject_Call(init, args, kwargs);
    
    Py_DECREF(moduleString);
    Py_DECREF(module);
    Py_DECREF(init);
    
    if(!result) {
        TF_Log(LOG_ERROR) << "error calling tissue_forge.jwidget.init: " << py::pyerror_str();
    }
    
    return result;
}

PyObject *py::jwidget_run(PyObject *args, PyObject *kwargs) {
    PyObject* moduleString = PyUnicode_FromString((char*)"tissue_forge.jwidget");
    
    if(!moduleString) {
        return NULL;
    }
    
    #if defined(__has_feature)
    #  if __has_feature(thread_sanitizer)
        std::cout << "thread sanitizer, returning NULL" << std::endl;
        return NULL;
    #  endif
    #endif
    
    PyObject* module = PyImport_Import(moduleString);
    if(!module) {
        tf_error(E_FAIL, "could not import tissue_forge.jwidget package");
        return NULL;
    }
    
    // Then getting a reference to your function :

    PyObject* run = PyObject_GetAttrString(module,(char*)"run");
    
    if(!run) {
        tf_error(E_FAIL, "tissue_forge.jwidget package does not have an run function");
        return NULL;
    }

    PyObject* result = PyObject_Call(run, args, kwargs);

    if (!result) {
        TF_Log(LOG_ERROR) << "error calling tissue_forge.jwidget.run: " << py::pyerror_str();
    }

    Py_DECREF(moduleString);
    Py_DECREF(module);
    Py_DECREF(run);
    
    return result;
    
}

std::vector<std::string> py::color_mapper_names() { return system::colorMapperNames(); }

int py::add_render_arrow(rendering::ArrowData *arrow) { return system::addRenderArrow(arrow); }

std::pair<int, rendering::ArrowData*> py::add_render_arrow(const FVector3 &pos, const FVector3 &comps, const rendering::Style &st, const float &sc) {
    return system::addRenderArrow(pos, comps, st, sc);
}

HRESULT py::remove_render_arrow(const int &arrowId) { return system::removeRenderArrow(arrowId); }

rendering::ArrowData *py::get_render_arrow(const int &arrowId) { return system::getRenderArrow(arrowId); }
