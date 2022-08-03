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

#ifndef _SOURCE_LANGS_PY_TF_SYSTEMPY_H_
#define _SOURCE_LANGS_PY_TF_SYSTEMPY_H_

#include "tf_py.h"

#include <rendering/tfUniverseRenderer.h>
#include <rendering/tfArrowRenderer.h>

#include <list>
#include <unordered_map>


namespace TissueForge::py {


   CPPAPI_FUNC(void) print_performance_counters();

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

   CPPAPI_FUNC(bool) context_has_current();
   CPPAPI_FUNC(HRESULT) context_make_current();
   CPPAPI_FUNC(HRESULT) context_release();

   /**
   * @brief Set the camera view parameters
   * 
   * @param eye camera eye
   * @param center view center
   * @param up view upward direction
   * @return HRESULT 
   */
   CPPAPI_FUNC(HRESULT) camera_move_to(const FVector3 &eye, const FVector3 &center, const FVector3 &up);

   /**
   * @brief Set the camera view parameters
   * 
   * @param center target camera view center position
   * @param rotation target camera rotation
   * @param zoom target camera zoom
   * @return HRESULT 
   */
   CPPAPI_FUNC(HRESULT) camera_move_to(const FVector3 &center, const FQuaternion &rotation, const float &zoom);

   /**
   * @brief Move the camera to view the domain from the bottm
   * 
   * @return HRESULT 
   */
   CPPAPI_FUNC(HRESULT) camera_view_bottom();

   /**
   * @brief Move the camera to view the domain from the top
   * 
   * @return HRESULT 
   */
   CPPAPI_FUNC(HRESULT) camera_view_top();

   /**
   * @brief Move the camera to view the domain from the left
   * 
   * @return HRESULT 
   */
   CPPAPI_FUNC(HRESULT) camera_view_left();

   /**
   * @brief Move the camera to view the domain from the right
   * 
   * @return HRESULT 
   */
   CPPAPI_FUNC(HRESULT) camera_view_right();

   /**
   * @brief Move the camera to view the domain from the back
   * 
   * @return HRESULT 
   */
   CPPAPI_FUNC(HRESULT) camera_view_back();

   /**
   * @brief Move the camera to view the domain from the front
   * 
   * @return HRESULT 
   */
   CPPAPI_FUNC(HRESULT) camera_view_front();

   /**
   * @brief Reset the camera
   * 
   * @return HRESULT 
   */
   CPPAPI_FUNC(HRESULT) camera_reset();

   /* Rotate the camera from the previous (screen) mouse position to the
   current (screen) position */
   CPPAPI_FUNC(HRESULT) camera_rotate_mouse(const iVector2 &mousePos);

   /* Translate the camera from the previous (screen) mouse position to
   the current (screen) mouse position */
   CPPAPI_FUNC(HRESULT) camera_translate_mouse(const iVector2 &mousePos);

   /**
   * @brief Translate the camera down
   * 
   * @return HRESULT 
   */
   CPPAPI_FUNC(HRESULT) camera_translate_down();

   /**
   * @brief Translate the camera up
   * 
   * @return HRESULT 
   */
   CPPAPI_FUNC(HRESULT) camera_translate_up();

   /**
   * @brief Translate the camera right
   * 
   * @return HRESULT 
   */
   CPPAPI_FUNC(HRESULT) camera_translate_right();

   /**
   * @brief Translate the camera left
   * 
   * @return HRESULT 
   */
   CPPAPI_FUNC(HRESULT) camera_translate_left();

   /**
   * @brief Translate the camera forward
   * 
   * @return HRESULT 
   */
   CPPAPI_FUNC(HRESULT) camera_translate_forward();

   /**
   * @brief Translate the camera backward
   * 
   * @return HRESULT 
   */
   CPPAPI_FUNC(HRESULT) camera_translate_backward();

   /**
   * @brief Rotate the camera down
   * 
   * @return HRESULT 
   */
   CPPAPI_FUNC(HRESULT) camera_rotate_down();

   /**
   * @brief Rotate the camera up
   * 
   * @return HRESULT 
   */
   CPPAPI_FUNC(HRESULT) camera_rotate_up();

   /**
   * @brief Rotate the camera left
   * 
   * @return HRESULT 
   */
   CPPAPI_FUNC(HRESULT) camera_rotate_left();

   /**
   * @brief Rotate the camera right
   * 
   * @return HRESULT 
   */
   CPPAPI_FUNC(HRESULT) camera_rotate_right();

   /**
   * @brief Roll the camera left
   * 
   * @return HRESULT 
   */
   CPPAPI_FUNC(HRESULT) camera_roll_left();

   /**
   * @brief Rotate the camera right
   * 
   * @return HRESULT 
   */
   CPPAPI_FUNC(HRESULT) camera_roll_right();

   /**
   * @brief Zoom the camera in
   * 
   * @return HRESULT 
   */
   CPPAPI_FUNC(HRESULT) camera_zoom_in();

   /**
   * @brief Zoom the camera out
   * 
   * @return HRESULT 
   */
   CPPAPI_FUNC(HRESULT) camera_zoom_out();

   /* Rotate the camera from the previous (screen) mouse position to the
   current (screen) position */
   CPPAPI_FUNC(HRESULT) camera_init_mouse(const iVector2 &mousePos);

   /* Translate the camera by the delta amount of (NDC) mouse position.
   Note that NDC position must be in [-1, -1] to [1, 1]. */
   CPPAPI_FUNC(HRESULT) camera_translate_by(const FVector2 &trans);

   /**
   * @brief Zoom the camera by an increment in distance. 
   * 
   * Positive values zoom in. 
   * 
   * @param delta zoom increment
   * @return HRESULT 
   */
   CPPAPI_FUNC(HRESULT) camera_zoom_by(const float &delta);

   /**
   * @brief Zoom the camera to a distance. 
   * 
   * @param distance zoom distance
   * @return HRESULT 
   */
   CPPAPI_FUNC(HRESULT) camera_zoom_to(const float &distance);

   /**
   * @brief Rotate the camera to a point from the view center a distance along an axis. 
   * 
   * Only rotates the view to the given eye position.
   * 
   * @param axis axis from the view center
   * @param distance distance along the axis
   * @return HRESULT 
   */
   CPPAPI_FUNC(HRESULT) camera_rotate_to_axis(const FVector3 &axis, const float &distance);

   /**
   * @brief Rotate the camera to a set of Euler angles. 
   * 
   * Rotations are Z-Y-X. 
   * 
   * @param angles 
   * @return HRESULT 
   */
   CPPAPI_FUNC(HRESULT) camera_rotate_to_euler_angle(const FVector3 &angles);

   /**
   * @brief Rotate the camera by a set of Euler angles. 
   * 
   * Rotations are Z-Y-X. 
   * 
   * @param angles 
   * @return HRESULT 
   */
   CPPAPI_FUNC(HRESULT) camera_rotate_by_euler_angle(const FVector3 &angles);

   /**
   * @brief Get the current camera view center position
   * 
   * @return FVector3 
   */
   CPPAPI_FUNC(FVector3) camera_center();

   /**
   * @brief Get the current camera rotation
   * 
   * @return FQuaternion 
   */
   CPPAPI_FUNC(FQuaternion) camera_rotation();

   /**
   * @brief Get the current camera zoom
   * 
   * @return float 
   */
   CPPAPI_FUNC(float) camera_zoom();

   /**
   * @brief Get the universe renderer
   * 
   * @return struct rendering::UniverseRenderer* 
   */
   CPPAPI_FUNC(struct rendering::UniverseRenderer*) get_renderer();

   /**
   * @brief Get the ambient color
   * 
   * @return FVector3 
   */
   CPPAPI_FUNC(FVector3) get_ambient_color();

   /**
   * @brief Set the ambient color
   * 
   * @param color 
   * @return HRESULT 
   */
   CPPAPI_FUNC(HRESULT) set_ambient_color(const FVector3 &color);

   /**
   * @brief Set the ambient color of a subrenderer
   * 
   * @param color 
   * @param srFlag 
   * @return HRESULT 
   */
   CPPAPI_FUNC(HRESULT) set_ambient_color(const FVector3 &color, const unsigned int &srFlag);

   /**
   * @brief Get the diffuse color
   * 
   * @return FVector3 
   */
   CPPAPI_FUNC(FVector3) get_diffuse_color();

   /**
   * @brief Set the diffuse color
   * 
   * @param color 
   * @return HRESULT 
   */
   CPPAPI_FUNC(HRESULT) set_diffuse_color(const FVector3 &color);

   /**
   * @brief Set the diffuse color of a subrenderer
   * 
   * @param color 
   * @param srFlag 
   * @return HRESULT 
   */
   CPPAPI_FUNC(HRESULT) set_diffuse_color(const FVector3 &color, const unsigned int &srFlag);

   /**
   * @brief Get specular color
   * 
   * @return FVector3 
   */
   CPPAPI_FUNC(FVector3) get_specular_color();

   /**
   * @brief Set the specular color
   * 
   * @param color 
   * @return HRESULT 
   */
   CPPAPI_FUNC(HRESULT) set_specular_color(const FVector3 &color);

   /**
   * @brief Set the specular color of a subrenderer
   * 
   * @param color 
   * @param srFlag 
   * @return HRESULT 
   */
   CPPAPI_FUNC(HRESULT) set_specular_color(const FVector3 &color, const unsigned int &srFlag);

   /**
   * @brief Get the shininess
   * 
   * @return float 
   */
   CPPAPI_FUNC(float) get_shininess();

   /**
   * @brief Set the shininess
   * 
   * @param shininess 
   * @return HRESULT 
   */
   CPPAPI_FUNC(HRESULT) set_shininess(const float &shininess);

   /**
   * @brief Set the shininess of a subrenderer
   * 
   * @param shininess 
   * @param srFlag 
   * @return HRESULT 
   */
   CPPAPI_FUNC(HRESULT) set_shininess(const float &shininess, const unsigned int &srFlag);

   /**
   * @brief Get the grid color
   * 
   * @return FVector3 
   */
   CPPAPI_FUNC(FVector3) get_grid_color();

   /**
   * @brief Set the grid color
   * 
   * @param color 
   * @return HRESULT 
   */
   CPPAPI_FUNC(HRESULT) set_grid_color(const FVector3 &color);

   /**
   * @brief Get the scene box color
   * 
   * @return FVector3 
   */
   CPPAPI_FUNC(FVector3) get_scene_box_color();

   /**
   * @brief Set the scene box color
   * 
   * @param color 
   * @return HRESULT 
   */
   CPPAPI_FUNC(HRESULT) set_scene_box_color(const FVector3 &color);

   /**
   * @brief Get the light direction
   * 
   * @return FVector3 
   */
   CPPAPI_FUNC(FVector3) get_light_direction();

   /**
   * @brief Set the light direction
   * 
   * @param lightDir 
   * @return HRESULT 
   */
   CPPAPI_FUNC(HRESULT) set_light_direction(const FVector3& lightDir);

   /**
   * @brief Set the light direction of a subrenderer
   * 
   * @param lightDir 
   * @param srFlag 
   * @return HRESULT 
   */
   CPPAPI_FUNC(HRESULT) set_light_direction(const FVector3& lightDir, const unsigned int &srFlag);

   /**
   * @brief Get the light color
   * 
   * @return FVector3 
   */
   CPPAPI_FUNC(FVector3) get_light_color();

   /**
   * @brief Set the light color
   * 
   * @param color 
   * @return HRESULT 
   */
   CPPAPI_FUNC(HRESULT) set_light_color(const FVector3 &color);

   /**
   * @brief Set the light color of a subrenderer
   * 
   * @param color 
   * @param srFlag 
   * @return HRESULT 
   */
   CPPAPI_FUNC(HRESULT) set_light_color(const FVector3 &color, const unsigned int &srFlag);

   /**
   * @brief Get the background color
   * 
   * @return FVector3 
   */
   CPPAPI_FUNC(FVector3) get_background_color();

   /**
   * @brief Set the background color
   * 
   * @param color 
   * @return HRESULT 
   */
   CPPAPI_FUNC(HRESULT) set_background_color(const FVector3 &color);

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
   CPPAPI_FUNC(HRESULT) decorate_scene(const bool &decorate);

   /**
   * @brief Test whether discretization is current shown
   * 
   * @return true 
   * @return false 
   */
   CPPAPI_FUNC(bool) showing_discretization();

   /**
   * @brief Set flag to draw/not draw discretization
   * 
   * @param show flag; true says to show
   * @return HRESULT 
   */
   CPPAPI_FUNC(HRESULT) show_discretization(const bool &show);

   /**
   * @brief Get the current color of the discretization grid
   * 
   * @return FVector3 
   */
   CPPAPI_FUNC(FVector3) get_discretizationColor();

   /**
   * @brief Set the color of the discretization grid
   * 
   * @param color 
   * @return HRESULT 
   */
   CPPAPI_FUNC(HRESULT) set_discretizationColor(const FVector3 &color);

   /* Update screen size after the window has been resized */
   CPPAPI_FUNC(HRESULT) view_reshape(const iVector2 &windowSize);

   CPPAPI_FUNC(std::string) performance_counters();
   
   /**
   * @brief Get CPU info
   * 
   * @return std::unordered_map<std::string, bool> 
   */
   CPPAPI_FUNC(std::unordered_map<std::string, bool>) cpu_info();

   /**
   * @brief Get compiler flags of this installation
   * 
   * @return std::unordered_map<std::string, bool> 
   */
   CPPAPI_FUNC(std::unordered_map<std::string, bool>) compile_flags();

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

   CPPAPI_FUNC(PyObject*) image_data();

   /**
    * @brief Test whether Tissue Forge is running in an interactive terminal
    * 
    * @return true if running in an interactive terminal
    * @return false 
    */
   CPPAPI_FUNC(bool) is_terminal_interactive();

   /**
    * @brief Test whether Tissue Forge is running in a Jupyter notebook
    * 
    * @return true if running in a Jupyter notebook
    * @return false 
    */
   CPPAPI_FUNC(bool) is_jupyter_notebook();

   CPPAPI_FUNC(PyObject*) jwidget_init(PyObject *args, PyObject *kwargs);
   CPPAPI_FUNC(PyObject*) jwidget_run(PyObject *args, PyObject *kwargs);

   /**
    * @brief Get all available color mapper names
    */
   CPPAPI_FUNC(std::vector<std::string>) color_mapper_names();

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
   CPPAPI_FUNC(int) add_render_arrow(rendering::ArrowData *arrow);

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
   CPPAPI_FUNC(std::pair<int, rendering::ArrowData*>) add_render_arrow(
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
   CPPAPI_FUNC(HRESULT) remove_render_arrow(const int &arrowId);

   /**
    * @brief Gets a vector visualization specification. 
    * 
    * @param arrowId id of arrow according to the renderer
    */
   CPPAPI_FUNC(rendering::ArrowData*) get_render_arrow(const int &arrowId);

};

#endif // _SOURCE_LANGS_PY_TF_SYSTEMPY_H_