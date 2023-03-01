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

%{

#include <langs/py/tf_systemPy.h>

%}


%rename(_gl_info) TissueForge::py::gl_info();
%rename(_cpu_info) TissueForge::py::cpu_info();
%rename(_compile_flags) TissueForge::py::compile_flags;
%rename(_screenshot) TissueForge::py::screenshot(const std::string&);
%rename(_screenshot2) TissueForge::py::screenshot(const std::string&, const bool&, const FVector3&);

%rename(_system_print_performance_counters) TissueForge::py::print_performance_counters;
%rename(_system_context_has_current) TissueForge::py::context_has_current;
%rename(_system_context_make_current) TissueForge::py::context_make_current;
%rename(_system_context_release) TissueForge::py::context_release;
%rename(_system_camera_is_lagging) TissueForge::py::camera_is_lagging;
%rename(_system_camera_enable_lagging) TissueForge::py::camera_enable_lagging;
%rename(_system_camera_disable_lagging) TissueForge::py::camera_disable_lagging;
%rename(_system_camera_toggle_lagging) TissueForge::py::camera_toggle_lagging;
%rename(_system_camera_get_lagging) TissueForge::py::camera_get_lagging;
%rename(_system_camera_set_lagging) TissueForge::py::camera_set_lagging;
%rename(_system_camera_move_to) TissueForge::py::camera_move_to;
%rename(_system_camera_view_bottom) TissueForge::py::camera_view_bottom;
%rename(_system_camera_view_top) TissueForge::py::camera_view_top;
%rename(_system_camera_view_left) TissueForge::py::camera_view_left;
%rename(_system_camera_view_right) TissueForge::py::camera_view_right;
%rename(_system_camera_view_back) TissueForge::py::camera_view_back;
%rename(_system_camera_view_front) TissueForge::py::camera_view_front;
%rename(_system_camera_reset) TissueForge::py::camera_reset;
%rename(_system_camera_rotate_mouse) TissueForge::py::camera_rotate_mouse;
%rename(_system_camera_translate_mouse) TissueForge::py::camera_translate_mouse;
%rename(_system_camera_translate_down) TissueForge::py::camera_translate_down;
%rename(_system_camera_translate_up) TissueForge::py::camera_translate_up;
%rename(_system_camera_translate_right) TissueForge::py::camera_translate_right;
%rename(_system_camera_translate_left) TissueForge::py::camera_translate_left;
%rename(_system_camera_translate_forward) TissueForge::py::camera_translate_forward;
%rename(_system_camera_translate_backward) TissueForge::py::camera_translate_backward;
%rename(_system_camera_rotate_down) TissueForge::py::camera_rotate_down;
%rename(_system_camera_rotate_up) TissueForge::py::camera_rotate_up;
%rename(_system_camera_rotate_left) TissueForge::py::camera_rotate_left;
%rename(_system_camera_rotate_right) TissueForge::py::camera_rotate_right;
%rename(_system_camera_roll_left) TissueForge::py::camera_roll_left;
%rename(_system_camera_roll_right) TissueForge::py::camera_roll_right;
%rename(_system_camera_zoom_in) TissueForge::py::camera_zoom_in;
%rename(_system_camera_zoom_out) TissueForge::py::camera_zoom_out;
%rename(_system_camera_init_mouse) TissueForge::py::camera_init_mouse;
%rename(_system_camera_translate_by) TissueForge::py::camera_translate_by;
%rename(_system_camera_zoom_by) TissueForge::py::camera_zoom_by;
%rename(_system_camera_zoom_to) TissueForge::py::camera_zoom_to;
%rename(_system_camera_rotate_to_axis) TissueForge::py::camera_rotate_to_axis;
%rename(_system_camera_rotate_to_euler_angle) TissueForge::py::camera_rotate_to_euler_angle;
%rename(_system_camera_rotate_by_euler_angle) TissueForge::py::camera_rotate_by_euler_angle;
%rename(_system_camera_center) TissueForge::py::camera_center;
%rename(_system_camera_rotation) TissueForge::py::camera_rotation;
%rename(_system_camera_zoom) TissueForge::py::camera_zoom;
%rename(_system_get_renderer) TissueForge::py::get_renderer;
%rename(_system_get_rendering_3d_bonds) TissueForge::py::get_rendering_3d_bonds;
%rename(_system_set_rendering_3d_bonds) TissueForge::py::set_rendering_3d_bonds;
%rename(_system_toggle_rendering_3d_bonds) TissueForge::py::toggle_rendering_3d_bonds;
%rename(_system_get_rendering_3d_angles) TissueForge::py::get_rendering_3d_angles;
%rename(_system_set_rendering_3d_angles) TissueForge::py::set_rendering_3d_angles;
%rename(_system_toggle_rendering_3d_angles) TissueForge::py::toggle_rendering_3d_angles;
%rename(_system_get_rendering_3d_dihedrals) TissueForge::py::get_rendering_3d_dihedrals;
%rename(_system_set_rendering_3d_dihedrals) TissueForge::py::set_rendering_3d_dihedrals;
%rename(_system_toggle_rendering_3d_dihedrals) TissueForge::py::toggle_rendering_3d_dihedrals;
%rename(_system_set_rendering_3d_all) TissueForge::py::set_rendering_3d_all;
%rename(_system_toggle_rendering_3d_all) TissueForge::py::toggle_rendering_3d_all;
%rename(_system_get_line_width) TissueForge::py::get_line_width;
%rename(_system_set_line_width) TissueForge::py::set_line_width;
%rename(_system_get_line_width_min) TissueForge::py::get_line_width_min;
%rename(_system_get_line_width_max) TissueForge::py::get_line_width_max;
%rename(_system_get_ambient_color) TissueForge::py::get_ambient_color;
%rename(_system_set_ambient_color) TissueForge::py::set_ambient_color;
%rename(_system_get_diffuse_color) TissueForge::py::get_diffuse_color;
%rename(_system_set_diffuse_color) TissueForge::py::set_diffuse_color;
%rename(_system_get_specular_color) TissueForge::py::get_specular_color;
%rename(_system_set_specular_color) TissueForge::py::set_specular_color;
%rename(_system_get_shininess) TissueForge::py::get_shininess;
%rename(_system_set_shininess) TissueForge::py::set_shininess;
%rename(_system_get_grid_color) TissueForge::py::get_grid_color;
%rename(_system_set_grid_color) TissueForge::py::set_grid_color;
%rename(_system_get_scene_box_color) TissueForge::py::get_scene_box_color;
%rename(_system_set_scene_box_color) TissueForge::py::set_scene_box_color;
%rename(_system_get_light_direction) TissueForge::py::get_light_direction;
%rename(_system_set_light_direction) TissueForge::py::set_light_direction;
%rename(_system_get_light_color) TissueForge::py::get_light_color;
%rename(_system_set_light_color) TissueForge::py::set_light_color;
%rename(_system_get_background_color) TissueForge::py::get_background_color;
%rename(_system_set_background_color) TissueForge::py::set_background_color;
%rename(_system_decorated) TissueForge::py::decorated;
%rename(_system_decorate_scene) TissueForge::py::decorate_scene;
%rename(_system_showing_discretization) TissueForge::py::showing_discretization;
%rename(_system_show_discretization) TissueForge::py::show_discretization;
%rename(_system_get_discretizationColor) TissueForge::py::get_discretizationColor;
%rename(_system_set_discretizationColor) TissueForge::py::set_discretizationColor;
%rename(_system_view_reshape) TissueForge::py::view_reshape;
%rename(_system_performance_counters) TissueForge::py::performance_counters;
%rename(_system_egl_info) TissueForge::py::egl_info;
%rename(_system_image_data) TissueForge::py::image_data;
%rename(_system_is_terminal_interactive) TissueForge::py::is_terminal_interactive;
%rename(_system_is_jupyter_notebook) TissueForge::py::is_jupyter_notebook;
%rename(_system_jwidget_init) TissueForge::py::jwidget_init;
%rename(_system_jwidget_run) TissueForge::py::jwidget_run;
%rename(_system_color_mapper_names) TissueForge::py::color_mapper_names;
%rename(_system_add_render_arrow) TissueForge::py::add_render_arrow;
%rename(_system_remove_render_arrow) TissueForge::py::remove_render_arrow;
%rename(_system_get_render_arrow) TissueForge::py::get_render_arrow;


%include <langs/py/tf_systemPy.h>


%pythoncode %{

    def _system_cpu_info() -> dict:
        """Dictionary of CPU info"""
        return _cpu_info().asdict()

    def _system_compile_flags() -> dict:
        """Dictionary of CPU info"""
        return _compile_flags().asdict()

    def _system_gl_info() -> dict:
        """Dictionary of OpenGL info"""
        return _gl_info().asdict()

    def _system_screenshot(filepath: str, decorate: bool = None, bgcolor=None):
        """
        Save a screenshot of the current scene

        :param filepath: path of file to save
        :type filepath: str
        :param decorate: flag to decorate the scene in the screenshot
        :type decorate: bool
        :param bgcolor: background color of the scene in the screenshot
        :type bgcolor: FVector3 or [float, float, float] or (float, float, float) or float
        :rtype: int
        :return: HRESULT
        """    
        if decorate is None and bgcolor is None:
            return _screenshot(filepath)

        if decorate is None:
            decorate = _system_decorated()
        if bgcolor is None:
            bgcolor = _system_get_background_color()
        elif not isinstance(bgcolor, FVector3):
            bgcolor = FVector3(bgcolor)
        return _screenshot2(filepath, decorate, bgcolor)

%}
