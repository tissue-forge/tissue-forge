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

#ifndef _SOURCE_RENDERING_TFGLFWWINDOW_H_
#define _SOURCE_RENDERING_TFGLFWWINDOW_H_

#include <TissueForge_private.h>

#include "tfWindow.h"

#include <Magnum/Magnum.h>
#include <GLFW/glfw3.h>


namespace TissueForge::rendering {


    /**
     * The GLFWWindow provides a glue to connect generate Tissue Forge events from glfw events.
     */
    struct CAPI_EXPORT GlfwWindow : Window
    {
        /**
         * attach to an existing GLFW Window
         */
        GlfwWindow(GLFWwindow *win);

        // it's a wrapper around a native GLFW window
        GLFWwindow* _window;

        float f;

        iVector2 windowSize() const override;

        void redraw() override;
        
        void setTitle(const char* title);
        
        Magnum::GL::AbstractFramebuffer &framebuffer() override;

        const float &getFloatField();
        void setFloatField(const float &value);
        
    };

}

#endif // _SOURCE_RENDERING_TFGLFWWINDOW_H_