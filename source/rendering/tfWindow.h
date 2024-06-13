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

#ifndef _SOURCE_RENDERING_TFWINDOW_H_
#define _SOURCE_RENDERING_TFWINDOW_H_

#include <TissueForge_private.h>
#include <Magnum/Magnum.h>
#include <Magnum/GL/AbstractFramebuffer.h>
#include <GLFW/glfw3.h>


namespace TissueForge::rendering {


    struct Window
    {
        
        enum MouseButton {
            MouseButton1 = GLFW_MOUSE_BUTTON_1,
            MouseButton2 = GLFW_MOUSE_BUTTON_2,
            MouseButton3 = GLFW_MOUSE_BUTTON_3,
            MouseButton4 = GLFW_MOUSE_BUTTON_4,
            MouseButton5 = GLFW_MOUSE_BUTTON_5,
            MouseButton6 = GLFW_MOUSE_BUTTON_6,
            MouseButton7 = GLFW_MOUSE_BUTTON_7,
            MouseButton8 = GLFW_MOUSE_BUTTON_8,
            MouseButtonLast = GLFW_MOUSE_BUTTON_LAST,
            MouseButtonLeft = GLFW_MOUSE_BUTTON_LEFT,
            MouseButtonRight = GLFW_MOUSE_BUTTON_RIGHT,
            MouseButtonMiddle = GLFW_MOUSE_BUTTON_MIDDLE,
        };
        
        enum State {
            Release = GLFW_RELEASE,
            Press = GLFW_PRESS,
            Repeat = GLFW_REPEAT
        };

        virtual iVector2 windowSize() const = 0;
        
        virtual Magnum::GL::AbstractFramebuffer& framebuffer() = 0;
        
        virtual void redraw() = 0;
    };

}

#endif // _SOURCE_RENDERING_TFWINDOW_H_