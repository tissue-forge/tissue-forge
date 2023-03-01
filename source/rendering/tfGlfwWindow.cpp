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

#include "tfGlfwApplication.h"
#include "tfGlfwWindow.h"
#include <Magnum/GL/DefaultFramebuffer.h>

#include <iostream>


using namespace Magnum;
using namespace TissueForge;


const float &rendering::GlfwWindow::getFloatField()
{
    return this->f;
}

void rendering::GlfwWindow::setFloatField(const float &value)
{
    this->f = value;
}

rendering::GlfwWindow::GlfwWindow(GLFWwindow *win)
{
    _window = win;
}

TissueForge::iVector2 rendering::GlfwWindow::windowSize() const {
    CORRADE_ASSERT(_window, "Platform::GlfwApplication::windowSize(): no window opened", {});

    Vector2i size;
    glfwGetWindowSize(_window, &size.x(), &size.y());
    return size;
}

void rendering::GlfwWindow::redraw() {

    // TODO: get rid of GLFWApplication
    Simulator::get()->redraw();
}

Magnum::GL::AbstractFramebuffer &rendering::GlfwWindow::framebuffer() {
    return Magnum::GL::defaultFramebuffer;
}

void rendering::GlfwWindow::setTitle(const char* title) {
    
}

