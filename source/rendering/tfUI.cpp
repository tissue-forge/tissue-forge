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

#include "tfUI.h"
#include <tfLogger.h>
#include <tfError.h>

#include <GLFW/glfw3.h>
#include <Magnum/GL/GL.h>
#include <Magnum/GL/Version.h>
#include <iostream>


using namespace TissueForge;


HRESULT rendering::pollEvents()
{
    glfwPollEvents();
    return S_OK;
}

HRESULT rendering::waitEvents(double timeout)
{
    glfwWaitEventsTimeout(timeout);
    glfwWaitEvents();

    return S_OK;
}

HRESULT rendering::postEmptyEvent()
{
    glfwPostEmptyEvent();
    return S_OK;
}

static void error_callback(int error, const char* description)
{
    tf_error(error, description);
}


HRESULT rendering::initializeGraphics()
{
    TF_Log(LOG_TRACE);

    glfwSetErrorCallback(error_callback);

    if (!glfwInit()) {
        return E_FAIL;
    }


    glfwWindowHint(GLFW_SAMPLES, 4); // 4x antialiasing
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4); // We want OpenGL 3.3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // To make MacOS happy; should not be needed
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); // We don't want the old OpenGL

    return S_OK;
}
