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

#include "tfWindowlessApplication.h"
#include "tfWindow.h"
#include "tfUniverseRenderer.h"

#include <Magnum/GL/DefaultFramebuffer.h>
#include <Magnum/GL/Renderer.h>
#include <Magnum/GL/PixelFormat.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Image.h>
#include <Magnum/GL/Context.h>
#include <Magnum/GL/Version.h>
#include <Magnum/Platform/GLContext.h>

#include "tfGlInfo.h"
#include <tfLogger.h>
#include <tfError.h>
#include <iostream>

#include "access_private.hpp"


ACCESS_PRIVATE_FIELD(Magnum::Platform::WindowlessApplication, Magnum::Platform::WindowlessGLContext, _glContext);


using namespace TissueForge;


struct rendering::WindowlessWindow : rendering::Window
{
    rendering::WindowlessApplication *app;

    TissueForge::iVector2 windowSize() const override {
        return app->framebuffer().viewport().size();
    };

    void redraw() override {
        app->redraw();
    }

    Magnum::GL::AbstractFramebuffer &framebuffer() override {
        return app->frameBuffer;
    }

    /**
     * attach to an existing GLFW Window
     */
    WindowlessWindow(rendering::WindowlessApplication *_app) : app{_app} {
    };
};

rendering::WindowlessApplication::~WindowlessApplication()
{
    TF_Log(LOG_TRACE);
}

rendering::WindowlessApplication::WindowlessApplication(const Arguments &args) :
    Magnum::Platform::WindowlessApplication{args, Magnum::NoCreate},
    renderBuffer{Magnum::NoCreate},
    frameBuffer{Magnum::NoCreate},
    depthStencil{Magnum::NoCreate}
{
}

HRESULT rendering::WindowlessApplication::createContext(const Simulator::Config &conf) {

    // default Magnum WindowlessApplication config, does not have any options
    Configuration windowlessConf;

    TF_Log(LOG_INFORMATION) << "trying to create windowless context";

    if(!Magnum::Platform::WindowlessApplication::tryCreateContext(windowlessConf)) {
        return tf_error(E_FAIL, "could not create windowless context");
    }
    
    Magnum::Platform::WindowlessApplication &app = *this;
    Magnum::Platform::WindowlessGLContext &glContext = access_private::_glContext(app);
    _context = &GL::Context::current();
    
    
#if defined(TF_APPLE)
    const char* cname = "CGL Context";
#elif defined(TF_LINUX)
    const char* cname = "EGL Context";
#elif defined(TF_WINDOWS)
    const char* cname = "WGL Context";
#else
#error "NO GL Supported"
#endif

    TF_Log(LOG_INFORMATION) << "created windowless context, " << cname << glContext.glContext();
    TF_Log(LOG_INFORMATION) << "GL Info: " << gl_info();
    
    Vector2i size = conf.windowSize();

    // create the render buffers here, after we have a context,
    // default ctor makes this with a {Magnum::NoCreate},
    renderBuffer = Magnum::GL::Renderbuffer();
    depthStencil = Magnum::GL::Renderbuffer();

    depthStencil.setStorage(Magnum::GL::RenderbufferFormat::Depth24Stencil8, size);

    renderBuffer.setStorage(Magnum::GL::RenderbufferFormat::RGBA8, size);

    frameBuffer = Magnum::GL::Framebuffer{{{0,0}, size}};

    frameBuffer
        .attachRenderbuffer(Magnum::GL::Framebuffer::ColorAttachment{0}, renderBuffer)
        .attachRenderbuffer(Magnum::GL::Framebuffer::BufferAttachment::DepthStencil, depthStencil)
        .clear(Magnum::GL::FramebufferClear::Color)
        .bind();

    window = new WindowlessWindow(this);

    // renderer accesses the framebuffer from the window handle we pass in.
    renderer = new rendering::UniverseRenderer(conf, window);

    return S_OK;
}


rendering::UniverseRenderer *rendering::WindowlessApplication::getRenderer() {
    return renderer;
}

HRESULT rendering::WindowlessApplication::WindowlessApplication::pollEvents()
{
    return E_NOTIMPL;
}

HRESULT rendering::WindowlessApplication::WindowlessApplication::waitEvents()
{
    return E_NOTIMPL;
}

HRESULT rendering::WindowlessApplication::WindowlessApplication::waitEventsTimeout(double timeout)
{
    return E_NOTIMPL;
}

HRESULT rendering::WindowlessApplication::WindowlessApplication::postEmptyEvent()
{
    return E_NOTIMPL;
}

HRESULT rendering::WindowlessApplication::mainLoopIteration(double timeout)
{
    return E_NOTIMPL;
}

struct rendering::GlfwWindow* rendering::WindowlessApplication::getWindow()
{
    return NULL;
}

int rendering::WindowlessApplication::windowAttribute(rendering::WindowAttributes attr)
{
    return E_NOTIMPL;
}

HRESULT rendering::WindowlessApplication::setWindowAttribute(rendering::WindowAttributes attr,
        int val)
{
    return E_NOTIMPL;
}

HRESULT rendering::WindowlessApplication::redraw()
{
    TF_Log(LOG_TRACE);
    
    // TODO: need to re-evaluate drawing, should not have to check...
    // drawing code on the wrong thread should not call re-draw, only the renderer should
    
    if(Magnum::GL::Context::hasCurrent()) {
        frameBuffer
            .clear(Magnum::GL::FramebufferClear::Color)
            .bind();

        /* Draw particles */
        renderer->draw();
    }

    return S_OK;
}

HRESULT rendering::WindowlessApplication::close()
{
    return E_NOTIMPL;
}

HRESULT rendering::WindowlessApplication::destroy()
{
    return E_NOTIMPL;
}

HRESULT rendering::WindowlessApplication::show()
{
    return redraw();
}

HRESULT rendering::WindowlessApplication::messageLoop(double et)
{
    return E_NOTIMPL;
}

Magnum::GL::AbstractFramebuffer& rendering::WindowlessApplication::framebuffer() {
    return frameBuffer;
}

bool rendering::WindowlessApplication::contextMakeCurrent()
{
    TF_Log(LOG_TRACE);
    
    Magnum::Platform::WindowlessApplication &app = *this;

    Magnum::Platform::WindowlessGLContext &glContext = access_private::_glContext(app);

    if(glContext.makeCurrent()) {
        Magnum::GL::Context::makeCurrent(_context);
        return true;
    }

    return false;
}

bool rendering::WindowlessApplication::contextHasCurrent()
{
    TF_Log(LOG_TRACE);
    
    return Magnum::GL::Context::hasCurrent();
}

bool rendering::WindowlessApplication::contextRelease()
{
    TF_Log(LOG_TRACE);
    
    Magnum::Platform::WindowlessApplication &app = *this;

    Magnum::Platform::WindowlessGLContext &context = access_private::_glContext(app);

    Magnum::GL::Context::makeCurrent(nullptr);

    return context.release();
}
