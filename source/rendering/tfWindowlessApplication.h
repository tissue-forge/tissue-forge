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

#ifndef _SOURCE_RENDERING_TFWINDOWLESSAPPLICATION_H_
#define _SOURCE_RENDERING_TFWINDOWLESSAPPLICATION_H_

#include <tf_config.h>
#include <TissueForge.h>
#include "tfApplication.h"
#include <Magnum/GL/Context.h>

#include <Magnum/GL/Buffer.h>
#include <Magnum/GL/Framebuffer.h>
#include <Magnum/GL/Mesh.h>
#include <Magnum/GL/Renderbuffer.h>
#include <Magnum/GL/RenderbufferFormat.h>


#if defined(TF_APPLE)
    #include "Magnum/Platform/WindowlessCglApplication.h"
#elif defined(TF_LINUX)
    #include "Magnum/Platform/WindowlessEglApplication.h"

    // Freaking xlib.h defines these, and the wreak havoc with everythign... 
    #undef Button
    #undef Button1
    #undef Button2
    #undef Button3
    #undef Button4
    #undef Button5

#elif defined(TF_WINDOWS)
#include "Magnum/Platform/WindowlessWglApplication.h"
#else
#error no windowless application available on this platform
#endif


namespace TissueForge {


    namespace rendering {


        struct WindowlessApplication :
                public Application,
                private Magnum::Platform::WindowlessApplication
        {
        public:

            typedef Magnum::Platform::WindowlessApplication::Arguments Arguments;

            typedef Magnum::Platform::WindowlessApplication::Configuration Configuration;

            WindowlessApplication() = delete;

            /**
             * Set up the app, but don't create a context just yet.
             */
            WindowlessApplication(const Arguments &args);

            ~WindowlessApplication();
            
            UniverseRenderer *getRenderer() override;
            
            
            HRESULT createContext(const Simulator::Config &conf) override;

            /**
             * This function processes only those events that are already in the event
             * queue and then returns immediately. Processing events will cause the window
             * and input callbacks associated with those events to be called.
             *
             * On some platforms, a window move, resize or menu operation will cause
             * event processing to block. This is due to how event processing is designed
             * on those platforms. You can use the window refresh callback to redraw the
             * contents of your window when necessary during such operations.
             */
            HRESULT pollEvents () override;

            /**
             *   This function puts the calling thread to sleep until at least one
             *   event is available in the event queue. Once one or more events are
             *   available, it behaves exactly like glfwPollEvents, i.e. the events
             *   in the queue are processed and the function then returns immediately.
             *   Processing events will cause the window and input callbacks associated
             *   with those events to be called.
             *
             *   Since not all events are associated with callbacks, this function may return
             *   without a callback having been called even if you are monitoring all callbacks.
             *
             *  On some platforms, a window move, resize or menu operation will cause event
             *  processing to block. This is due to how event processing is designed on
             *  those platforms. You can use the window refresh callback to redraw the
             *  contents of your window when necessary during such operations.
             */
            HRESULT waitEvents () override;

            /**
             * This function puts the calling thread to sleep until at least
             * one event is available in the event queue, or until the specified
             * timeout is reached. If one or more events are available, it behaves
             * exactly like pollEvents, i.e. the events in the queue are
             * processed and the function then returns immediately. Processing
             * events will cause the window and input callbacks associated with those
             * events to be called.
             *
             * The timeout value must be a positive finite number.
             * Since not all events are associated with callbacks, this function may
             * return without a callback having been called even if you are monitoring
             * all callbacks.
             *
             * On some platforms, a window move, resize or menu operation will cause
             * event processing to block. This is due to how event processing is designed
             * on those platforms. You can use the window refresh callback to redraw the
             * contents of your window when necessary during such operations.
             */

            HRESULT waitEventsTimeout(double  timeout) override;


            /**
             * This function posts an empty event from the current thread
             * to the event queue, causing waitEvents or waitEventsTimeout to return.
             */
            HRESULT postEmptyEvent() override;

            HRESULT setSwapInterval(int si) override { return E_NOTIMPL;};
            
            HRESULT mainLoopIteration(double timeout) override;
            
            struct GlfwWindow *getWindow() override;
            
            int windowAttribute(WindowAttributes attr) override;
            
            HRESULT setWindowAttribute(WindowAttributes attr, int val) override;
            
            HRESULT redraw() override;
            
            HRESULT close() override;
            
            HRESULT destroy() override;
            
            HRESULT show() override;
            
            HRESULT messageLoop(double et) override;
            
            Magnum::GL::AbstractFramebuffer& framebuffer() override;
            
            
            bool contextMakeCurrent() override;
            
            bool contextHasCurrent() override;
            
            bool contextRelease() override;
            

        private:
            virtual int exec() override { return 0; };
            
            // the drawing buffer
            Magnum::GL::Renderbuffer renderBuffer;
            
            // attach render / stencil buffer to the framebuffer
            Magnum::GL::Renderbuffer depthStencil;
            
            Magnum::GL::Framebuffer frameBuffer;
            
            
            struct WindowlessWindow *window;
            struct UniverseRenderer *renderer;
            
            friend struct WindowlessWindow;

        };

}}

#endif // _SOURCE_RENDERING_TFWINDOWLESSAPPLICATION_H_