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

#ifndef _SOURCE_RENDERING_TFGLFWAPPLICATION_H_
#define _SOURCE_RENDERING_TFGLFWAPPLICATION_H_

#include <TissueForge_private.h>
#include "tfApplication.h"
#include <tfSimulator.h>
#include <Magnum/Platform/GlfwApplication.h>

#include "tfGlfwWindow.h"
#include "tfUniverseRenderer.h"


using namespace Magnum;


namespace TissueForge::rendering {


    class GlfwApplication :
            public Application,
            public Magnum::Platform::GlfwApplication {

    public:

        typedef Magnum::Platform::GlfwApplication::Arguments Arguments;



        /**
         * creates the app, but does not create the context.
         */
        GlfwApplication(const Arguments &args);


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

        HRESULT waitEventsTimeout(double timeout) override;


        /**
         * This function posts an empty event from the current thread
         * to the event queue, causing waitEvents or waitEventsTimeout to return.
         */
        HRESULT postEmptyEvent() override;


        HRESULT setSwapInterval(int si) override;

        void drawEvent() override;

        GlfwWindow *getWindow() override;

        UniverseRenderer *getRenderer() override;

        HRESULT redraw() override;

        Magnum::GL::AbstractFramebuffer& framebuffer() override;

        GlfwWindow *_win;

        // TODO implement events and move these to simulator.
        UniverseRenderer *_ren;

        Int _substeps = 1;
        bool _pausedSimulation = false;
        bool _mousePressed = false;

        /* Timeline to adjust number of simulation steps per frame */
        Timeline _timeline;

        HRESULT mainLoopIteration(double timeout) override;


        void viewportEvent(ViewportEvent& event) override;
        void keyPressEvent(KeyEvent& event) override;
        void mousePressEvent(MouseEvent& event) override;
        void mouseReleaseEvent(MouseEvent& event) override;
        void mouseMoveEvent(MouseMoveEvent& event) override;
        void mouseScrollEvent(MouseScrollEvent& event) override;
        void exitEvent(ExitEvent& event) override;

        int windowAttribute(WindowAttributes attr) override;

        HRESULT setWindowAttribute(WindowAttributes attr, int val) override;

        HRESULT destroy() override;

        HRESULT close() override;

        HRESULT show() override;
                
        HRESULT messageLoop(double et) override;
                
        HRESULT showWindow();
                
        bool contextMakeCurrent() override;
        
        bool contextHasCurrent() override;
        
        bool contextRelease() override;

    private:

        Magnum::GL::Context* _context;
    };

}

#endif /* _SOURCE_RENDERING_TFGLFWAPPLICATION_H_ */