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

#ifndef _SOURCE_RENDERING_TFAPPLICATION_H_
#define _SOURCE_RENDERING_TFAPPLICATION_H_

#include <TissueForge.h>
#include <TissueForge_private.h>
#include <tfSimulator.h>
#include <Magnum/GL/Context.h>
#include <GLFW/glfw3.h>
#include "tfImageConverters.h"


namespace TissueForge::rendering {


    enum WindowAttributes {
        TF_FOCUSED = GLFW_FOCUSED,
        TF_ICONIFIED = GLFW_ICONIFIED,
        TF_RESIZABLE = GLFW_RESIZABLE,
        TF_VISIBLE = GLFW_VISIBLE,
        TF_DECORATED = GLFW_DECORATED,
        TF_AUTO_ICONIFY = GLFW_AUTO_ICONIFY,
        TF_FLOATING = GLFW_FLOATING,
        TF_MAXIMIZED = GLFW_MAXIMIZED,
        TF_CENTER_CURSOR = GLFW_CENTER_CURSOR,
        TF_TRANSPARENT_FRAMEBUFFER = GLFW_TRANSPARENT_FRAMEBUFFER,
        TF_HOVERED = GLFW_HOVERED,
        TF_FOCUS_ON_SHOW = GLFW_FOCUS_ON_SHOW,
        TF_RED_BITS = GLFW_RED_BITS,
        TF_GREEN_BITS = GLFW_GREEN_BITS,
        TF_BLUE_BITS = GLFW_BLUE_BITS,
        TF_ALPHA_BITS = GLFW_ALPHA_BITS,
        TF_DEPTH_BITS = GLFW_DEPTH_BITS,
        TF_STENCIL_BITS = GLFW_STENCIL_BITS,
        TF_ACCUM_RED_BITS = GLFW_ACCUM_RED_BITS,
        TF_ACCUM_GREEN_BITS = GLFW_ACCUM_GREEN_BITS,
        TF_ACCUM_BLUE_BITS = GLFW_ACCUM_BLUE_BITS,
        TF_ACCUM_ALPHA_BITS = GLFW_ACCUM_ALPHA_BITS,
        TF_AUX_BUFFERS = GLFW_AUX_BUFFERS,
        TF_STEREO = GLFW_STEREO,
        TF_SAMPLES = GLFW_SAMPLES,
        TF_SRGB_CAPABLE = GLFW_SRGB_CAPABLE,
        TF_REFRESH_RATE = GLFW_REFRESH_RATE,
        TF_DOUBLEBUFFER = GLFW_DOUBLEBUFFER,
        TF_CLIENT_API = GLFW_CLIENT_API,
        TF_CONTEXT_VERSION_MAJOR = GLFW_CONTEXT_VERSION_MAJOR,
        TF_CONTEXT_VERSION_MINOR = GLFW_CONTEXT_VERSION_MINOR,
        TF_CONTEXT_REVISION = GLFW_CONTEXT_REVISION,
        TF_CONTEXT_ROBUSTNESS = GLFW_CONTEXT_ROBUSTNESS,
        TF_OPENGL_FORWARD_COMPAT = GLFW_OPENGL_FORWARD_COMPAT,
        TF_OPENGL_DEBUG_CONTEXT = GLFW_OPENGL_DEBUG_CONTEXT,
        TF_OPENGL_PROFILE = GLFW_OPENGL_PROFILE,
        TF_CONTEXT_RELEASE_BEHAVIOR = GLFW_CONTEXT_RELEASE_BEHAVIOR,
        TF_CONTEXT_NO_ERROR = GLFW_CONTEXT_NO_ERROR,
        TF_CONTEXT_CREATION_API = GLFW_CONTEXT_CREATION_API,
        TF_SCALE_TO_MONITOR = GLFW_SCALE_TO_MONITOR,
        TF_COCOA_RETINA_FRAMEBUFFER = GLFW_COCOA_RETINA_FRAMEBUFFER,
        TF_COCOA_FRAME_NAME = GLFW_COCOA_FRAME_NAME,
        TF_COCOA_GRAPHICS_SWITCHING = GLFW_COCOA_GRAPHICS_SWITCHING,
        TF_X11_CLASS_NAME = GLFW_X11_CLASS_NAME,
        TF_X11_INSTANCE_NAME = GLFW_X11_INSTANCE_NAME
    };


    /**
     * Set config options for opengl for now.
     */
    struct ApplicationConfig {
    public:
        /**
         * @brief Window flag
         *
         * @see @ref WindowFlags, @ref setWindowFlags()
         */
        enum WindowFlag  {
            None = 0,
            Fullscreen = 1 << 0,   /**< Fullscreen window */
            Resizable = 1 << 1,    /**< Resizable window */
            Hidden = 1 << 2,       /**< Hidden window */


            Maximized = 1 << 3,


            Minimized = 1 << 4,    /**< Minimized window */
            Floating = 1 << 5,     /**< Window floating above others, top-most */

            /**
             * Automatically iconify (minimize) if fullscreen window loses
             * input focus
             */
            AutoIconify = 1 << 6,

            Focused = 1 << 7,      /**< Window has input focus */


            /**
             * Do not create any GPU context. Use together with
             * @ref GlfwApplication(const Arguments&),
             * @ref GlfwApplication(const Arguments&, const Configuration&),
             * @ref create(const Configuration&) or
             * @ref tryCreate(const Configuration&) to prevent implicit
             * creation of an OpenGL context.
             *
             * @note Supported since GLFW 3.2.
             */
            Contextless = 1 << 8
        };

        unsigned windowFlag = 0;


        /**
         * @brief DPI scaling policy
         *
         * DPI scaling policy when requesting a particular window size. Can
         * be overriden on command-line using `--magnum-dpi-scaling` or via
         * the `MAGNUM_DPI_SCALING` environment variable.
         * @see @ref setSize(), @ref Platform-Sdl2Application-dpi
         */
        enum class DpiScalingPolicy {
            /**
             * Framebuffer DPI scaling. The window will have the same size as
             * requested, but the framebuffer size will be different. Supported
             * only on macOS and iOS and is also the only supported value
             * there.
             */
            Framebuffer,

            /**
             * Virtual DPI scaling. Scales the window based on UI scaling
             * setting in the system. Falls back to
             * @ref DpiScalingPolicy::Physical on platforms that don't support
             * it. Supported only on desktop platforms (except macOS) and it's
             * the default there.
             *
             * Equivalent to `--magnum-dpi-scaling virtual` passed on
             * command-line.
             */
            Virtual,

            /**
             * Physical DPI scaling. Takes the requested window size as a
             * physical size that a window would have on platform's default DPI
             * and scales it to have the same size on given display physical
             * DPI. On platforms that don't have a concept of a window it
             * causes the framebuffer to match screen pixels 1:1 without any
             * scaling. Supported on desktop platforms except macOS and on
             * mobile and web. Default on mobile and web.
             *
             * Equivalent to `--magnum-dpi-scaling physical` passed on
             * command-line.
             */
            Physical,

            /**
             * Default policy for current platform. Alias to one of
             * @ref DpiScalingPolicy::Framebuffer, @ref DpiScalingPolicy::Virtual
             * or @ref DpiScalingPolicy::Physical depending on platform. See
             * @ref Platform-Sdl2Application-dpi for details.
             */
            Default
        };
    };

    struct UniverseRenderer;
    struct GlfwWindow;


    struct CAPI_EXPORT Application
    {
    public:

        /**
         * list of windows.
         */
        std::vector<GlfwWindow*> windows;

        virtual ~Application() {};
        
        //virtual HRESULT createContext(const class Simulator::Config &conf) = 0;


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
        virtual HRESULT pollEvents () = 0;

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
        virtual HRESULT waitEvents () = 0;

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

        virtual HRESULT waitEventsTimeout(double  timeout) = 0;


        /**
         * This function posts an empty event from the current thread
         * to the event queue, causing waitEvents or waitEventsTimeout to return.
         */
        virtual HRESULT postEmptyEvent() = 0;


        virtual HRESULT mainLoopIteration(double timeout)= 0;


        virtual HRESULT setSwapInterval(int si) = 0;


        // temporary hack until we setup events correctly
        virtual struct GlfwWindow *getWindow() = 0;

        virtual int windowAttribute(WindowAttributes attr) = 0;

        virtual HRESULT setWindowAttribute(WindowAttributes attr, int val) = 0;
        
        virtual HRESULT createContext(const Simulator::Config &conf) = 0;

        virtual UniverseRenderer *getRenderer() = 0;
        
        /**
         * post a re-draw event, to tell the renderer
         * that it should re-draw
         */
        virtual HRESULT redraw() = 0;

        virtual HRESULT run(double et);


        // soft hide the window
        virtual HRESULT close() = 0;

        // hard window close
        virtual HRESULT destroy() = 0;

        // display the window if closed.
        virtual HRESULT show() = 0;
        
        virtual HRESULT simulationStep();
        
        virtual HRESULT messageLoop(double et) = 0;
        
        virtual Magnum::GL::AbstractFramebuffer& framebuffer() = 0;
        
        
        virtual bool contextMakeCurrent() = 0;
        
        virtual bool contextHasCurrent() = 0;
        
        virtual bool contextRelease() = 0;
        
        
    protected:
        bool _dynamicBoundary = true;
        
        
        Magnum::Float _boundaryOffset = 0.0f; /* For boundary animation */
        
        int currentStep = 0;


    };

    Corrade::Containers::Array<char> JpegImageData();
    Corrade::Containers::Array<char> BMPImageData();
    Corrade::Containers::Array<char> HDRImageData();
    Corrade::Containers::Array<char> PNGImageData();
    Corrade::Containers::Array<char> TGAImageData();

    std::tuple<char*, size_t> framebufferImageData();

    /**
     * @brief Save a screenshot of the current scene
     * 
     * @param filePath path of file to save
     * @return HRESULT 
     */
    HRESULT screenshot(const std::string &filePath);

};

#endif // _SOURCE_RENDERING_TFAPPLICATION_H_
