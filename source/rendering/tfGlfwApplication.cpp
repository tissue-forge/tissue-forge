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

#include "tfGlfwApplication.h"
#include "tfKeyEvent.h"
#include "tfUniverseRenderer.h"

#include <cstring>
#include <tuple>
#include <Corrade/Containers/Array.h>
#include <Corrade/Containers/StridedArrayView.h>
#include <Corrade/Utility/Arguments.h>
#include <Corrade/Utility/String.h>
#include <Corrade/Utility/Unicode.h>

#include "Magnum/ImageView.h"
#include "Magnum/PixelFormat.h"
#include "Magnum/Math/ConfigurationValue.h"
#include "Magnum/Platform/ScreenedApplication.hpp"
#include "Magnum/Platform/Implementation/DpiScaling.h"

#include <Magnum/GL/DefaultFramebuffer.h>
#include <Magnum/GL/Renderer.h>
#include <Magnum/GL/PixelFormat.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Image.h>
#include <Magnum/GL/Context.h>
#include <Magnum/GL/Version.h>
#include <Magnum/Platform/GLContext.h>

#include <Magnum/SceneGraph/Scene.h>
#include <Magnum/SceneGraph/Camera.h>
#include <Magnum/SceneGraph/Drawable.h>

#include <tfLogger.h>
#include <tfError.h>

#if defined(_WIN32)
  #define GLFW_EXPOSE_NATIVE_WIN32
  #include <GLFW/glfw3native.h>
#endif


using namespace TissueForge;


#define TF_GLFW_ERROR() { \
        const char* glfwErrorDesc = NULL; \
        glfwGetError(&glfwErrorDesc); \
        tf_exp(std::domain_error(std::string("GLFW Error in ") + TF_FUNCTION + ": " +  glfwErrorDesc)); \
}

#define TF_GLFW_CHECK() { \
        const char* glfwErrorDesc = NULL; \
        int ret = glfwGetError(&glfwErrorDesc); \
        return ret == 0 ? S_OK : tf_error(ret, glfwErrorDesc); \
}


static Platform::GlfwApplication::Configuration confconf(const Simulator::Config &conf) {
    Platform::GlfwApplication::Configuration res;

    res.setSize(conf.windowSize(), conf.dpiScaling());
    res.setTitle(conf.title());
    res.setWindowFlags(Platform::GlfwApplication::Configuration::WindowFlag::Resizable);

    return res;
}

/**
 * checks if Glfw can be initialized, throws an exception.
 *
 * This intercepts Magnum's GlfwApplication's very very nasty behavior exit(0)
 * if glfw cant be initialized.
 */

static const rendering::GlfwApplication::Arguments& glfwChecker(const rendering::GlfwApplication::Arguments &args) {

    if(!glfwInit()) {
        std::string err = "Could not initialize GLFW: ";

        const char* description = NULL;
        int code = glfwGetError(&description);

        err += "error code: " + std::to_string(code) + ", " + description;

        tf_exp(std::runtime_error(err));
    }

    return args;
}

rendering::GlfwApplication::GlfwApplication(const Arguments &args) :
    Platform::GlfwApplication{glfwChecker(args), NoCreate}
{
    TF_Log(LOG_TRACE);
}



HRESULT rendering::GlfwApplication::pollEvents()
{
    glfwPollEvents();
    TF_GLFW_CHECK();
}

HRESULT rendering::GlfwApplication::waitEvents()
{
    glfwWaitEvents();
    TF_GLFW_CHECK();
}

HRESULT rendering::GlfwApplication::waitEventsTimeout(double timeout)
{
    glfwWaitEventsTimeout(timeout);
    TF_GLFW_CHECK();
}

HRESULT rendering::GlfwApplication::postEmptyEvent()
{
    glfwPostEmptyEvent();
    TF_GLFW_CHECK();
}

Magnum::GL::AbstractFramebuffer& rendering::GlfwApplication::framebuffer() {
    return GL::defaultFramebuffer;
}



void rendering::GlfwApplication::drawEvent() {
    GL::defaultFramebuffer.clear(GL::FramebufferClear::Color | GL::FramebufferClear::Depth);


    /* Draw particles */
    _ren->draw();

    swapBuffers();
    _timeline.nextFrame();
}

static rendering::GlfwApplication::Configuration magConf(const Simulator::Config &sc) {
    rendering::GlfwApplication::Configuration mc;

    mc.setTitle(sc.title());

    uint32_t wf = sc.windowFlags();

    if(wf & Simulator::AutoIconify) {
        mc.addWindowFlags(rendering::GlfwApplication::Configuration::WindowFlag::AutoIconify);
    }

    if(wf & Simulator::AlwaysOnTop) {
        mc.addWindowFlags(rendering::GlfwApplication::Configuration::WindowFlag::AlwaysOnTop);
    }

    if(wf & Simulator::AutoIconify) {
        mc.addWindowFlags(rendering::GlfwApplication::Configuration::WindowFlag::AutoIconify);
    }

    if(wf & Simulator::Borderless) {
        mc.addWindowFlags(rendering::GlfwApplication::Configuration::WindowFlag::Borderless);
    }

    if(wf & Simulator::Contextless) {
        mc.addWindowFlags(rendering::GlfwApplication::Configuration::WindowFlag::Contextless);
    }

    if(wf & Simulator::Focused) {
        mc.addWindowFlags(rendering::GlfwApplication::Configuration::WindowFlag::Focused);
    }

    if(wf & Simulator::Fullscreen) {
        mc.addWindowFlags(rendering::GlfwApplication::Configuration::WindowFlag::Fullscreen);
    }
    if(wf & Simulator::Hidden) {
        mc.addWindowFlags(rendering::GlfwApplication::Configuration::WindowFlag::Hidden);
    }
    if(wf & Simulator::Maximized) {
        mc.addWindowFlags(rendering::GlfwApplication::Configuration::WindowFlag::Maximized);
    }
    if(wf & Simulator::Minimized) {
        mc.addWindowFlags(rendering::GlfwApplication::Configuration::WindowFlag::Minimized);
    }

    if(wf & Simulator::Resizable) {
        mc.addWindowFlags(rendering::GlfwApplication::Configuration::WindowFlag::Resizable);
    }


    return mc;
}

HRESULT rendering::GlfwApplication::createContext(const Simulator::Config &conf)
{
    TF_Log(LOG_DEBUG);

    const Vector2 dpiScaling = this->dpiScaling({});
    Configuration c = magConf(conf);
    c.setSize(conf.windowSize(), dpiScaling);

    GLConfiguration glConf;
    glConf.setSampleCount(dpiScaling.max() < 2.0f ? 8 : 2);

    glfwWindowHint(GLFW_FOCUS_ON_SHOW, GLFW_TRUE);

    TF_Log(LOG_TRACE) << "calling tryCreate(c)";
    bool b = tryCreate(c);

    if(!b) {
        TF_Log(LOG_DEBUG) << "tryCreate failed";
        return E_FAIL;
    }

    _context = &GL::Context::current();

    _win = new rendering::GlfwWindow(this->window());

    if(conf.windowFlags() & Simulator::WindowFlags::Focused) {
        glfwFocusWindow(this->window());
    }

    _ren = new rendering::UniverseRenderer{conf, _win};

    return S_OK;
}

rendering::GlfwWindow* rendering::GlfwApplication::getWindow()
{
    return _win;
}

HRESULT rendering::GlfwApplication::setSwapInterval(int si)
{
    glfwSwapInterval(si);
    TF_GLFW_CHECK();
}

rendering::UniverseRenderer* rendering::GlfwApplication::getRenderer()
{
    return _ren;
}



HRESULT rendering::GlfwApplication::messageLoop(double et)
{
    FloatP_t initTime = _Engine.time * _Engine.dt;
    FloatP_t endTime = std::numeric_limits<FloatP_t>::infinity();

    if(et >= 0) {
        endTime = initTime + et;
    }

    TF_Log(LOG_DEBUG) << "GlfwApplication::messageLoop(" << et
                   << ") {now_time: " << initTime << ", end_time: " << endTime <<  "}" ;

    // process initial messages.
    Magnum::Platform::GlfwApplication::mainLoopIteration();

    // show the window
    showWindow();

    // need to post an empty message for some reason.
    // if you start the app, the simulation loop won't start
    // until the moust moves or some event happens, so send a
    // empty event here to start it.
    glfwPostEmptyEvent();

#if defined(_WIN32)
    TF_Log(LOG_DEBUG) << "set forground window";
    GLFWwindow* wnd = window();
    HWND hwnd = glfwGetWin32Window(wnd);
    SetForegroundWindow(hwnd);
#endif

    HRESULT hr;

    // run while it's visible, process window messages
    while(_Engine.time * _Engine.dt < endTime &&
          Magnum::Platform::GlfwApplication::mainLoopIteration() &&
          glfwGetWindowAttrib(Magnum::Platform::GlfwApplication::window(), GLFW_VISIBLE)) {

        // keep processing messages until window closes.
        if(!errOccurred() && Universe_Flag(Universe::Flags::RUNNING)) {
            if(!SUCCEEDED((hr = Application::simulationStep()))) {
                TF_Log(LOG_CRITICAL) << "something went wrong.";
                close();
            }
            Magnum::Platform::GlfwApplication::redraw();
        }
    }
    return S_OK;
}

HRESULT rendering::GlfwApplication::mainLoopIteration(double timeout) {
    HRESULT hr;
    if(!errOccurred() && Universe_Flag(Universe::Flags::RUNNING)) {

        // perform a simulation step if universe is in running state
        if(FAILED((hr = Application::simulationStep()))) {
            // window close message
            close();

            // process messages until window closes
            while(Magnum::Platform::GlfwApplication::window() &&
                  glfwGetWindowAttrib(Magnum::Platform::GlfwApplication::window(), GLFW_VISIBLE)) {
                Magnum::Platform::GlfwApplication::mainLoopIteration();
            }
            return hr;
        }
    }
    else {
        glfwPostEmptyEvent();
    }

    Simulator::get()->redraw();

    // process messages
    Magnum::Platform::GlfwApplication::mainLoopIteration();
    return S_OK;
}

HRESULT rendering::GlfwApplication::redraw()
{
    Magnum::Platform::GlfwApplication::redraw();
    return S_OK;
}

void rendering::GlfwApplication::viewportEvent(ViewportEvent &event)
{
    _ren->viewportEvent(event);
}

void rendering::GlfwApplication::keyPressEvent(KeyEvent &event)
{
    event::KeyEvent::invoke(event);

    if(event.isAccepted()) {
        return;
    }

    bool handled = false;
    switch(event.key()) {
        case Magnum::Platform::GlfwApplication::KeyEvent::Key::Space: {
            bool current = Universe_Flag(Universe::Flags::RUNNING);
            Universe_SetFlag(Universe::Flags::RUNNING, !current);
            break;
        }
        case Magnum::Platform::GlfwApplication::KeyEvent::KeyEvent::Key::S: {
            Universe::step(0, 0);
            Simulator::redraw();
        }
        default:
            break;
    }

    if(handled) {
        event.setAccepted();
    }
    else {
        _ren->keyPressEvent(event);
    }
}

void rendering::GlfwApplication::mousePressEvent(MouseEvent &event)
{
    _ren->mousePressEvent(event);
}

void rendering::GlfwApplication::mouseReleaseEvent(MouseEvent &event)
{
    _ren->mouseReleaseEvent(event);
}

void rendering::GlfwApplication::mouseMoveEvent(MouseMoveEvent &event)
{
    _ren->mouseMoveEvent(event);
}

void rendering::GlfwApplication::mouseScrollEvent(MouseScrollEvent &event)
{
    _ren->mouseScrollEvent(event);
}

void rendering::GlfwApplication::exitEvent(ExitEvent &event)
{
    TF_Log(LOG_DEBUG);

    // stop the window from getting (getting destroyed)
    glfwSetWindowShouldClose(window(), false);


    // "close", actually hide the window.
    close();

    event.setAccepted();
}

HRESULT rendering::GlfwApplication::destroy()
{
    TF_Log(LOG_DEBUG);

    GLFWwindow *window = Magnum::Platform::GlfwApplication::window();

    glfwSetWindowShouldClose(window, true);

    return S_OK;

}

HRESULT rendering::GlfwApplication::close()
{
    TF_Log(LOG_DEBUG);

    glfwHideWindow(window());

    return S_OK;
}

int rendering::GlfwApplication::windowAttribute(rendering::WindowAttributes attr)
{
    return glfwGetWindowAttrib(window(), attr);
}

HRESULT rendering::GlfwApplication::setWindowAttribute(rendering::WindowAttributes attr, int val)
{
    glfwSetWindowAttrib(window(), attr, val);
    TF_GLFW_CHECK();
}

#ifdef _WIN32

HRESULT ForceForgoundWindow1(GLFWwindow *wnd) {

    TF_Log(LOG_DEBUG);

    HWND window = glfwGetWin32Window(wnd);

    // This implementation registers a hot key (F22) and then
    // triggers the hot key.  When receiving the hot key, we'll
    // be in the foreground and allowed to move the target window
    // into the foreground too.

    //set_window_style(WS_POPUP);
    //Init(NULL, gfx::Rect());

    static const int kHotKeyId = 0x0000baba;
    static const int kHotKeyWaitTimeout = 2000;

    // Store the target window into our USERDATA for use in our
    // HotKey handler.
    RegisterHotKey(window, kHotKeyId, 0, VK_F22);

    // If the calling thread is not yet a UI thread, call PeekMessage
    // to ensure creation of its message queue.
    MSG msg = { 0 };
    PeekMessage(&msg, NULL, 0, 0, PM_NOREMOVE);

    // Send the Hotkey.
    INPUT hotkey = { 0 };
    hotkey.type = INPUT_KEYBOARD;
    hotkey.ki.wVk = VK_F22;
    if (1 != SendInput(1, &hotkey, sizeof(hotkey))) {
        std::cerr << "Failed to send input; GetLastError(): " << GetLastError();
        return E_FAIL;
    }

    // There are scenarios where the WM_HOTKEY is not dispatched by the
 // the corresponding foreground thread. To prevent us from indefinitely
 // waiting for the hotkey, we set a timer and exit the loop.
    SetTimer(window, kHotKeyId, kHotKeyWaitTimeout, NULL);

    // Loop until we get the key or the timer fires.
    while (GetMessage(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);

        if (WM_HOTKEY == msg.message)
            break;
        if (WM_TIMER == msg.message) {
            SetForegroundWindow(window);
            break;
        }
    }

    UnregisterHotKey(window, kHotKeyId);
    KillTimer(window, kHotKeyId);

    return S_OK;
}


void ForceForgoundWindow2(GLFWwindow* wnd)
{
    TF_Log(LOG_DEBUG);

    HWND hWnd = glfwGetWin32Window(wnd);

    if (!::IsWindow(hWnd)) return;

    //relation time of SetForegroundWindow lock
    DWORD lockTimeOut = 0;
    HWND  hCurrWnd = ::GetForegroundWindow();
    DWORD dwThisTID = ::GetCurrentThreadId(),
        dwCurrTID = ::GetWindowThreadProcessId(hCurrWnd, 0);

    //we need to bypass some limitations from Microsoft :)
    if (dwThisTID != dwCurrTID)
    {
        ::AttachThreadInput(dwThisTID, dwCurrTID, TRUE);

        ::SystemParametersInfo(SPI_GETFOREGROUNDLOCKTIMEOUT, 0, &lockTimeOut, 0);
        ::SystemParametersInfo(SPI_SETFOREGROUNDLOCKTIMEOUT, 0, 0, SPIF_SENDWININICHANGE | SPIF_UPDATEINIFILE);

        ::AllowSetForegroundWindow(ASFW_ANY);
    }

    ::SetForegroundWindow(hWnd);

    if (dwThisTID != dwCurrTID)
    {
        ::SystemParametersInfo(SPI_SETFOREGROUNDLOCKTIMEOUT, 0, (PVOID)lockTimeOut, SPIF_SENDWININICHANGE | SPIF_UPDATEINIFILE);
        ::AttachThreadInput(dwThisTID, dwCurrTID, FALSE);
    }
}

#endif

HRESULT rendering::GlfwApplication::show()
{
    TF_Log(LOG_DEBUG);

    showWindow();

    if (!isTerminalInteractiveShell()) {
        return messageLoop(-1);
    }

    return S_OK;
}

HRESULT rendering::GlfwApplication::showWindow()
{
    TF_Log(LOG_DEBUG);

    glfwShowWindow(window());

#ifdef _WIN32
    if (!isTerminalInteractiveShell()) {
        ForceForgoundWindow1(window());
    }
#endif

    TF_GLFW_CHECK();
}

bool rendering::GlfwApplication::contextMakeCurrent()
{
    // tell open go to make the context current.
    glfwMakeContextCurrent(_win->_window);

    // tell Magnum to set it's context
    Magnum::GL::Context::makeCurrent(_context);

    return true;
}

bool rendering::GlfwApplication::contextHasCurrent()
{
    bool hasGlfw = glfwGetCurrentContext() != NULL;
    bool hasMagnum = Magnum::GL::Context::hasCurrent();

    if(!(hasGlfw ^ hasMagnum)) {
        std::string msg = "GLFW and Magnum OpenGL contexts not synchronized, glfw context: ";
        msg += std::to_string(hasGlfw);
        msg += ", magnum context: " + std::to_string(hasMagnum);
        tf_exp(std::runtime_error(msg));
    }

    return hasGlfw && hasMagnum;
}

bool rendering::GlfwApplication::contextRelease()
{
    return false;
}
