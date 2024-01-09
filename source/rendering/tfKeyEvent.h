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

#ifndef _SOURCE_RENDERING_TFKEYEVENT_H_
#define _SOURCE_RENDERING_TFKEYEVENT_H_

#include <Magnum/Platform/GlfwApplication.h>
#include <tf_port.h>


namespace TissueForge {


    namespace event {


        struct KeyEvent;

        using KeyEventDelegateType = HRESULT (*)(Magnum::Platform::GlfwApplication::KeyEvent*);
        using KeyEventHandlerType = HRESULT (*)(struct KeyEvent*);

        typedef size_t KeyEventDelegateHandle;
        typedef size_t KeyEventHandlerHandle;

        struct CAPI_EXPORT KeyEvent
        {
            Magnum::Platform::GlfwApplication::KeyEvent *glfw_event;

            KeyEvent(Magnum::Platform::GlfwApplication::KeyEvent *glfw_event=NULL) : glfw_event(glfw_event) {}

            HRESULT invoke();
            static HRESULT invoke(Magnum::Platform::GlfwApplication::KeyEvent &ke);
            
            /**
             * @brief Adds an event delegate
             * 
             * @param _delegate delegate to add
             * @return handle for future getting and removing 
             */
            static KeyEventDelegateHandle addDelegate(KeyEventDelegateType *_delegate);

            /**
             * @brief Adds an event handler
             * 
             * @param _handler handler to add
             * @return handle for future getting and removing
             */
            static KeyEventHandlerHandle addHandler(KeyEventHandlerType *_handler);

            /**
             * @brief Get an event delegate
             * 
             * @param handle delegate handle
             * @return delegate if handle is valid, otherwise NULL
             */
            static KeyEventDelegateType *getDelegate(const KeyEventDelegateHandle &handle);

            /**
             * @brief Get an event handler
             * 
             * @param handle handler handle
             * @return handler if handle is valid, otherwise NULL
             */
            static KeyEventHandlerType *getHandler(const KeyEventHandlerHandle &handle);

            /**
             * @brief Remove an event delegate
             * 
             * @param handle delegate handle
             * @return true when delegate is removed
             * @return false when handle is invalid
             */
            static bool removeDelegate(const KeyEventDelegateHandle &handle);

            /**
             * @brief Remove an event handler
             * 
             * @param handle handler handle
             * @return true when handler is removed
             * @return false when handle is invalid
             */
            static bool removeHandler(const KeyEventHandlerHandle &handle);

            std::string keyName();
            bool keyAlt();
            bool keyCtrl();
            bool keyShift();
        };

}}

#endif // _SOURCE_RENDERING_TFKEYEVENT_H_