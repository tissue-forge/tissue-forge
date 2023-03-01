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

%{

#include <rendering/tfKeyEvent.h>
#include <langs/py/tfKeyEventPy.h>

%}


tfEventPyExecutor_extender(KeyEventPyExecutor, TissueForge::event::KeyEvent)

%rename(_event_KeyEvent) TissueForge::event::KeyEvent;

%include <rendering/tfKeyEvent.h>
%include <langs/py/tfKeyEventPy.h>

%extend TissueForge::event::KeyEvent {
    %pythoncode %{
        @property
        def key_name(self) -> str:
            """Key pressed for this event"""
            return self.keyName()

        @property
        def key_alt(self) -> bool:
            """Flag for whether Alt key is pressed"""
            return self.keyAlt()

        @property
        def key_ctrl(self) -> bool:
            """Flag for whether Ctrl key is pressed"""
            return self.keyCtrl()

        @property
        def key_shift(self) -> bool:
            """Flag for whether Shift key is pressed"""
            return self.keyShift()
    %}
}

%extend TissueForge::py::KeyEventPyExecutor {
    %pythoncode %{
        @staticmethod
        def on_keypress(delegate):
            if KeyEventPyExecutor.hasStaticKeyEventPyExecutor():
                ex = KeyEventPyExecutor.getStaticKeyEventPyExecutor()
                cb = ex._callback
                def callback(e):
                    cb(e)
                    delegate(e)
            else:
                callback = delegate
            
            KeyEventPyExecutor.setStaticKeyEventPyExecutor(initKeyEventPyExecutor(callback))
    %}
}

%pythoncode %{

    def _event_on_keypress(invoke_method):
        """
        Registers a callback for handling keyboard events

        :type invoke_method: PyObject
        :param invoke_method: an invoke method; evaluated when an event occurs. 
            Takes an :class:`KeyEvent` instance as argument and returns None
        """
        return KeyEventPyExecutor.on_keypress(invoke_method)
%}
