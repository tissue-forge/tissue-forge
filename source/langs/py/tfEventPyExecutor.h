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

#ifndef _SOURCE_LANGS_PY_TFEVENTPYEXECUTOR_H_
#define _SOURCE_LANGS_PY_TFEVENTPYEXECUTOR_H_

#include "tf_py.h"


namespace TissueForge {


   namespace py {


        template<typename event_t>
        struct EventPyExecutor {

            /**
             * @brief Issues call to execute event callback in python layer on existing event
             * 
             * @return HRESULT 
             */
            HRESULT invoke() {
                if(!hasExecutorPyCallable() || !activeEvent) return E_ABORT;

                PyObject *result = PyObject_CallObject(executorPyCallable, NULL);

                if(result == NULL) {
                    PyObject *err = PyErr_Occurred();
                    PyErr_Clear();
                    return E_FAIL;
                }
                Py_DECREF(result);

                return S_OK;
            }

            /**
             * @brief Issues call to execute event callback in python layer on new event
             * 
             * @param ke event on which to execute callback
             * @return HRESULT 
             */
            HRESULT invoke(event_t &ke) {
                activeEvent = &ke;

                return invoke();
            }

            /**
             * @brief Gets the current event object
             * 
             * @return event_t* 
             */
            event_t *getEvent() {
                return activeEvent;
            }

            /**
             * @brief Tests whether the executor callback from the python layer has been set
             * 
             * @return true callback has been set
             * @return false callback has not been set
             */
            bool hasExecutorPyCallable() { return executorPyCallable != NULL; }

            /**
             * @brief Sets the executor callback from the python layer
             * 
             * @param callable executor callback from the python layer
             */
            void setExecutorPyCallable(PyObject *callable) {
                resetExecutorPyCallable();
                executorPyCallable = callable;
                Py_INCREF(callable);
            }

            /**
             * @brief Sets the executor callback from the python layer if it has not yet been set
             * 
             * @param callable executor callback from the python layer
             */
            void maybeSetExecutorPyCallable(PyObject *callable) {
                if(hasExecutorPyCallable()) return;
                executorPyCallable = callable;
                Py_INCREF(callable);
            }

            /**
             * @brief Resets the executor callback from the python layer
             * 
             */
            void resetExecutorPyCallable() {
                if(hasExecutorPyCallable()) Py_DECREF(executorPyCallable);
                executorPyCallable = NULL;
            }

        protected:

            // The current event object, if any
            event_t *activeEvent = NULL;

            // The executor callback from the python layer
            PyObject *executorPyCallable = NULL;

        };

}};

#endif // _SOURCE_LANGS_PY_TFEVENTPYEXECUTOR_H_