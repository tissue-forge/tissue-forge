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

/**
 * @file tf_py.h
 * 
 */

#ifndef _SOURCE_LANGS_PY_TF_PY_H_
#define _SOURCE_LANGS_PY_TF_PY_H_

/// Include Python header, disable linking to pythonX_d.lib on Windows in debug mode
#if defined(_MSC_VER) || defined(_WIN32)
#  pragma warning(push)
#  pragma warning(disable: 4510 4610 4512 4005)
#  include <corecrt.h>
#  if defined(_DEBUG)
#    define TF_DEBUG_MARKER
#    undef _DEBUG
#  endif
#endif

#include <Python.h>

#if defined(_MSC_VER) || defined(_WIN32)
#  if defined(TF_DEBUG_MARKER)
#    define _DEBUG
#    undef  TF_DEBUG_MARKER
#  endif
#  pragma warning(pop)
#endif

#include <tf_port.h>
#include <types/tf_types.h>
#include <types/tf_cast.h>

#include <Magnum/Magnum.h>
#include <Magnum/Math/Vector3.h>
#include <Magnum/Math/Vector4.h>
#include <Magnum/Math/Matrix3.h>

#include <string>


namespace TissueForge {


    template<>
    Magnum::Vector2 cast(PyObject *obj);

    template<>
    Magnum::Vector3 cast(PyObject *obj);

    template<>
    Magnum::Vector4 cast(PyObject *obj);

    template<>
    Magnum::Vector2i cast(PyObject *obj);

    template<>
    Magnum::Vector3i cast(PyObject *obj);

    template<>
    fVector2 cast(PyObject *obj);

    template<>
    fVector3 cast(PyObject *obj);

    template<>
    fVector4 cast(PyObject *obj);

    template<>
    dVector2 cast(PyObject *obj);

    template<>
    dVector3 cast(PyObject *obj);

    template<>
    dVector4 cast(PyObject *obj);

    template<>
    iVector2 cast(PyObject *obj);

    template<>
    iVector3 cast(PyObject *obj);

    template<>
    PyObject* cast<int16_t, PyObject*>(const int16_t &i);

    template<>
    PyObject* cast<uint16_t, PyObject*>(const uint16_t &i);

    template<>
    PyObject* cast<uint32_t, PyObject*>(const uint32_t &i);

    template<>
    PyObject* cast<uint64_t, PyObject*>(const uint64_t &i);

    template<>
    PyObject* cast<float, PyObject*>(const float &f);

    template<>
    PyObject* cast<double, PyObject*>(const double &f);

    template<>
    float cast(PyObject *obj);

    template<>
    double cast(PyObject *obj);

    template<>
    PyObject* cast<bool, PyObject*>(const bool &f);

    template<>
    bool cast(PyObject *obj);

    template<>
    PyObject* cast<int, PyObject*>(const int &i);

    template<>
    int cast(PyObject *obj);

    template<>
    PyObject* cast<std::string, PyObject*>(const std::string &s);

    template<>
    std::string cast(PyObject *o);

    template<>
    int16_t cast(PyObject *o);

    template<>
    uint16_t cast(PyObject *o);

    template<>
    uint32_t cast(PyObject *o);

    template<>
    uint64_t cast(PyObject *o);


    namespace py {


        CAPI_FUNC(PyObject*) Import_ImportString(const std::string &name);
        CAPI_FUNC(PyObject*) iPython_Get();
        CAPI_FUNC(bool) terminalInteractiveShell();
        CAPI_FUNC(bool) ZMQInteractiveShell();

        /**
         * check if type can be converted
         */
        template <typename T>
        bool check(PyObject *o);

        /**
         * grab either the i'th arg from the args, or keywords.
         *
         * gets a reference to the object, NULL if not exist.
         */
        PyObject *py_arg(const char* name, int index, PyObject *_args, PyObject *_kwargs);

            /**
             * gets the __repr__ / __str__ representations of python objects
             */
        std::string repr(PyObject *o);
        std::string str(PyObject *o);

        /**
         * get the python error string, empty string if no error.
         */
        std::string pyerror_str();

        template<typename T>
        T arg(const char* name, int index, PyObject *args, PyObject *kwargs) {
            PyObject *value = py_arg(name, index, args, kwargs);
            if(value) {
                return cast<PyObject, T>(value);
            }
            throw std::runtime_error(std::string("missing argument ") + name);
        };

        template<typename T>
        T arg(const char* name, int index, PyObject *args, PyObject *kwargs, T deflt) {

            PyObject *value = py_arg(name, index, args, kwargs);
            if(value) {
                return cast<PyObject, T>(value);
            }
            return deflt;
        };

    };

}

#endif // _SOURCE_LANGS_PY_TF_PY_H_
