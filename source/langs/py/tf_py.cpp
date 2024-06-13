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

#include "tf_py.h"

#include <tfLogger.h>


using namespace TissueForge;


bool hasPython() {
    if(!Py_IsInitialized()) {
        TF_Log(LOG_DEBUG) << "Python not initialized";
        return false;
    }
    return true;
}

#define TF_CHECKPYRET(x) if(!hasPython()) return x;

PyObject *py::Import_ImportString(const std::string &name) {
    TF_CHECKPYRET(0)

    PyObject *s = cast<std::string, PyObject*>(name);
    PyObject *mod = PyImport_Import(s);
    Py_DECREF(s);
    return mod;
}

std::string py::pyerror_str()
{
    TF_CHECKPYRET("")

    std::string result;
    // get the error details
    PyObject *pExcType = NULL , *pExcValue = NULL , *pExcTraceback = NULL ;
    PyErr_Fetch( &pExcType , &pExcValue , &pExcTraceback ) ;
    if ( pExcType != NULL )
    {
        PyObject* pRepr = PyObject_Repr( pExcType ) ;
        
        PyObject * _str=PyUnicode_AsASCIIString(pRepr);
        result += std::string("EXC type: ") + PyBytes_AsString(_str);
        Py_DECREF(_str);
        
        Py_DecRef( pRepr ) ;
        Py_DecRef( pExcType ) ;
    }
    if ( pExcValue != NULL )
    {
        PyObject* pRepr = PyObject_Repr( pExcValue ) ;

        PyObject * _str=PyUnicode_AsASCIIString(pRepr);
        result += std::string("EXC value: ") + PyBytes_AsString(_str);
        Py_DECREF(_str);
        
        Py_DecRef( pRepr ) ;
        Py_DecRef( pExcValue ) ;
    }
    if ( pExcTraceback != NULL )
    {
        PyObject* pRepr = PyObject_Repr( pExcValue ) ;
        
        PyObject * _str=PyUnicode_AsASCIIString(pRepr);
        result += std::string("EXC traceback: ") + PyBytes_AsString(_str);
        Py_DECREF(_str);
        
        Py_DecRef( pRepr ) ;
        Py_DecRef( pExcTraceback ) ;
    }
    
    return result;
}

PyObject* py::iPython_Get() {
    TF_CHECKPYRET(0)

    PyObject* moduleString = PyUnicode_FromString("IPython.core.getipython");
    
    if(!moduleString) {
        return NULL;
    }
    
    #if defined(__has_feature)
    #  if __has_feature(thread_sanitizer)
        std::cout << "thread sanitizer, returning NULL" << std::endl;
        return NULL;
    #  endif
    #endif
    
    PyObject* module = PyImport_Import(moduleString);
    if(!module) {
        PyObject *err = PyErr_Occurred();
        
        TF_Log(LOG_DEBUG) << "could not import IPython.core.getipython"
            << ", "
            << py::pyerror_str()
            << ", returning NULL";
        PyErr_Clear();
        Py_DECREF(moduleString);
        return NULL;
    }
    
    // Then getting a reference to your function :

    PyObject* get_ipython = PyObject_GetAttrString(module,(char*)"get_ipython");
    
    if(!get_ipython) {
        PyObject *err = PyErr_Occurred();
        TF_Log(LOG_WARNING) << "PyObject_GetAttrString(\"get_ipython\") failed: "
            << py::pyerror_str()
            << ", returning NULL";
        PyErr_Clear();
        Py_DECREF(moduleString);
        Py_DECREF(module);
        return NULL;
    }

    PyObject* result = PyObject_CallObject(get_ipython, NULL);
    
    if(result == NULL) {
        PyObject* err = PyErr_Occurred();
        std::string _str = "error calling IPython.core.getipython.get_ipython(): ";
        _str += py::pyerror_str();
        TF_Log(LOG_FATAL) << _str;
        PyErr_Clear();
    }
    
    Py_DECREF(moduleString);
    Py_DECREF(module);
    Py_DECREF(get_ipython);
    
    TF_Log(LOG_TRACE);
    return result;
}

bool py::terminalInteractiveShell() {
    TF_CHECKPYRET(false)

    PyObject* ipy = py::iPython_Get();
    bool result = false;

    if (ipy && strcmp("TerminalInteractiveShell", ipy->ob_type->tp_name) == 0) {
        result = true;
    }
    
    TF_Log(LOG_TRACE) << "returning: " << result;
    Py_XDECREF(ipy);
    return result;
}

bool py::ZMQInteractiveShell() {
    PyObject* ipy = py::iPython_Get();
    bool result = false;

    if (ipy && strcmp("ZMQInteractiveShell", ipy->ob_type->tp_name) == 0) {
        result = true;
    }
    
    TF_Log(LOG_TRACE) << "returning: " << result;
    Py_XDECREF(ipy);
    return result;
}

template<>
float TissueForge::cast(PyObject *obj) {
    TF_CHECKPYRET(0)

    if(PyNumber_Check(obj)) {
        return PyFloat_AsDouble(obj);
    }
    throw std::domain_error("can not convert to number");
}

template<>
double TissueForge::cast(PyObject *obj) {
    TF_CHECKPYRET(0)

    if(PyNumber_Check(obj)) {
        return PyFloat_AsDouble(obj);
    }
    throw std::domain_error("can not convert to number");
}

static Magnum::Vector3 vector3_from_list(PyObject *obj) {
    Magnum::Vector3 result = {};
    
    TF_CHECKPYRET(result)

    if(PyList_Size(obj) != 3) {
        throw std::domain_error("error, must be length 3 list to convert to vector3");
    }
    
    for(int i = 0; i < 3; ++i) {
        PyObject *item = PyList_GetItem(obj, i);
        if(PyNumber_Check(item)) {
            result[i] = PyFloat_AsDouble(item);
        }
        else {
            throw std::domain_error("error, can not convert list item to number");
        }
    }
    
    return result;
}

static Magnum::Vector4 vector4_from_list(PyObject *obj) {
    Magnum::Vector4 result = {};

    TF_CHECKPYRET(result)
    
    if(PyList_Size(obj) != 4) {
        throw std::domain_error("error, must be length 3 list to convert to vector3");
    }
    
    for(int i = 0; i < 4; ++i) {
        PyObject *item = PyList_GetItem(obj, i);
        if(PyNumber_Check(item)) {
            result[i] = PyFloat_AsDouble(item);
        }
        else {
            throw std::domain_error("error, can not convert list item to number");
        }
    }
    
    return result;
}

static Magnum::Vector2 vector2_from_list(PyObject *obj) {
    Magnum::Vector2 result = {};
    
    TF_CHECKPYRET(result)

    if(PyList_Size(obj) != 2) {
        throw std::domain_error("error, must be length 2 list to convert to vector3");
    }
    
    for(int i = 0; i < 2; ++i) {
        PyObject *item = PyList_GetItem(obj, i);
        if(PyNumber_Check(item)) {
            result[i] = PyFloat_AsDouble(item);
        }
        else {
            throw std::domain_error("error, can not convert list item to number");
        }
    }
    
    return result;
}

static Magnum::Vector3i vector3i_from_list(PyObject *obj) {
    Magnum::Vector3i result = {};

    TF_CHECKPYRET(result)

    if(PyList_Size(obj) != 3) {
        throw std::domain_error("error, must be length 3 list to convert to vector3");
    }
    
    for(int i = 0; i < 3; ++i) {
        PyObject *item = PyList_GetItem(obj, i);
        if(PyNumber_Check(item)) {
            result[i] = PyLong_AsLong(item);
        }
        else {
            throw std::domain_error("error, can not convert list item to number");
        }
    }
    
    return result;
}
    
static Magnum::Vector2i vector2i_from_list(PyObject *obj) {
    Magnum::Vector2i result = {};

    TF_CHECKPYRET(result)

    if(PyList_Size(obj) != 2) {
        throw std::domain_error("error, must be length 2 list to convert to vector2");
    }
    
    for(int i = 0; i < 2; ++i) {
        PyObject *item = PyList_GetItem(obj, i);
        if(PyNumber_Check(item)) {
            result[i] = PyLong_AsLong(item);
        }
        else {
            throw std::domain_error("error, can not convert list item to number");
        }
    }
    
    return result;
}

template<>
Magnum::Vector3 TissueForge::cast(PyObject *obj) {

    TF_CHECKPYRET(Magnum::Vector3(0))

    if(PyList_Check(obj)) {
        return vector3_from_list(obj);
    }
    throw std::domain_error("can not convert non-list to vector");
}

template<>
Magnum::Vector4 TissueForge::cast(PyObject *obj) {

    TF_CHECKPYRET(Magnum::Vector4(0))

    if(PyList_Check(obj)) {
        return vector4_from_list(obj);
    }
    throw std::domain_error("can not convert non-list to vector");
}

template<>
Magnum::Vector2 TissueForge::cast(PyObject *obj) {

    TF_CHECKPYRET(Magnum::Vector2(0))

    if(PyList_Check(obj)) {
        return vector2_from_list(obj);
    }
    throw std::domain_error("can not convert non-list to vector");
}

template<>
Magnum::Vector3i TissueForge::cast(PyObject *obj) {

    TF_CHECKPYRET(Magnum::Vector3i(0))

    if(PyList_Check(obj)) {
        return vector3i_from_list(obj);
    }
    throw std::domain_error("can not convert non-list to vector");
}

template<>
Magnum::Vector2i TissueForge::cast(PyObject *obj) {

    TF_CHECKPYRET(Magnum::Vector2i(0))

    if(PyList_Check(obj)) {
        return vector2i_from_list(obj);
    }
    throw std::domain_error("can not convert non-list to vector");
}

template<>
fVector2 TissueForge::cast(PyObject *obj) { return fVector2(cast<PyObject, Magnum::Vector2>(obj)); }

template<>
fVector3 TissueForge::cast(PyObject *obj) { return fVector3(cast<PyObject, Magnum::Vector3>(obj)); }

template<>
fVector4 TissueForge::cast(PyObject *obj) { return fVector4(cast<PyObject, Magnum::Vector4>(obj)); }

template<>
dVector2 TissueForge::cast(PyObject *obj) { return dVector2(cast<PyObject, fVector2>(obj)); }

template<>
dVector3 TissueForge::cast(PyObject *obj) { return dVector3(cast<PyObject, fVector3>(obj)); }

template<>
dVector4 TissueForge::cast(PyObject *obj) { return dVector4(cast<PyObject, fVector4>(obj)); }

template<>
iVector2 TissueForge::cast(PyObject *obj) { return iVector2(cast<PyObject, Magnum::Vector2i>(obj)); }

template<>
iVector3 TissueForge::cast(PyObject *obj) { return iVector3(cast<PyObject, Magnum::Vector3i>(obj)); }

template<>
PyObject* TissueForge::cast<float, PyObject*>(const float &f) {
    TF_CHECKPYRET(0)

    return PyFloat_FromDouble(f);
}

template<>
PyObject* TissueForge::cast<double, PyObject*>(const double &f) {
    TF_CHECKPYRET(0)

    return PyFloat_FromDouble(f);
}

template<>
PyObject* TissueForge::cast<int16_t, PyObject*>(const int16_t &i) {
    TF_CHECKPYRET(0)

    return PyLong_FromLong(i);
}

template<>
PyObject* TissueForge::cast<uint16_t, PyObject*>(const uint16_t &i) {
    TF_CHECKPYRET(0)

    return PyLong_FromLong(i);
}

template<>
PyObject* TissueForge::cast<uint32_t, PyObject*>(const uint32_t &i) {
    TF_CHECKPYRET(0)

    return PyLong_FromLong(i);
}

template<>
PyObject* TissueForge::cast<uint64_t, PyObject*>(const uint64_t &i) {
    TF_CHECKPYRET(0)

    return PyLong_FromLong(i);
}

template<>
bool TissueForge::cast(PyObject *obj) {
    TF_CHECKPYRET(0)

    if(PyBool_Check(obj)) {
        return obj == Py_True ? true : false;
    }
    throw std::domain_error("can not convert to boolean");
}

template<>
PyObject* TissueForge::cast<bool, PyObject*>(const bool &b) {
    TF_CHECKPYRET(0)

    if(b) {
        Py_RETURN_TRUE;
    }
    else {
        Py_RETURN_FALSE;
    }
}

template <>
bool py::check<bool>(PyObject *o) {
    TF_CHECKPYRET(0)

    return PyBool_Check(o);
}

PyObject *py::py_arg(const char* name, int index, PyObject *_args, PyObject *_kwargs) {
    TF_CHECKPYRET(0)

    PyObject *kwobj = _kwargs ?  PyDict_GetItemString(_kwargs, name) : NULL;
    PyObject *aobj = _args && (PyTuple_Size(_args) > index) ? PyTuple_GetItem(_args, index) : NULL;
    
    if(aobj && kwobj) {
        std::string msg = std::string("Error, argument \"") + name + "\" given both as a keyword and positional";
        throw std::logic_error(msg.c_str());
    }
    
    return aobj ? aobj : kwobj;
}

template<>
PyObject* TissueForge::cast<int, PyObject*>(const int &i) {
    TF_CHECKPYRET(0)

    return PyLong_FromLong(i);
}

template<>
int TissueForge::cast(PyObject *obj){
    TF_CHECKPYRET(0)

    if(PyNumber_Check(obj)) {
        return PyLong_AsLong(obj);
    }
    throw std::domain_error("can not convert to number");
}

template<>
PyObject* TissueForge::cast<std::string, PyObject*>(const std::string &s) {
    TF_CHECKPYRET(0)

    return PyUnicode_FromString(s.c_str());
}

template<>
std::string TissueForge::cast(PyObject *o) {
    TF_CHECKPYRET("")

    if(PyUnicode_Check(o)) {
        const char* c = PyUnicode_AsUTF8(o);
        return std::string(c);
    }
    else {
        std::string msg = "could not convert ";
        msg += o->ob_type->tp_name;
        msg += " to string";
        throw std::domain_error(msg);
    }
}

template<>
int16_t TissueForge::cast(PyObject *o) {return (int16_t)cast<PyObject, int>(o);}

template<>
uint16_t TissueForge::cast(PyObject *o) {return (uint16_t)cast<PyObject, int>(o);}

template<>
uint32_t TissueForge::cast(PyObject *o) {return (uint32_t)cast<PyObject, int>(o);}

template<>
uint64_t TissueForge::cast(PyObject *o) {return (uint64_t)cast<PyObject, int>(o);}

template <>
bool py::check<std::string>(PyObject *o) {
    TF_CHECKPYRET(0)

    return o && PyUnicode_Check(o);
}

template <>
bool py::check<float>(PyObject *o) {
    TF_CHECKPYRET(0)

    return o && PyNumber_Check(o);
}

template <>
bool py::check<double>(PyObject *o) {
    TF_CHECKPYRET(0)

    return o && PyNumber_Check(o);
}

std::string py::repr(PyObject *o) {
    TF_CHECKPYRET("")

    PyObject* pRepr = PyObject_Repr( o ) ;
    
    PyObject * _str=PyUnicode_AsASCIIString(pRepr);
    std::string result = std::string(PyBytes_AsString(_str));
    Py_DECREF(_str);
    
    Py_DecRef( pRepr ) ;
    return result;
}

std::string py::str(PyObject *o) {
    TF_CHECKPYRET("")

    std::string result;
    if(o) {
        PyObject* pStr = PyObject_Str( o ) ;
        if(pStr) {
            PyObject *_str = PyUnicode_AsASCIIString(pStr);
            result = std::string(PyBytes_AsString(_str));
            Py_DECREF(_str);
            Py_DecRef( pStr ) ;
        }
        else {
            result += "error calling PyObject_Str(o)";
        }
    }
    else {
        result = "NULL";
    }
    return result;
}
