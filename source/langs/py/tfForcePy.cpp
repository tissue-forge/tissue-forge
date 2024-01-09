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

#include "tfForcePy.h"

#include <tfLogger.h>


using namespace TissueForge;


FVector3 pyConstantForceFunction(PyObject *callable) {
    TF_Log(LOG_TRACE);

    PyObject *result = PyObject_CallObject(callable, NULL);

    if(result == NULL) {
        PyObject *err = PyErr_Occurred();
        TF_Log(LOG_CRITICAL) << py::pyerror_str();
        PyErr_Clear();
        return FVector3();
    }
    FVector3 out = cast<PyObject, FVector3>(result);
    Py_DECREF(result);
    return out;
}

py::CustomForcePy::CustomForcePy() : 
    CustomForce() 
{
    type = FORCE_CUSTOM;
}

py::CustomForcePy::CustomForcePy(const FVector3 &f, const FloatP_t &period) : 
    CustomForce(f, period)
{
    type = FORCE_CUSTOM;
    callable = NULL;
}

py::CustomForcePy::CustomForcePy(PyObject *f, const FloatP_t &period) : 
    CustomForce(), 
    callable(f)
{
    type = FORCE_CUSTOM;

    setPeriod(period);
    if(PyList_Check(f)) {
        FVector3 fv = cast<PyObject, FVector3>(f);
        callable = NULL;
        CustomForce::setValue(fv);
    }
    else if(callable) {
        Py_IncRef(callable);
    }
}

py::CustomForcePy::~CustomForcePy(){
    if(callable) Py_DecRef(callable);
}

void py::CustomForcePy::onTime(FloatP_t time)
{
    if(callable && time >= lastUpdate + updateInterval) {
        lastUpdate = time;
        setValue(callable);
    }
}

FVector3 py::CustomForcePy::getValue() {
    if(callable && callable != Py_None) return pyConstantForceFunction(callable);
    return force;
}

void py::CustomForcePy::setValue(PyObject *_userFunc) {
    if(_userFunc) callable = _userFunc;
    if(callable && callable != Py_None) CustomForce::setValue(getValue());
}

py::CustomForcePy *py::CustomForcePy::fromForce(Force *f) {
    if(f->type != FORCE_CUSTOM) 
        return 0;
    return (py::CustomForcePy*)f;
}


namespace TissueForge::io { 


    template <>
    HRESULT toFile(const py::CustomForcePy &dataElement, const MetaData &metaData, IOElement &fileElement) {
        
        TF_IOTOEASY(fileElement, metaData, "type", dataElement.type);
        TF_IOTOEASY(fileElement, metaData, "stateVectorIndex", dataElement.stateVectorIndex);
        TF_IOTOEASY(fileElement, metaData, "updateInterval", dataElement.updateInterval);
        TF_IOTOEASY(fileElement, metaData, "lastUpdate", dataElement.lastUpdate);
        TF_IOTOEASY(fileElement, metaData, "force", dataElement.force);

        fileElement.get()->type = "ConstantPyForce";

        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, py::CustomForcePy *dataElement) {

        TF_IOFROMEASY(fileElement, metaData, "type", &dataElement->type);
        TF_IOFROMEASY(fileElement, metaData, "stateVectorIndex", &dataElement->stateVectorIndex);
        TF_IOFROMEASY(fileElement, metaData, "updateInterval", &dataElement->updateInterval);
        TF_IOFROMEASY(fileElement, metaData, "lastUpdate", &dataElement->lastUpdate);
        TF_IOFROMEASY(fileElement, metaData, "force", &dataElement->force);
        dataElement->userFunc = NULL;
        dataElement->callable = NULL;

        return S_OK;
    }

};
