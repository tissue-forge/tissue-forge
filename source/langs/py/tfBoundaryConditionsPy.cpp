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

#include "tfBoundaryConditionsPy.h"

#include <tfLogger.h>


using namespace TissueForge;


py::BoundaryConditionsArgsContainerPy::BoundaryConditionsArgsContainerPy(PyObject *obj) {
    if(PyLong_Check(obj)) setValueAll(cast<PyObject, int>(obj));
    else if(PyDict_Check(obj)) {
        PyObject *keys = PyDict_Keys(obj);

        for(unsigned int i = 0; i < PyList_Size(keys); ++i) {
            PyObject *key = PyList_GetItem(keys, i);
            PyObject *value = PyDict_GetItem(obj, key);

            std::string name = cast<PyObject, std::string>(key);
            if(PyLong_Check(value)) {
                unsigned int v = cast<PyObject, unsigned int>(value);

                TF_Log(LOG_DEBUG) << name << ": " << value;

                setValue(name, v);
            }
            else if(py::check<std::string>(value)) {
                std::string s = cast<PyObject, std::string>(value);

                TF_Log(LOG_DEBUG) << name << ": " << s;

                setValue(name, BoundaryConditions::boundaryKindFromString(s));
            }
            else if(PySequence_Check(value)) {
                std::vector<std::string> kinds;
                PyObject *valueItem;
                for(unsigned int j = 0; j < PySequence_Size(value); j++) {
                    valueItem = PySequence_GetItem(value, j);
                    if(py::check<std::string>(valueItem)) {
                        std::string s = cast<PyObject, std::string>(valueItem);

                        TF_Log(LOG_DEBUG) << name << ": " << s;

                        kinds.push_back(s);
                    }
                }
                setValue(name, BoundaryConditions::boundaryKindFromStrings(kinds));
            }
            else if(PyDict_Check(value)) {
                PyObject *vel = PyDict_GetItemString(value, "velocity");
                if(!vel) {
                    throw std::invalid_argument("attempt to initialize a boundary condition with a "
                                                "dictionary that does not contain a \'velocity\' item, "
                                                "only velocity boundary conditions support dictionary init");
                }
                FVector3 v = cast<PyObject, FVector3>(vel);

                TF_Log(LOG_DEBUG) << name << ": " << v;

                setVelocity(name, v);

                PyObject *restore = PyDict_GetItemString(value, "restore");
                if(restore) {
                    FloatP_t r = cast<PyObject, FloatP_t>(restore);

                    TF_Log(LOG_DEBUG) << name << ": " << r;

                    setRestore(name, r);
                }
            }
        }

        Py_DECREF(keys);
    }
}
