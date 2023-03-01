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

#ifndef _SOURCE_LANGS_PY_TFFORCEPY_H_
#define _SOURCE_LANGS_PY_TFFORCEPY_H_


#include "tf_py.h"
#include <tfForce.h>


namespace TissueForge {


   namespace py {


        struct CAPI_EXPORT CustomForcePy : CustomForce {
            PyObject *callable;

            CustomForcePy();
            CustomForcePy(const FVector3 &f, const FloatP_t &period=std::numeric_limits<FloatP_t>::max());

            /**
             * @brief Creates an instance from an underlying custom python function
             * 
             * @param f python function. Takes no arguments and returns a three-component vector. 
             * @param period period at which the force is updated. 
             */
            CustomForcePy(PyObject *f, const FloatP_t &period=std::numeric_limits<FloatP_t>::max());
            virtual ~CustomForcePy();

            void onTime(FloatP_t time);
            FVector3 getValue();

            void setValue(PyObject *_userFunc=NULL);

            /**
             * @brief Convert basic force to CustomForcePy. 
             * 
             * If the basic force is not a CustomForcePy, then NULL is returned. 
             * 
             * @param f 
             * @return CustomForcePy* 
             */
            static CustomForcePy *fromForce(Force *f);

        };

    };


    namespace io { 


        template <>
        HRESULT toFile(const py::CustomForcePy &dataElement, const MetaData &metaData, IOElement &fileElement);

        template <>
        HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, py::CustomForcePy *dataElement);

}};

#endif // _SOURCE_LANGS_PY_TFFORCEPY_H_