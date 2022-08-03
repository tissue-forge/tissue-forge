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

#ifndef _SOURCE_TYPES_TF_TYPES_H_
#define _SOURCE_TYPES_TF_TYPES_H_

#include <tf_config.h>

#include "tfVector.h"
#include "tfVector2.h"
#include "tfVector3.h"
#include "tfVector4.h"
#include "tfMatrix.h"
#include "tfMatrix3.h"
#include "tfMatrix4.h"
#include "tfQuaternion.h"


namespace TissueForge {


    typedef types::TVector2<double> dVector2;
    typedef types::TVector3<double> dVector3;
    typedef types::TVector4<double> dVector4;

    typedef types::TVector2<float> fVector2;
    typedef types::TVector3<float> fVector3;
    typedef types::TVector4<float> fVector4;

    typedef types::TVector2<FloatP_t> FVector2;
    typedef types::TVector3<FloatP_t> FVector3;
    typedef types::TVector4<FloatP_t> FVector4;

    typedef types::TVector2<int> iVector2;
    typedef types::TVector3<int> iVector3;
    typedef types::TVector4<int> iVector4;

    typedef types::TVector2<unsigned int> uiVector2;
    typedef types::TVector3<unsigned int> uiVector3;
    typedef types::TVector4<unsigned int> uiVector4;

    typedef types::TMatrix3<double> dMatrix3;
    typedef types::TMatrix4<double> dMatrix4;

    typedef types::TMatrix3<float> fMatrix3;
    typedef types::TMatrix4<float> fMatrix4;

    typedef types::TMatrix3<FloatP_t> FMatrix3;
    typedef types::TMatrix4<FloatP_t> FMatrix4;

    typedef types::TQuaternion<double> dQuaternion;
    typedef types::TQuaternion<float> fQuaternion;
    typedef types::TQuaternion<FloatP_t> FQuaternion;

};

#endif // _SOURCE_TYPES_TF_TYPES_H_