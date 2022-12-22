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

#ifndef _SOURCE_TYPES_TFMATRIX3_H_
#define _SOURCE_TYPES_TFMATRIX3_H_

#include "tfMatrix.h"
#include "tfVector3.h"

#include <Magnum/Math/Matrix3.h>


namespace TissueForge::types {

    
    template<class T> using Matrix3Base = Magnum::Math::Matrix3<T>;
    
    template<class T>
    class TMatrix3 : public Matrix3Base<T> {
        public:
            /** Initialize a rotation matrix */
            static TMatrix3<T> rotation(T angle) { return (TMatrix3<T>)Matrix3Base<T>::rotation(Magnum::Math::Rad<T>(angle)); }

            /** Initialize a shearing matrix along the X (first) direction */
            constexpr static TMatrix3<T> shearingX(T amount) { return (TMatrix3<T>)Matrix3Base<T>::shearingX(amount); }

            /** Initialize a shearing matrix along the Y (second) direction */
            constexpr static TMatrix3<T> shearingY(T amount) { return (TMatrix3<T>)Matrix3Base<T>::shearingY(amount); }

            constexpr TMatrix3() noexcept: Matrix3Base<T>() {}
            
            constexpr TMatrix3(const TVector3<T>& first, const TVector3<T>& second, const TVector3<T>& third) noexcept: 
                Matrix3Base<T>(Magnum::Math::Vector3<T>(first), Magnum::Math::Vector3<T>(second), Magnum::Math::Vector3<T>(third)) {}

            constexpr explicit TMatrix3(T value) noexcept: Matrix3Base<T>(value) {}

            template<class U> constexpr explicit TMatrix3(const TMatrix3<U>& other) noexcept: Matrix3Base<T>((Matrix3Base<U>)other) {}

            template<std::size_t otherSize> constexpr explicit TMatrix3(const TMatrixS<otherSize, T>& other) noexcept: Matrix3Base<T>{(TMatrixS<otherSize, T>)other} {}

            /** Test whether the matrix is a rigid transformation */
            bool isRigidTransformation() const { return Matrix3Base<T>::isRigidTransformation(); }

            /** Get the inverted rigid transformation. Must be a rigid transformation. */
            TMatrix3<T> invertedRigid() const { return (TMatrix3<T>)Matrix3Base<T>::invertedRigid(); }

            TMatrix3(const std::vector<T> &v1, const std::vector<T> &v2, const std::vector<T> &v3) : 
                TMatrix3<T>(TVector3<T>(v1), TVector3<T>(v2), TVector3<T>(v3)) {}

            MAGNUM_BASE_MATRIX_CAST_METHODS(3, TMatrix3, Matrix3Base)

            REVISED_MAGNUM_MATRIX_SUBCLASS_IMPLEMENTATION(3, TMatrix3, Matrix3Base, TVector3)

            #ifdef SWIGPYTHON
            SWIGPYTHON_MAGNUM_MATRIX_SUBCLASS_IMPLEMENTATION(3, TMatrix3, TVector3)
            #endif

    };

}

TF_MATRIX_IMPL_OSTREAM(TissueForge::types::TMatrix3)

#endif // _SOURCE_TYPES_TFMATRIX3_H_