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

#ifndef _SOURCE_TYPES_TFMATRIX4_H_
#define _SOURCE_TYPES_TFMATRIX4_H_

#include "tfMatrix3.h"
#include "tfVector4.h"

#include <Magnum/Math/Matrix4.h>


namespace TissueForge::types {


    template<class T> using Matrix4Base = Magnum::Math::Matrix4<T>;
    
    template<class T>
    class TMatrix4 : public Matrix4Base<T> {
        public:
            constexpr static TMatrix4<T> translation(const TVector3<T>& vector) {
                return Matrix4Base<T>::translation(vector);
            }
            constexpr static TMatrix4<T> scaling(const TVector3<T>& vector) {
                return Matrix4Base<T>::scaling(vector);
            }
            static TMatrix4<T> rotation(T angle, const TVector3<T>& normalizedAxis) {
                return Matrix4Base<T>::rotation(Magnum::Math::Rad<T>(angle), normalizedAxis);
            }
            static TMatrix4<T> rotationX(T angle) {
                return Matrix4Base<T>::rotationX(Magnum::Math::Rad<T>(angle));
            }
            static TMatrix4<T> rotationY(T angle) {
                return Matrix4Base<T>::rotationY(Magnum::Math::Rad<T>(angle));
            }
            static TMatrix4<T> rotationZ(T angle) {
                return Matrix4Base<T>::rotationZ(Magnum::Math::Rad<T>(angle));
            }
            static TMatrix4<T> reflection(const TVector3<T>& normal) {
                return Matrix4Base<T>::reflection(normal);
            }
            constexpr static TMatrix4<T> shearingXY(T amountX, T amountY) {
                return Matrix4Base<T>::shearingXY(amountX, amountY);
            }
            constexpr static TMatrix4<T> shearingXZ(T amountX, T amountZ) {
                return Matrix4Base<T>::shearingXZ(amountX, amountZ);
            }
            constexpr static TMatrix4<T> shearingYZ(T amountY, T amountZ) {
                return Matrix4Base<T>::shearingYZ(amountY, amountZ);
            }
            static TMatrix4<T> orthographicProjection(const TVector2<T>& size, T near, T far) {
                return Matrix4Base<T>::orthographicProjection(size, near, far);
            }
            static TMatrix4<T> perspectiveProjection(const TVector2<T>& size, T near, T far) {
                return Matrix4Base<T>::perspectiveProjection(size, near, far);
            }
            static TMatrix4<T> perspectiveProjection(T fov, T aspectRatio, T near, T far) {
                return Matrix4Base<T>::perspectiveProjection(Magnum::Math::Rad<T>(fov), aspectRatio, near, far);
            }
            static TMatrix4<T> perspectiveProjection(const TVector2<T>& bottomLeft, const TVector2<T>& topRight, T near, T far) {
                return Matrix4Base<T>::perspectiveProjection(bottomLeft, topRight, near, far);
            }
            static TMatrix4<T> lookAt(const TVector3<T>& eye, const TVector3<T>& target, const TVector3<T>& up) {
                return Matrix4Base<T>::lookAt(eye, target, up);
            }
            constexpr static TMatrix4<T> from(const TMatrix3<T>& rotationScaling, const TVector3<T>& translation) {
                return Matrix4Base<T>::from((const Magnum::Math::Matrix3<T>&)rotationScaling, translation);
            }

            constexpr TMatrix4() noexcept: Matrix4Base<T>() {}
            constexpr TMatrix4(const TVector4<T>& first, const TVector4<T>& second, const TVector4<T>& third, const TVector4<T>& fourth) noexcept: 
                Matrix4Base<T>(first, second, third, fourth) {}
            constexpr explicit TMatrix4(T value) noexcept: Matrix4Base<T>{value} {}
            template<std::size_t otherSize> constexpr explicit TMatrix4(const TMatrixS<otherSize, T>& other) noexcept: Matrix4Base<T>{other} {}
            constexpr TMatrix4(const TMatrix4<T>& other) noexcept: Matrix4Base<T>(other) {}

            bool isRigidTransformation() const { return Matrix4Base<T>::isRigidTransformation(); }
            constexpr TMatrix3<T> rotationScaling() const { return Matrix4Base<T>::rotationScaling(); }
            TMatrix3<T> rotationShear() const { return Matrix4Base<T>::rotationShear(); }
            TMatrix3<T> rotation() const { return Matrix4Base<T>::rotation(); }
            TMatrix3<T> rotationNormalized() const { return Matrix4Base<T>::rotationNormalized(); }
            TVector3<T> scalingSquared() const { return Matrix4Base<T>::scalingSquared(); }
            TVector3<T> scaling() const { return Matrix4Base<T>::scaling(); }
            T uniformScalingSquared() const { return Matrix4Base<T>::uniformScalingSquared(); }
            T uniformScaling() const { return Matrix4Base<T>::uniformScaling(); }
            TMatrix3<T> normalMatrix() const { return Matrix4Base<T>::normalMatrix(); }
            TVector3<T>& right() { return (TVector3<T>&)Matrix4Base<T>::right(); }
            constexpr TVector3<T> right() const { return Matrix4Base<T>::right(); }
            TVector3<T>& up() { return (TVector3<T>&)Matrix4Base<T>::up(); }
            constexpr TVector3<T> up() const { return Matrix4Base<T>::up(); }
            TVector3<T>& backward() { return (TVector3<T>&)Matrix4Base<T>::backward(); }
            constexpr TVector3<T> backward() const { return Matrix4Base<T>::backward(); }
            TVector3<T>& translation() { return (TVector3<T>&)Matrix4Base<T>::translation(); }
            constexpr TVector3<T> translation() const { return Matrix4Base<T>::translation(); }
            TMatrix4<T> invertedRigid() const { return Matrix4Base<T>::invertedRigid(); }
            TVector3<T> transformVector(const TVector3<T>& vector) const { return Matrix4Base<T>::transformVector(vector); }
            TVector3<T> transformPoint(const TVector3<T>& vector) const { return Matrix4Base<T>::transformPoint(vector); }

            MAGNUM_BASE_MATRIX_CAST_METHODS(4, TMatrix4, Matrix4Base)

            REVISED_MAGNUM_MATRIX_SUBCLASS_IMPLEMENTATION(4, TMatrix4, Matrix4Base, TVector4)

            #ifdef SWIGPYTHON
            SWIGPYTHON_MAGNUM_MATRIX_SUBCLASS_IMPLEMENTATION(4, TMatrix4, TVector4)
            #endif

    };

}

TF_MATRIX_IMPL_OSTREAM(TissueForge::types::TMatrix4)

#endif // _SOURCE_TYPES_TFMATRIX4_H_