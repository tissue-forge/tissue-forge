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
            /** Initialize a translation matrix from a translation vector */
            constexpr static TMatrix4<T> translation(const TVector3<T>& vector) {
                return Matrix4Base<T>::translation(vector);
            }

            /** Initialize a scaling matrix from a scaling vector */
            constexpr static TMatrix4<T> scaling(const TVector3<T>& vector) {
                return Matrix4Base<T>::scaling(vector);
            }

            /** Initialize a rotation matrix from a rotation angle and normalized axis of rotation*/
            static TMatrix4<T> rotation(T angle, const TVector3<T>& normalizedAxis) {
                return Matrix4Base<T>::rotation(Magnum::Math::Rad<T>(angle), normalizedAxis);
            }

            /** Initialize a rotation matrix about the X (first) axis */
            static TMatrix4<T> rotationX(T angle) {
                return Matrix4Base<T>::rotationX(Magnum::Math::Rad<T>(angle));
            }

            /** Initialize a rotation matrix about the Y (second) axis */
            static TMatrix4<T> rotationY(T angle) {
                return Matrix4Base<T>::rotationY(Magnum::Math::Rad<T>(angle));
            }

            /** Initialize a rotation matrix about the Z (third) axis */
            static TMatrix4<T> rotationZ(T angle) {
                return Matrix4Base<T>::rotationZ(Magnum::Math::Rad<T>(angle));
            }

            /** Initialize a reflection matrix about a plane using the normalized plane normal */
            static TMatrix4<T> reflection(const TVector3<T>& normal) {
                return Matrix4Base<T>::reflection(normal);
            }

            /** Initialize a shearing matrix about the X (first) and Y (second) axes */
            constexpr static TMatrix4<T> shearingXY(T amountX, T amountY) {
                return Matrix4Base<T>::shearingXY(amountX, amountY);
            }

            /** Initialize a shearing matrix about the X (first) and Z (third) axes */
            constexpr static TMatrix4<T> shearingXZ(T amountX, T amountZ) {
                return Matrix4Base<T>::shearingXZ(amountX, amountZ);
            }

            /** Initialize a shearing matrix about the Y (second) and Z (third) axes */
            constexpr static TMatrix4<T> shearingYZ(T amountY, T amountZ) {
                return Matrix4Base<T>::shearingYZ(amountY, amountZ);
            }

            /**
             * @brief Initialize an orthographic projection matrix
             * 
             * @param size size of the view
             * @param near distance to near clipping plane
             * @param far distance for far clipping plane
             */
            static TMatrix4<T> orthographicProjection(const TVector2<T>& size, T near, T far) {
                return Matrix4Base<T>::orthographicProjection(size, near, far);
            }

            /**
             * @brief Initialize an perspective projection matrix
             * 
             * @param size size of the near clipping plane
             * @param near distance to near clipping plane
             * @param far distance for far clipping plane
             */
            static TMatrix4<T> perspectiveProjection(const TVector2<T>& size, T near, T far) {
                return Matrix4Base<T>::perspectiveProjection(size, near, far);
            }

            /**
             * @brief Initialize a perspective projection matrix
             * 
             * @param fov horizontal angle of the field of view
             * @param aspectRatio horizontal:vertical field of view aspect ratio
             * @param near distance to near clipping plane
             * @param far distance for far clipping plane
             */
            static TMatrix4<T> perspectiveProjection(T fov, T aspectRatio, T near, T far) {
                return Matrix4Base<T>::perspectiveProjection(Magnum::Math::Rad<T>(fov), aspectRatio, near, far);
            }

            /**
             * @brief Initialize a perspective projection matrix
             * 
             * @param bottomLeft bottom-left point of field of view
             * @param topRight top-right point of field of view
             * @param near distance to near clipping plane
             * @param far distance for far clipping plane
             */
            static TMatrix4<T> perspectiveProjection(const TVector2<T>& bottomLeft, const TVector2<T>& topRight, T near, T far) {
                return Matrix4Base<T>::perspectiveProjection(bottomLeft, topRight, near, far);
            }

            /**
             * @brief Initialize a matrix oriented towards a point
             * 
             * @param eye location of view
             * @param target location of target point
             * @param up upward-normal of view
             */
            static TMatrix4<T> lookAt(const TVector3<T>& eye, const TVector3<T>& target, const TVector3<T>& up) {
                return Matrix4Base<T>::lookAt(eye, target, up);
            }

            /** Initialize from an array */
            constexpr static TMatrix4<T> from(const TMatrix3<T>& rotationScaling, const TVector3<T>& translation) {
                return Matrix4Base<T>::from((const Magnum::Math::Matrix3<T>&)rotationScaling, translation);
            }

            constexpr TMatrix4() noexcept: Matrix4Base<T>() {}
            constexpr TMatrix4(const TVector4<T>& first, const TVector4<T>& second, const TVector4<T>& third, const TVector4<T>& fourth) noexcept: 
                Matrix4Base<T>(first, second, third, fourth) {}
            constexpr explicit TMatrix4(T value) noexcept: Matrix4Base<T>{value} {}
            template<std::size_t otherSize> constexpr explicit TMatrix4(const TMatrixS<otherSize, T>& other) noexcept: Matrix4Base<T>{other} {}
            constexpr TMatrix4(const TMatrix4<T>& other) noexcept: Matrix4Base<T>(other) {}

            /** Test whether the matrix is a rigid transformation */
            bool isRigidTransformation() const { return Matrix4Base<T>::isRigidTransformation(); }

            /** Get the rotation and scaling matrix */
            constexpr TMatrix3<T> rotationScaling() const { return Matrix4Base<T>::rotationScaling(); }

            /** Get the rotation and shear matrix */
            TMatrix3<T> rotationShear() const { return Matrix4Base<T>::rotationShear(); }

            /** Get the rotation matrix */
            TMatrix3<T> rotation() const { return Matrix4Base<T>::rotation(); }

            /** Get the normalized rotation matrix */
            TMatrix3<T> rotationNormalized() const { return Matrix4Base<T>::rotationNormalized(); }

            /** Get the squared scaling vector */
            TVector3<T> scalingSquared() const { return Matrix4Base<T>::scalingSquared(); }

            /** Get the scaling vector */
            TVector3<T> scaling() const { return Matrix4Base<T>::scaling(); }

            /** Get the uniform squared scaling vector */
            T uniformScalingSquared() const { return Matrix4Base<T>::uniformScalingSquared(); }

            /** Get the uniform scaling vector */
            T uniformScaling() const { return Matrix4Base<T>::uniformScaling(); }

            /** Get the normal matrix */
            TMatrix3<T> normalMatrix() const { return Matrix4Base<T>::normalMatrix(); }

            /** Get the rightward-pointing vector */
            TVector3<T>& right() { return (TVector3<T>&)Matrix4Base<T>::right(); }

            /** Get the rightward-pointing vector */
            constexpr TVector3<T> right() const { return Matrix4Base<T>::right(); }

            /** Get the upward-pointing vector */
            TVector3<T>& up() { return (TVector3<T>&)Matrix4Base<T>::up(); }

            /** Get the upward-pointing vector */
            constexpr TVector3<T> up() const { return Matrix4Base<T>::up(); }

            /** Get the backward-pointing vector */
            TVector3<T>& backward() { return (TVector3<T>&)Matrix4Base<T>::backward(); }

            /** Get the backward-pointing vector */
            constexpr TVector3<T> backward() const { return Matrix4Base<T>::backward(); }

            /** Get the translation vector */
            TVector3<T>& translation() { return (TVector3<T>&)Matrix4Base<T>::translation(); }

            /** Get the translation vector */
            constexpr TVector3<T> translation() const { return Matrix4Base<T>::translation(); }

            /** Get the inverted rigid transformation matrix. Must be a rigid transformation matrix. */
            TMatrix4<T> invertedRigid() const { return Matrix4Base<T>::invertedRigid(); }

            /** Transform a vector */
            TVector3<T> transformVector(const TVector3<T>& vector) const { return Matrix4Base<T>::transformVector(vector); }

            /** Transform a point */
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