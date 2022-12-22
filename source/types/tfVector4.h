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

#ifndef _SOURCE_TYPES_TFVECTOR4_H_
#define _SOURCE_TYPES_TFVECTOR4_H_

#include "tfVector3.h"
#include <Magnum/Math/Vector4.h>


namespace TissueForge::types {


    template<class T> using Vector4Base = Magnum::Math::Vector4<T>;
    
    template<typename T>
    class TVector4 : public Vector4Base<T> {
        public:
            /**
             * @brief Initialize a plane equation
             * 
             * @param normal plane normal
             * @param point point on the plane
             */
            template<class U = T, typename std::enable_if<std::is_floating_point<U>::value, bool>::type = true>
            static TVector4<T> planeEquation(const TVector3<T> &normal, const TVector3<T> &point) {
                return Magnum::Math::planeEquation(normal, point);
            }

            /**
             * @brief Initialize a plane equation
             * 
             * @param normal plane normal
             * @param point point on the plane
             */
            template<class U = T, typename std::enable_if<std::is_floating_point<U>::value, bool>::type = true>
            static TVector4<T> planeEquation(const TVector3<T>& p0, const TVector3<T>& p1, const TVector3<T>& p2) {
                return Magnum::Math::planeEquation(p0, p1, p2);
            }

            constexpr TVector4() noexcept: Vector4Base<T>() {}

            constexpr explicit TVector4(T value) noexcept: Vector4Base<T>(value) {}

            constexpr TVector4(T x, T y, T z, T w) noexcept: Vector4Base<T>(x, y, z, w) {}

            constexpr TVector4(const TVector3<T>& xyz, T w) noexcept: Vector4Base<T>(xyz[0], xyz[1], xyz[2], w) {}

            template<class U> constexpr explicit TVector4(const TVector4<U>& other) noexcept: Vector4Base<T>(other) {}

            /** Get the X (first) component */
            T& x() { return Vector4Base<T>::x(); }

            /** Get the Y (second) component */
            T& y() { return Vector4Base<T>::y(); }

            /** Get the Z (third) component */
            T& z() { return Vector4Base<T>::z(); }

            /** Get the W (fourth) component */
            T& w() { return Vector4Base<T>::w(); }

            /** Get the X (first) component */
            constexpr T x() const { return Vector4Base<T>::x(); }

            /** Get the Y (second) component */
            constexpr T y() const { return Vector4Base<T>::y(); }

            /** Get the Z (third) component */
            constexpr T z() const { return Vector4Base<T>::z(); }

            /** Get the W (fourth) component */
            constexpr T w() const { return Vector4Base<T>::w(); }

            /** Get the red (first) component */
            T& r() { return Vector4Base<T>::r(); }

            /** Get the green (second) component */
            T& g() { return Vector4Base<T>::g(); }

            /** Get the blue (third) component */
            T& b() { return Vector4Base<T>::b(); }

            /** Get the alpha (fourth) component */
            T& a() { return Vector4Base<T>::a(); }

            /** Get the red (first) component */
            constexpr T r() const { return Vector4Base<T>::r(); }

            /** Get the green (second) component */
            constexpr T g() const { return Vector4Base<T>::g(); }

            /** Get the blue (third) component */
            constexpr T b() const { return Vector4Base<T>::b(); }

            /** Get the alpha (fourth) component */
            constexpr T a() const { return Vector4Base<T>::a(); }

            /** Get the x-y-z (first-second-third) components */
            TVector3<T>& xyz() { return TVector3<T>::from(Vector4Base<T>::data()); }

            /** Get the x-y-z (first-second-third) components */
            constexpr const TVector3<T> xyz() const {
                return {Vector4Base<T>::_data[0], Vector4Base<T>::_data[1], Vector4Base<T>::_data[2]};
            }

            /** Get the red-green-blue (first-second-third) components */
            TVector3<T>& rgb() { return TVector3<T>::from(Vector4Base<T>::data()); }

            /** Get the red-green-blue (first-second-third) components */
            constexpr const TVector3<T> rgb() const {
                return {Vector4Base<T>::_data[0], Vector4Base<T>::_data[1], Vector4Base<T>::_data[2]};
            }

            /** Get the x-y (first-second) components */
            TVector2<T>& xy() { return TVector2<T>::from(Vector4Base<T>::data()); }

            /** Get the x-y (first-second) components */
            constexpr const TVector2<T> xy() const {
                return {Vector4Base<T>::_data[0], Vector4Base<T>::_data[1]};
            }

            /** Get the distance to a point */
            template<class U = T, typename std::enable_if<std::is_floating_point<U>::value, bool>::type = true>
            T distance(const TVector3<T> &point) const {
                return Magnum::Math::Distance::pointPlane(point, *this);
            }

            /** Get the distance to a point, scaled by the length of the plane normal */
            template<class U = T, typename std::enable_if<std::is_floating_point<U>::value, bool>::type = true>
            T distanceScaled(const TVector3<T> &point) const {
                return Magnum::Math::Distance::pointPlaneScaled(point, *this);
            }

            template<class U = T, typename std::enable_if<std::is_floating_point<U>::value, bool>::type = true>
            TVector3<T> shortestDisplacementFrom(const TVector3<T> &pt) const {
                return - distance(pt) * xyz().normalized();
            }

            MAGNUM_BASE_VECTOR_CAST_METHODS(4, TVector4, Vector4Base)

            REVISED_MAGNUM_VECTOR_SUBCLASS_IMPLEMENTATION(4, TVector4, Vector4Base)

            #ifdef SWIGPYTHON
            SWIGPYTHON_MAGNUM_VECTOR_SUBCLASS_IMPLEMENTATION(4, TVector4)
            #endif
    };

}

TF_VECTOR_IMPL_OSTREAM(TissueForge::types::TVector4)

#endif // _SOURCE_TYPES_TFVECTOR4_H_