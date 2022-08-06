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
            template<class U = T, typename std::enable_if<std::is_floating_point<U>::value, bool>::type = true>
            static TVector4<T> planeEquation(const TVector3<T> &normal, const TVector3<T> &point) {
                return Magnum::Math::planeEquation(normal, point);
            }
            template<class U = T, typename std::enable_if<std::is_floating_point<U>::value, bool>::type = true>
            static TVector4<T> planeEquation(const TVector3<T>& p0, const TVector3<T>& p1, const TVector3<T>& p2) {
                return Magnum::Math::planeEquation(p0, p1, p2);
            }

            constexpr TVector4() noexcept: Vector4Base<T>() {}

            constexpr explicit TVector4(T value) noexcept: Vector4Base<T>(value) {}

            constexpr TVector4(T x, T y, T z, T w) noexcept: Vector4Base<T>(x, y, z, w) {}

            constexpr TVector4(const TVector3<T>& xyz, T w) noexcept: Vector4Base<T>(xyz[0], xyz[1], xyz[2], w) {}

            template<class U> constexpr explicit TVector4(const TVector4<U>& other) noexcept: Vector4Base<T>(other) {}

            T& x() { return Vector4Base<T>::x(); }
            T& y() { return Vector4Base<T>::y(); }
            T& z() { return Vector4Base<T>::z(); }
            T& w() { return Vector4Base<T>::w(); }

            constexpr T x() const { return Vector4Base<T>::x(); }
            constexpr T y() const { return Vector4Base<T>::y(); }
            constexpr T z() const { return Vector4Base<T>::z(); }
            constexpr T w() const { return Vector4Base<T>::w(); }

            T& r() { return Vector4Base<T>::r(); }
            T& g() { return Vector4Base<T>::g(); }
            T& b() { return Vector4Base<T>::b(); }
            T& a() { return Vector4Base<T>::a(); }

            constexpr T r() const { return Vector4Base<T>::r(); }
            constexpr T g() const { return Vector4Base<T>::g(); }
            constexpr T b() const { return Vector4Base<T>::b(); }
            constexpr T a() const { return Vector4Base<T>::a(); }

            TVector3<T>& xyz() { return TVector3<T>::from(Vector4Base<T>::data()); }
            constexpr const TVector3<T> xyz() const {
                return {Vector4Base<T>::_data[0], Vector4Base<T>::_data[1], Vector4Base<T>::_data[2]};
            }

            TVector3<T>& rgb() { return TVector3<T>::from(Vector4Base<T>::data()); }
            constexpr const TVector3<T> rgb() const {
                return {Vector4Base<T>::_data[0], Vector4Base<T>::_data[1], Vector4Base<T>::_data[2]};
            }

            TVector2<T>& xy() { return TVector2<T>::from(Vector4Base<T>::data()); }
            constexpr const TVector2<T> xy() const {
                return {Vector4Base<T>::_data[0], Vector4Base<T>::_data[1]};
            }

            template<class U = T, typename std::enable_if<std::is_floating_point<U>::value, bool>::type = true>
            T distance(const TVector3<T> &point) const {
                return Magnum::Math::Distance::pointPlane(point, *this);
            }
            template<class U = T, typename std::enable_if<std::is_floating_point<U>::value, bool>::type = true>
            T distanceScaled(const TVector3<T> &point) const {
                return Magnum::Math::Distance::pointPlaneScaled(point, *this);
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