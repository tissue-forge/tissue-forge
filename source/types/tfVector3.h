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

#ifndef _SOURCE_TYPES_TFVECTOR3_H_
#define _SOURCE_TYPES_TFVECTOR3_H_

#include "tfVector2.h"

#include <Magnum/Math/Vector3.h>
#include <Magnum/Math/Distance.h>


namespace TissueForge::types {


    template<class T> using Vector3Base = Magnum::Math::Vector3<T>;
    
    template<typename T>
    class TVector3 : public Vector3Base<T> {
        public:
            constexpr static TVector3<T> xAxis(T length = T(1)) { return (TVector3<T>)Vector3Base<T>::xAxis(length); }

            constexpr static TVector3<T> yAxis(T length = T(1)) { return (TVector3<T>)Vector3Base<T>::yAxis(length); }

            constexpr static TVector3<T> zAxis(T length = T(1)) { return (TVector3<T>)Vector3Base<T>::zAxis(length); }

            constexpr static TVector3<T> xScale(T scale) { return (TVector3<T>)Vector3Base<T>::xScale(scale); }

            constexpr static TVector3<T> yScale(T scale) { return (TVector3<T>)Vector3Base<T>::yScale(scale); }

            constexpr static TVector3<T> zScale(T scale) { return (TVector3<T>)Vector3Base<T>::zScale(scale); }

            constexpr TVector3() noexcept: Vector3Base<T>() {}

            constexpr explicit TVector3(T value) noexcept: Vector3Base<T>(value) {}

            constexpr TVector3(T x, T y, T z) noexcept: Vector3Base<T>(x, y, z) {}

            constexpr TVector3(const TVector2<T>& xy, T z) noexcept: Vector3Base<T>(xy[0], xy[1], z) {}

            template<class U> constexpr explicit TVector3(const TVector3<U>& other) noexcept: Vector3Base<T>(other) {}

            T& x() { return Vector3Base<T>::x(); }
            T& y() { return Vector3Base<T>::y(); }
            T& z() { return Vector3Base<T>::z(); }

            constexpr T x() const { return Vector3Base<T>::x(); }
            constexpr T y() const { return Vector3Base<T>::y(); }
            constexpr T z() const { return Vector3Base<T>::z(); }

            T& r() { return Vector3Base<T>::r(); }
            T& g() { return Vector3Base<T>::g(); }
            T& b() { return Vector3Base<T>::b(); }

            constexpr T r() const { return Vector3Base<T>::r(); }
            constexpr T g() const { return Vector3Base<T>::g(); }
            constexpr T b() const { return Vector3Base<T>::b(); }

            TVector2<T>& xy() { return TVector2<T>::from(Vector3Base<T>::data()); }
            constexpr const TVector2<T> xy() const {
                return {Vector3Base<T>::_data[0], Vector3Base<T>::_data[1]};
            }

            template<class U = T, typename std::enable_if<std::is_floating_point<U>::value, bool>::type = true>
            T distance(const TVector3<T> &lineStartPt, const TVector3<T> &lineEndPt) {
                return Magnum::Math::Distance::lineSegmentPoint(lineStartPt, lineEndPt, *this);
            }

            template<class U = T, typename std::enable_if<std::is_floating_point<U>::value, bool>::type = true>
            TVector3<T> relativeTo(const TVector3<T> &origin, const TVector3<T> &dim, const bool &periodic_x, const bool &periodic_y, const bool &periodic_z) {
                TVector3<T> result = *this - origin;
                TVector3<T> crit = dim / 2.0;
                if(periodic_x) {
                    if(result[0] < -crit[0]) result[0] += dim[0];
                    else if(result[0] > crit[0]) result[0] -= dim[0];
                }
                if(periodic_y) {
                    if(result[1] < -crit[1]) result[1] += dim[1];
                    else if(result[1] > crit[1]) result[1] -= dim[1];
                }
                if(periodic_z) {
                    if(result[2] < -crit[2]) result[2] += dim[2];
                    else if(result[2] > crit[2]) result[2] -= dim[2];
                }
                return result;
            }

            MAGNUM_BASE_VECTOR_CAST_METHODS(3, TVector3, Vector3Base)

            REVISED_MAGNUM_VECTOR_SUBCLASS_IMPLEMENTATION(3, TVector3, Vector3Base)

            #ifdef SWIGPYTHON
            SWIGPYTHON_MAGNUM_VECTOR_SUBCLASS_IMPLEMENTATION(3, TVector3)
            #endif
    };

}

TF_VECTOR_IMPL_OSTREAM(TissueForge::types::TVector3)

#endif // _SOURCE_TYPES_TFVECTOR3_H_