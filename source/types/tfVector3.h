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

            /** Get the X (first) component */
            T& x() { return Vector3Base<T>::x(); }

            /** Get the Y (second) component */
            T& y() { return Vector3Base<T>::y(); }

            /** Get the Z (third) component */
            T& z() { return Vector3Base<T>::z(); }

            /** Get the X (first) component */
            constexpr T x() const { return Vector3Base<T>::x(); }

            /** Get the Y (second) component */
            constexpr T y() const { return Vector3Base<T>::y(); }

            /** Get the Z (third) component */
            constexpr T z() const { return Vector3Base<T>::z(); }

            /** Get the red (first) component */
            T& r() { return Vector3Base<T>::r(); }

            /** Get the green (second) component */
            T& g() { return Vector3Base<T>::g(); }

            /** Get the blue (third) component */
            T& b() { return Vector3Base<T>::b(); }

            /** Get the red (first) component */
            constexpr T r() const { return Vector3Base<T>::r(); }

            /** Get the green (second) component */
            constexpr T g() const { return Vector3Base<T>::g(); }

            /** Get the blue (third) component */
            constexpr T b() const { return Vector3Base<T>::b(); }

            /** Get the x-y (first-second) components */
            TVector2<T>& xy() { return TVector2<T>::from(Vector3Base<T>::data()); }

            /** Get the x-y (first-second) components */
            constexpr const TVector2<T> xy() const {
                return {Vector3Base<T>::_data[0], Vector3Base<T>::_data[1]};
            }

            /** Get the distance to a line defined by two points */
            template<class U = T, typename std::enable_if<std::is_floating_point<U>::value, bool>::type = true>
            T distance(const TVector3<T> &lineStartPt, const TVector3<T> &lineEndPt) {
                return Magnum::Math::Distance::lineSegmentPoint(lineStartPt, lineEndPt, *this);
            }

            /**
             * @brief Get the position relative to a point
             * 
             * @param origin target point
             * @param dim dimensions of domain
             * @param periodic_x flag for whether to apply periodic boundary conditions along the X- (first-) direction
             * @param periodic_y flag for whether to apply periodic boundary conditions along the Y- (second-) direction
             * @param periodic_z flag for whether to apply periodic boundary conditions along the Z- (third-) direction
             */
            template<class U = T, typename std::enable_if<std::is_floating_point<U>::value, bool>::type = true>
            TVector3<T> lineShortestDisplacementTo(const TVector3<T> &lineStartPt, const TVector3<T> &lineEndPt) const {
                const TVector3<T> lineDir = (lineEndPt - lineStartPt).normalized();
                return lineStartPt + (*this - lineStartPt).dot(lineDir) * lineDir - *this;
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

            /** Get the cross product with another vector */
            TVector3<T> cross(const TVector3<T> &other) {
                return Magnum::Math::cross(*this, other);
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