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

#ifndef _SOURCE_TYPES_TFVECTOR2_H_
#define _SOURCE_TYPES_TFVECTOR2_H_

#include "tfVector.h"

#include <Magnum/Math/Vector2.h>
#include <Magnum/Math/Distance.h>


namespace TissueForge::types {


    template<class T> using Vector2Base = Magnum::Math::Vector2<T>;
    
    template<typename T>
    class TVector2 : public Vector2Base<T> {
        public:
            constexpr static TVector2<T> xAxis(T length = T(1)) { return (TVector2<T>)Vector2Base<T>::xAxis(length); }

            constexpr static TVector2<T> yAxis(T length = T(1)) { return (TVector2<T>)Vector2Base<T>::yAxis(length); }

            constexpr static TVector2<T> xScale(T scale) { return (TVector2<T>)Vector2Base<T>::xScale(scale); }

            constexpr static TVector2<T> yScale(T scale) { return (TVector2<T>)Vector2Base<T>::yScale(scale); }

            constexpr TVector2() noexcept: Vector2Base<T>() {}

            constexpr explicit TVector2(T value) noexcept: Vector2Base<T>(value) {}

            constexpr TVector2(T x, T y) noexcept: Vector2Base<T>(x, y) {}

            template<class U> constexpr explicit TVector2(const TVector2<U>& other) noexcept: Vector2Base<T>(other) {}

            T& x() { return Vector2Base<T>::x(); }
            T& y() { return Vector2Base<T>::y(); }

            constexpr T x() const { return Vector2Base<T>::x(); }
            constexpr T y() const { return Vector2Base<T>::y(); }

            template<class U = T, typename std::enable_if<std::is_floating_point<U>::value, bool>::type = true>
            T distance(const TVector2<T> &lineStartPt, const TVector2<T> &lineEndPt) {
                return Magnum::Math::Distance::lineSegmentPoint(lineStartPt, lineEndPt, *this);
            }

            MAGNUM_BASE_VECTOR_CAST_METHODS(2, TVector2, Vector2Base)

            REVISED_MAGNUM_VECTOR_SUBCLASS_IMPLEMENTATION(2, TVector2, Vector2Base)

            #ifdef SWIGPYTHON
            SWIGPYTHON_MAGNUM_VECTOR_SUBCLASS_IMPLEMENTATION(2, TVector2)
            #endif
    };

}

TF_VECTOR_IMPL_OSTREAM(TissueForge::types::TVector2)

#endif // _SOURCE_TYPES_TFVECTOR2_H_
