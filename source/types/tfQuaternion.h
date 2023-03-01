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

#ifndef _SOURCE_TYPES_TFQUATERNION_H_
#define _SOURCE_TYPES_TFQUATERNION_H_

#include <Magnum/Math/Quaternion.h>

#include "tfVector3.h"
#include "tfMatrix3.h"


namespace TissueForge::types {


    template<class T> using QuaternionBase = Magnum::Math::Quaternion<T>;

    template<class T>
    class TQuaternion : public QuaternionBase<T> {
        public:
            /** Initialize from a rotation angle and normalized axis of rotation */
            static TQuaternion<T> rotation(T angle, const TVector3<T>& normalizedAxis) {
                return QuaternionBase<T>::rotation(Magnum::Math::Rad<T>(angle), normalizedAxis);
            }

            /** Initialize from an orthogonal rotation matrix */
            static TQuaternion<T> fromMatrix(const TMatrix3<T>& matrix) {
                return QuaternionBase<T>::fromMatrix((const Magnum::Math::Matrix<3, T>&)matrix);
            }

            constexpr TQuaternion() noexcept: QuaternionBase<T>() {}
            constexpr TQuaternion(const TVector3<T>& vector, T scalar) noexcept: QuaternionBase<T>(vector, scalar) {}
            constexpr TQuaternion(T scalar) noexcept: TQuaternion<T>(TVector3<T>(0), scalar) {}
            constexpr explicit TQuaternion(const TVector3<T>& vector) noexcept: QuaternionBase<T>(vector) {}
            template<class U> constexpr explicit TQuaternion(const TQuaternion<U>& other) noexcept: QuaternionBase<T>(other) {}

            template<class U = T> TQuaternion(const QuaternionBase<U>& other) noexcept: QuaternionBase<T>(other) {}
            template<class U = T> TQuaternion(const QuaternionBase<U>* other) noexcept: QuaternionBase<T>(*other) {}
            template<class U = T> operator QuaternionBase<U>*() { return static_cast<QuaternionBase<U>*>(this); }
            template<class U = T> operator const QuaternionBase<U>*() { return static_cast<const QuaternionBase<U>*>(this); }
            template<class U = T> operator QuaternionBase<U>() { return static_cast<QuaternionBase<U>>(*this); }
            template<class U = T> operator const QuaternionBase<U>() const { return static_cast<const QuaternionBase<U>>(*this); }

            /** Get the underlying array */
            T* data() { return QuaternionBase<T>::data(); }

            /** Get the underlying array */
            constexpr const T* data() const { return QuaternionBase<T>::data(); }
            bool operator==(const TQuaternion<T>& other) const { return QuaternionBase<T>::operator==(other); }
            bool operator!=(const TQuaternion<T>& other) const { return QuaternionBase<T>::operator!=(other); }

            /** Test whether the quaternion is normalized */
            bool isNormalized() const { return QuaternionBase<T>::isNormalized(); }
            #ifndef SWIGPYTHON

            /** Get the 3-component vector */
            TVector3<T>& vector() { return (TVector3<T>&)TQuaternion<T>::vector(); }

            /** Get the scalar */
            T& scalar() { return QuaternionBase<T>::scalar(); }
            #endif

            /** Get the 3-component vector */
            constexpr const TVector3<T> vector() const { return QuaternionBase<T>::vector(); }

            /** Get the scalar */
            constexpr T scalar() const { return QuaternionBase<T>::scalar(); }

            /** Get the angle. Quaternion should be normalized. */
            T angle() const { return T(QuaternionBase<T>::angle()); }

            /** Get the angle */
            T angle(const TQuaternion& other) const { return T(Magnum::Math::angle(this->normalized(), other.normalized())); }

            /** Get the rotation axis. Quaternion should be normalized. */
            TVector3<T> axis() const { return QuaternionBase<T>::axis(); }

            /** Get the rotation matrix */
            TMatrix3<T> toMatrix() const { return QuaternionBase<T>::toMatrix(); }

            /** Get the Euler angles */
            TVector3<T> toEuler() const {
                auto v = QuaternionBase<T>::toEuler();
                return TVector3<T>(T(v[0]), T(v[1]), T(v[2]));;
            }
            TQuaternion<T> operator-() const { return QuaternionBase<T>::operator-(); }
            TQuaternion<T>& operator+=(const TQuaternion<T>& other) { return (TQuaternion<T>&)QuaternionBase<T>::operator+=(other); }
            TQuaternion<T> operator+(const TQuaternion<T>& other) const { return QuaternionBase<T>::operator+(other); }
            TQuaternion<T>& operator-=(const TQuaternion<T>& other) { return (TQuaternion<T>&)QuaternionBase<T>::operator-=(other); }
            TQuaternion<T> operator-(const TQuaternion<T>& other) const { return QuaternionBase<T>::operator-(other); }
            TQuaternion<T>& operator*=(T scalar) { return (TQuaternion<T>&)QuaternionBase<T>::operator*=(scalar); }
            TQuaternion<T> operator*(T scalar) const { return QuaternionBase<T>::operator*(scalar); }
            TQuaternion<T>& operator/=(T scalar) { return (TQuaternion<T>&)QuaternionBase<T>::operator/=(scalar); }
            TQuaternion<T> operator/(T scalar) const { return QuaternionBase<T>::operator/(scalar); }
            TQuaternion<T> operator*(const QuaternionBase<T>& other) const { return QuaternionBase<T>::operator*(other); }

            /** Get the squared length */
            T dot() const { return QuaternionBase<T>::dot(); }

            /** Get the length */
            T length() const { return QuaternionBase<T>::length(); }

            /** Get the normalized quaternion */
            TQuaternion<T> normalized() const { return QuaternionBase<T>::normalized(); }

            /** Get the conjugated quaternion */
            TQuaternion<T> conjugated() const { return QuaternionBase<T>::conjugated(); }

            /** Get the inverted quaternion */
            TQuaternion<T> inverted() const { return QuaternionBase<T>::inverted(); }

            /** Get the inverted and normalized quaternion */
            TQuaternion<T> invertedNormalized() const { return QuaternionBase<T>::invertedNormalized(); }

            /** Transform a vector */
            TVector3<T> transformVector(const TVector3<T>& vector) const { return QuaternionBase<T>::transformVector(vector); }

            /** Transform a normalized vector */
            TVector3<T> transformVectorNormalized(const TVector3<T>& vector) const { return QuaternionBase<T>::transformVectorNormalized(vector); }

            operator std::vector<T>&() const {
                std::vector<T>& v = vector();
                std::vector<T> *result = new std::vector<T>(v.begin(), v.end());
                result->push_back(scalar());
                return *result;
            }
            #ifdef SWIGPYTHON
            constexpr explicit TQuaternion(const std::vector<T>& vector, T scalar) noexcept: QuaternionBase<T>(TVector3<T>(vector), scalar) {}

            /** Get the quaternion as a vector */
            std::vector<T>& asVector() { 
                std::vector<T> *result = new std::vector<T>(*this);
                return *result;
            }
            #endif // SWIGPYTHON
    };

}

template<typename T>
inline std::ostream& operator<<(std::ostream& os, const TissueForge::types::TQuaternion<T>& q)
{
    auto vec = q.vector();
    os << std::string("{") << vec[0];
    for(int i = 1; i < vec.length(); ++i) os << std::string(",") << vec[i];
    os << "," << q.scalar() << std::string("}");
    return os;
}

#endif // _SOURCE_TYPES_TFQUATERNION_H_