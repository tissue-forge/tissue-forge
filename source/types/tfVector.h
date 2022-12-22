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

#ifndef _SOURCE_TYPES_TFVECTOR_H_
#define _SOURCE_TYPES_TFVECTOR_H_

#include <tfError.h>

#include <Magnum/Magnum.h>
#include <Magnum/Math/Vector.h>

#include <string>
#include <vector>


namespace TissueForge::types { 


    template<std::size_t size, class T> using VectorBase = Magnum::Math::Vector<size, T>;

    template<std::size_t size, class T> 
    class TVectorS : public VectorBase<size, T> {
        public:
            /** Initialize from an array */
            static TVectorS<size, T>& from(T* data) { return *reinterpret_cast<TVectorS<size, T>*>(&VectorBase<size, T>::from(data)); }

            /** Initialize from an array */
            static const TVectorS<size, T>& from(const T* data) { return *reinterpret_cast<const TVectorS<size, T>*>(&VectorBase<size, T>::from(data)); }

            /** Pad the vector with values if the vector size is smaller than the size of another vector */
            template<std::size_t otherSize> constexpr static TVectorS<size, T> pad(const TVectorS<otherSize, T>& a, T value = T()) {
                return (TVectorS<size, T>)VectorBase<size, T>::pad<otherSize>(a, value);
            }

            TVectorS() : VectorBase<size, T>() {}

            TVectorS(const TVectorS<size, T>&) = default;

            template<class ...U, class V = typename std::enable_if<sizeof...(U)+1 == size, T>::type> constexpr TVectorS(T first, U... next) : VectorBase<size, T>(first, next...) {}

            /** Get the underlying array */
            T* data() { return VectorBase<size, T>::data(); }

            /** Get the underlying array */
            constexpr const T* data() const { return VectorBase<size, T>::data(); }

            T& operator[](std::size_t pos) { return VectorBase<size, T>::operator[](pos); }
            constexpr T operator[](std::size_t pos) const { return VectorBase<size, T>::operator[](pos); }

            bool operator==(const VectorBase<size, T>& other) const { return VectorBase<size, T>::operator==(other); }

            bool operator!=(const VectorBase<size, T>& other) const { return VectorBase<size, T>::operator!=(other); }

            TVectorS<size, T>& operator+=(const TVectorS<size, T>& other) {
                TVectorS<size, T>::operator+=(other);
                return *this;
            }

            TVectorS<size, T> operator+(const TVectorS<size, T>& other) const { return (TVectorS<size, T>)VectorBase<size, T>::operator+((VectorBase<size, T>)other); }

            TVectorS<size, T>& operator-=(const TVectorS<size, T>& other) {
                TVectorS<size, T>::operator-=(other);
                return *this;
            }

            TVectorS<size, T> operator-(const TVectorS<size, T>& other) const { return (TVectorS<size, T>)VectorBase<size, T>::operator-((VectorBase<size, T>)other); }

            TVectorS<size, T>& operator*=(T scalar) {
                TVectorS<size, T>::operator*=(scalar);
                return *this;
            }

            TVectorS<size, T> operator*(T scalar) const { return TVectorS<size, T>(VectorBase<size, T>::operator*(scalar)); }

            TVectorS<size, T>& operator/=(T scalar) {
                VectorBase<size, T>::operator/=(scalar);
                return *this;
            }

            TVectorS<size, T> operator/(T scalar) const { return TVectorS<size, T>(VectorBase<size, T>::operator/(scalar)); }

            TVectorS<size, T>& operator*=(const VectorBase<size, T>& other) { 
                VectorBase<size, T>::operator*=(other);
                return *this;
            }

            TVectorS<size, T> operator*(const VectorBase<size, T>& other) const { return (TVectorS<size, T>)VectorBase<size, T>::operator*((VectorBase<size, T>)other); }

            TVectorS<size, T>& operator/=(const VectorBase<size, T>& other) { 
                VectorBase<size, T>::operator/=(other);
                return *this;
            }

            TVectorS<size, T> operator/(const VectorBase<size, T>& other) const {
                return TVectorS<size, T>(VectorBase<size, T>::operator/(other));
            }

            /** Get the dot product with itself */
            T dot() const { return VectorBase<size, T>::dot(); }

            /** Get the dot product with another vector */
            T dot(const TVectorS<size, T>& other) const { return Magnum::Math::dot(*this, other); }

            /** Get the length */
            T length() const { return VectorBase<size, T>::length(); }

            /** Get the inverted length */
            template<class U = T> typename std::enable_if<std::is_floating_point<U>::value, T>::type
            lengthInverted() const { return VectorBase<size, T>::lengthInverted(); }

            /** Get the normalized vector */
            template<class U = T> typename std::enable_if<std::is_floating_point<U>::value, VectorBase<size, T>>::type
            normalized() const { return VectorBase<size, T>::normalized(); }

            /** Resize the vector */
            template<class U = T> typename std::enable_if<std::is_floating_point<U>::value, VectorBase<size, T>>::type
            resized(T length) const { return VectorBase<size, T>::resized(length); }

            /** Get the vector projected onto another vector */
            template<class U = T> typename std::enable_if<std::is_floating_point<U>::value, VectorBase<size, T>>::type
            projected(const VectorBase<size, T>& line) const { return VectorBase<size, T>::projected(line); }

            /** Get the vector projected onto another normalized vector */
            template<class U = T> typename std::enable_if<std::is_floating_point<U>::value, VectorBase<size, T>>::type
            projectedOntoNormalized(const VectorBase<size, T>& line) const { return VectorBase<size, T>::projectedOntoNormalized(line); }

            /** Get the vector with components in reverse order */
            constexpr TVectorS<size, T> flipped() const { return VectorBase<size, T>::flipped(); }

            /** Get the sum of the elements */
            T sum() const { return VectorBase<size, T>::sum(); }

            /** Get the product of the elements */
            T product() const { return VectorBase<size, T>::product(); }

            /** Get the minimum of the elements */
            T min() const { return VectorBase<size, T>::min(); }

            /** Get the maximum of the elements */
            T max() const { return VectorBase<size, T>::max(); }

            /** Get the minmax of the elements */
            std::pair<T, T> minmax() const { return VectorBase<size, T>::minmax(); }

            TVectorS<size, T>(const VectorBase<size, T> &other) : VectorBase<size, T>() {
                for(int i = 0; i < other.Size; ++i) this->_data[i] = other[i];
            }

            operator VectorBase<size, T>*() { return static_cast<VectorBase<size, T>*>(this); }

            operator VectorBase<size, T>&() { return *static_cast<VectorBase<size, T>*>(this); }

            #ifdef SWIGPYTHON
            T __getitem__(std::size_t i) { return this->operator[](i); }
            void __setitem__(std::size_t i, const T &val) { this->operator[](i) = val; }
            #endif

    };

    #define REVISED_MAGNUM_VECTOR_SUBCLASS_IMPLEMENTATION(size, Type, MagnumImplType)       \
        /** Initialize from an array */                                                     \
        static Type<T>& from(T* data) {                                                     \
            return *reinterpret_cast<Type<T>*>(data);                                       \
        }                                                                                   \
                                                                                            \
        /** Initialize from an array */                                                     \
        static const Type<T>& from(const T* data) {                                         \
            return *reinterpret_cast<const Type<T>*>(data);                                 \
        }                                                                                   \
                                                                                            \
        /** Pad the vector with values if the vector size                                   \
         * is smaller than the size of another vector */                                    \
        template<std::size_t otherSize>                                                     \
        constexpr static Type<T> pad(const Type<T>& a, T value = T()) {                     \
            return MagnumImplType<T>::pad(a, value);                                        \
        }                                                                                   \
                                                                                            \
        template<class U = T>                                                               \
        typename std::enable_if<std::is_signed<U>::value, Type<T>>::type                    \
        operator-() const {                                                                 \
            return MagnumImplType<T>::operator-();                                          \
        }                                                                                   \
        Type<T>& operator+=(const Type<T>& other) {                                         \
            MagnumImplType<T>::operator+=(other);                                           \
            return *this;                                                                   \
        }                                                                                   \
        Type<T> operator+(const Type<T>& other) const {                                     \
            return MagnumImplType<T>::operator+(other);                                     \
        }                                                                                   \
        Type<T>& operator-=(const Type<T>& other) {                                         \
            MagnumImplType<T>::operator-=(other);                                           \
            return *this;                                                                   \
        }                                                                                   \
        Type<T> operator-(const Type<T>& other) const {                                     \
            return MagnumImplType<T>::operator-(other);                                     \
        }                                                                                   \
        Type<T>& operator*=(T number) {                                                     \
            MagnumImplType<T>::operator*=(number);                                          \
            return *this;                                                                   \
        }                                                                                   \
        Type<T> operator*(T number) const {                                                 \
            return MagnumImplType<T>::operator*(number);                                    \
        }                                                                                   \
        Type<T>& operator/=(T number) {                                                     \
            MagnumImplType<T>::operator/=(number);                                          \
            return *this;                                                                   \
        }                                                                                   \
        Type<T> operator/(T number) const {                                                 \
            return MagnumImplType<T>::operator/(number);                                    \
        }                                                                                   \
        Type<T>& operator*=(const Type<T>& other) {                                         \
            MagnumImplType<T>::operator*=(other);                                           \
            return *this;                                                                   \
        }                                                                                   \
        Type<T> operator*(const Type<T>& other) const {                                     \
            return MagnumImplType<T>::operator*(other);                                     \
        }                                                                                   \
        Type<T>& operator/=(const Type<T>& other) {                                         \
            MagnumImplType<T>::operator/=(other);                                           \
            return *this;                                                                   \
        }                                                                                   \
        Type<T> operator/(const Type<T>& other) const {                                     \
            return MagnumImplType<T>::operator/(other);                                     \
        }                                                                                   \
                                                                                            \
        /** Get the length */                                                               \
        template<class U = T>                                                               \
        typename std::enable_if<std::is_floating_point<U>::value, T>::type                  \
        length() const { return MagnumImplType<T>::length(); }                              \
                                                                                            \
        /** Get the normalized vector */                                                    \
        template<class U = T>                                                               \
        typename std::enable_if<std::is_floating_point<U>::value, Type<T>>::type            \
        normalized() const { return MagnumImplType<T>::normalized(); }                      \
                                                                                            \
        /** Resize the vector */                                                            \
        template<class U = T>                                                               \
        typename std::enable_if<std::is_floating_point<U>::value, Type<T>>::type            \
        resized(T length) const {                                                           \
            return MagnumImplType<T>::resized(length);                                      \
        }                                                                                   \
                                                                                            \
        /** Get the vector projected onto another vector */                                 \
        template<class U = T>                                                               \
        typename std::enable_if<std::is_floating_point<U>::value, Type<T>>::type            \
        projected(const Type<T>& other) const {                                             \
            return MagnumImplType<T>::projected(other);                                     \
        }                                                                                   \
                                                                                            \
        /** Get the vector projected onto another normalized vector */                      \
        template<class U = T>                                                               \
        typename std::enable_if<std::is_floating_point<U>::value, Type<T>>::type            \
        projectedOntoNormalized(const Type<T>& other) const {                               \
            return MagnumImplType<T>::projectedOntoNormalized(other);                       \
        }                                                                                   \
                                                                                            \
        /** Get the vector with components in reverse order */                              \
        constexpr Type<T> flipped() const { return MagnumImplType<T>::flipped(); }          \
                                                                                            \
        /** Get the dot product with itself */                                              \
        T dot() const { return Magnum::Math::dot(*this, *this); }                           \
                                                                                            \
        /** Get the dot product with another vector */                                      \
        T dot(const Type<T>& other) const { return Magnum::Math::dot(*this, other); }       \
                                                                                            \
        /** Get the angle made with another vector */                                       \
        template<class U = T>                                                               \
        typename std::enable_if<std::is_floating_point<U>::value, T>::type                  \
        angle(const Type<T>& other) const {                                                 \
            Type<T> a = this->isNormalized() ? *this : this->normalized();                  \
            Type<T> b = other.isNormalized() ? other : other.normalized();                  \
            return T(Magnum::Math::angle(a, b));                                            \
        }                                                                                   \
        T& operator[](std::size_t pos) { return MagnumImplType<T>::operator[](pos); }       \
        constexpr T operator[](std::size_t pos) const {                                     \
            return MagnumImplType<T>::operator[](pos);                                      \
        }                                                                                   \

    #define SWIGPYTHON_MAGNUM_VECTOR_SUBCLASS_IMPLEMENTATION(size, Type)                    \
        T _getitem(int i) {                                                                 \
            if(i < 0) return _getitem(size - 1 + i);                                        \
            return this->operator[](i);                                                     \
        }                                                                                   \
        void _setitem(int i, const T &val) {                                                \
            if(i < 0) return _setitem(size - 1 + i, val);                                   \
            this->operator[](i) = val;                                                      \
        }                                                                                   \
        int __len__() { return size; }                                                      \
                                                                                            \
        /** Initialize from an array */                                                     \
        static Type<T>& fromData(T* data) { return Type<T>::from(data); }                   \
                                                                                            \
        /** Initialize from an array */                                                     \
        static const Type<T>& fromData(const T* data) { return Type<T>::from(data); }       \
        std::vector<T>& asVector() {                                                        \
            std::vector<T> *result = new std::vector<T>(*this);                             \
            return *result;                                                                 \
        }                                                                                   \
                                                                                            \
        Type<T>& operator+=(const std::vector<T>& other) {                                  \
            Type<T>::operator+=(Type<T>(other));                                            \
            return *this;                                                                   \
        }                                                                                   \
        Type<T> operator+(const std::vector<T>& other) const {                              \
            return Type<T>::operator+(Type<T>(other));                                      \
        }                                                                                   \
        Type<T>& operator-=(const std::vector<T>& other) {                                  \
            Type<T>::operator-=(Type<T>(other));                                            \
            return *this;                                                                   \
        }                                                                                   \
        Type<T> operator-(const std::vector<T>& other) const {                              \
            return Type<T>::operator-(Type<T>(other));                                      \
        }                                                                                   \
        Type<T>& operator*=(const std::vector<T>& other) {                                  \
            Type<T>::operator*=(Type<T>(other));                                            \
            return *this;                                                                   \
        }                                                                                   \
        Type<T> operator*(const std::vector<T>& other) const {                              \
            return Type<T>::operator*(Type<T>(other));                                      \
        }                                                                                   \
        Type<T>& operator/=(const std::vector<T>& other) {                                  \
            Type<T>::operator/=(Type<T>(other));                                            \
            return *this;                                                                   \
        }                                                                                   \
        Type<T> operator/(const std::vector<T>& other) const {                              \
            return Type<T>::operator/(Type<T>(other));                                      \
        }                                                                                   \


    #define MAGNUM_BASE_VECTOR_CAST_METHODS(size, Type, MagnumImplType)                     \
        template<class U = T>                                                               \
        Type(const MagnumImplType<U> &other) : MagnumImplType<T>(other) {}                  \
        template<class U = T>                                                               \
        Type(const MagnumImplType<U> *other) : MagnumImplType<T>(*other) {}                 \
        template<class U = T>                                                               \
        operator MagnumImplType<U>*() {                                                     \
            return static_cast<MagnumImplType<U>*>(this);                                   \
        }                                                                                   \
        template<class U = T>                                                               \
        operator const MagnumImplType<U>*() {                                               \
            return static_cast<const MagnumImplType<U>*>(this);                             \
        }                                                                                   \
                                                                                            \
        template<class U = T>                                                               \
        Type(const VectorBase<size, U> &other) : MagnumImplType<T>(other) {}                \
                                                                                            \
        Type(const std::vector<T> &v) : MagnumImplType<T>() {                               \
            for(int i = 0; i < size; ++i) this->_data[i] = v[i];                            \
        }                                                                                   \
        operator std::vector<T>&() const {                                                  \
            std::vector<T> *result = new std::vector<T>(std::begin(this->_data),            \
                std::end(this->_data));                                                     \
            return *result;                                                                 \
        }                                                                                   \

}

template<std::size_t size, typename T>
inline std::ostream& operator<<(std::ostream& os, const TissueForge::types::TVectorS<size, T>& vec)
{
    os << std::string("{") << vec[0];
    for(int i = 1; i < vec.Size; ++i) os << std::string(",") << vec[i];
    os << std::string("}");
    return os;
}


#define TF_VECTOR_IMPL_OSTREAM(type)                                                    \
    template<typename T>                                                                \
    inline std::ostream& operator<<(std::ostream& os, const type<T>& vec)               \
    {                                                                                   \
        os << std::string("{") << vec[0];                                               \
        for(int i = 1; i < vec.Size; ++i) os << std::string(",") << vec[i];             \
        os << std::string("}");                                                         \
        return os;                                                                      \
    }                                                                                   \

#endif // _SOURCE_TYPES_TFVECTOR_H_