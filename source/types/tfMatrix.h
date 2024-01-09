/*******************************************************************************
 * This file is part of Tissue Forge.
 * Copyright (c) 2022-2024 T.J. Sego
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

#ifndef _SOURCE_TYPES_TFMATRIX_H_
#define _SOURCE_TYPES_TFMATRIX_H_

#include <tfError.h>

#include "tfVector.h"

#include <Magnum/Magnum.h>
#include <Magnum/Math/Matrix.h>

#include <ostream>


namespace TissueForge::types { 


    template<std::size_t size, class T> using MatrixBase = Magnum::Math::Matrix<size, T>;

    template<std::size_t size, class T> 
    class TMatrixS : public MatrixBase<size, T> {
        public:
            constexpr TMatrixS() noexcept: MatrixBase<size, T>() {}

            template<class ...U> constexpr TMatrixS(const TVectorS<size, T>& first, const U&... next) noexcept: MatrixBase<size, T>(first, next...) {}

            constexpr explicit TMatrixS(T value) noexcept: MatrixBase<size, T>(value) {}

            template<class U> constexpr explicit TMatrixS(const TMatrixS<size, U>& other) noexcept: MatrixBase<size, T>((MatrixBase<size, U>)other) {}

            template<std::size_t otherSize> constexpr explicit TMatrixS(const TMatrixS<otherSize, T>& other) noexcept: MatrixBase<size, T>((MatrixBase<otherSize, T>)other) {}

            constexpr /*implicit*/ TMatrixS(const TMatrixS<size, T>& other) noexcept: MatrixBase<size, T>((MatrixBase<size, T>)other) {}

            /** Test whether the matrix is orthogonal */
            bool isOrthogonal() const { return MatrixBase<size, T>::isOrthogonal(); };

            /** Get the trace */
            T trace() const { return MatrixBase<size, T>::trace(); }

            /** Get the matrix without a column and row */
            TMatrixS<size-1, T> ij(std::size_t skipCol, std::size_t skipRow) const { return (TMatrixS<size-1, T>)MatrixBase<size-1, T>::ij(skipCol, skipRow); }

            /** Get the cofactor */
            T cofactor(std::size_t col, std::size_t row) const { return MatrixBase<size, T>::cofactor(col, row); }

            /** Get the comatrix */
            TMatrixS<size, T> comatrix() const { return (TMatrixS<size, T>)MatrixBase<size, T>::comatrix(); }

            /** Get the adjugate */
            TMatrixS<size, T> adjugate() const { return (TMatrixS<size, T>)MatrixBase<size, T>::adjugate(); }

            /** Get the determinant */
            T determinant() const { return MatrixBase<size, T>::determinant(); }

            /** Get the invertex matrix */
            TMatrixS<size, T> inverted() const { return (TMatrixS<size, T>)MatrixBase<size, T>::inverted(); }

            /** Get the orthogonal inverted matrix */
            TMatrixS<size, T> invertedOrthogonal() const { return (TMatrixS<size, T>)MatrixBase<size, T>::invertedOrthogonal(); }

            /* Reimplementation of functions to return correct type */

            TMatrixS<size, T> operator*(const TMatrixS<size, T>& other) const { return (TMatrixS<size, T>)MatrixBase<size, T>::operator*((MatrixBase<size, T>)other); }
            TVectorS<size, T> operator*(const TVectorS<size, T>& other) const { return (TVectorS<size, T>)MatrixBase<size, T>::operator*(other); }

            /** Get the transpose */
            TMatrixS<size, T> transposed() const { return (TMatrixS<size, T>)MatrixBase<size, T>::transposed(); }

            /** Initialize from an array */
            static TMatrixS<size, T>& from(T* data) { return (TMatrixS<size, T>)MatrixBase<size, T>::from(data); }

            /** Initialize from an array */
            static const TMatrixS<size, T>& from(const T* data) { return (TMatrixS<size, T>)MatrixBase<size, T>::from(data); }
            
            TMatrixS<size, T> operator-() const {
                return (TMatrixS<size, T>)MatrixBase<size, T>::operator-();
            }
            TMatrixS<size, T>& operator+=(const TMatrixS<size, T>& other) {
                MatrixBase<size, T>::operator+=((MatrixBase<size, T>)other);
                return *this;
            }
            TMatrixS<size, T> operator+(const TMatrixS<size, T>& other) const {
                return (TMatrixS<size, T>)MatrixBase<size, T>::operator+((MatrixBase<size, T>)other);
            }
            TMatrixS<size, T>& operator-=(const TMatrixS<size, T>& other) {
                MatrixBase<size, T>::operator-=((MatrixBase<size, T>)other);
                return *this;
            }
            TMatrixS<size, T> operator-(const TMatrixS<size, T>& other) const {
                return (TMatrixS<size, T>)MatrixBase<size, T>::operator-((MatrixBase<size, T>)other);
            }
            TMatrixS<size, T>& operator*=(T number) {
                MatrixBase<size, T>::operator*=(number);
                return *this;
            }
            TMatrixS<size, T> operator*(T number) const {
                return (TMatrixS<size, T>)MatrixBase<size, T>::operator*(number);
            }
            TMatrixS<size, T>& operator/=(T number) {
                MatrixBase<size, T>::operator/=(number);
                return *this;
            }
            TMatrixS<size, T> operator/(T number) const {
                return (TMatrixS<size, T>)MatrixBase<size, T>::operator/(number);
            }

            /** Get the matrix with flipped columns */
            constexpr TMatrixS<size, T> flippedCols() const {
                return (TMatrixS<size, T>)MatrixBase<size, T>::flippedCols();
            }

            /** Get the matrix with flipped rows */
            constexpr TMatrixS<size, T> flippedRows() const {
                return (TMatrixS<size, T>)MatrixBase<size, T>::flippedRows();
            }
            
    };

    #define REVISED_MAGNUM_MATRIX_SUBCLASS_IMPLEMENTATION(size, Type, MagnumImplType, VectorType) \
        /** Initialize from an array */                                                     \
        static Type<T>& from(T* data) {                                                     \
        return *reinterpret_cast<Type<T>*>(&MagnumImplType<T>::from(data));                 \
        }                                                                                   \
                                                                                            \
        Type<T> operator-() const {                                                         \
            return (Type<T>)MagnumImplType<T>::operator-();                                 \
        }                                                                                   \
        Type<T>& operator+=(const Type<T>& other) {                                         \
            MagnumImplType<T>::operator+=((MagnumImplType<T>)other);                        \
            return *this;                                                                   \
        }                                                                                   \
        Type<T> operator+(const Type<T>& other) const {                                     \
            return (Type<T>)MagnumImplType<T>::operator+((MagnumImplType<T>)other);         \
        }                                                                                   \
        Type<T>& operator-=(const Type<T>& other) {                                         \
            MagnumImplType<T>::operator-=((MagnumImplType<T>)other);                        \
            return *this;                                                                   \
        }                                                                                   \
        Type<T> operator-(const Type<T>& other) const {                                     \
            return (Type<T>)MagnumImplType<T>::operator-((MagnumImplType<T>)other);         \
        }                                                                                   \
        Type<T>& operator*=(T number) {                                                     \
            MagnumImplType<T>::operator*=(number);                                          \
            return *this;                                                                   \
        }                                                                                   \
        Type<T> operator*(T number) const {                                                 \
            return (Type<T>)MagnumImplType<T>::operator*(number);                           \
        }                                                                                   \
        Type<T>& operator/=(T number) {                                                     \
            MagnumImplType<T>::operator/=(number);                                          \
            return *this;                                                                   \
        }                                                                                   \
        Type<T> operator/(T number) const {                                                 \
            return (Type<T>)MagnumImplType<T>::operator/(number);                           \
        }                                                                                   \
                                                                                            \
        /** Get the matrix with flipped columns */                                          \
        constexpr Type<T> flippedCols() const {                                             \
            return (Type<T>)MagnumImplType<T>::flippedCols();                               \
        }                                                                                   \
                                                                                            \
        /** Get the matrix with flipped rows */                                             \
        constexpr Type<T> flippedRows() const {                                             \
            return (Type<T>)MagnumImplType<T>::flippedRows();                               \
        }                                                                                   \
        VectorType<T>& operator[](std::size_t col) {                                        \
            return static_cast<VectorType<T>&>(MagnumImplType<T>::operator[](col));         \
        }                                                                                   \
        constexpr const VectorType<T> operator[](std::size_t col) const {                   \
            return VectorType<T>(MagnumImplType<T>::operator[](col));                       \
        }                                                                                   \
                                                                                            \
        /** Get a row */                                                                    \
        VectorType<T> row(std::size_t row) const {                                          \
            return VectorType<T>(MagnumImplType<T>::row(row));                              \
        }                                                                                   \
                                                                                            \
        Type<T> operator*(const Type<T>& other) const {                                     \
            return MagnumImplType<T>::operator*(other);                                     \
        }                                                                                   \
        VectorType<T> operator*(const TVectorS<size, T>& other) const {                     \
            return (VectorType<T>)MatrixBase<size, T>::operator*(other);                    \
        }                                                                                   \
                                                                                            \
        /** Get the transpose */                                                            \
        Type<T> transposed() const { return MatrixBase<size, T>::transposed(); }            \
        constexpr VectorType<T> diagonal() const {                                          \
            return (VectorType<T>)MatrixBase<size, T>::diagonal();                          \
        }                                                                                   \
                                                                                            \
        /** Get the inverted matrix */                                                      \
        Type<T> inverted() const { return MatrixBase<size, T>::inverted(); }                \
                                                                                            \
        /** Get the orthogonal inverted matrix */                                           \
        Type<T> invertedOrthogonal() const {                                                \
            return MatrixBase<size, T>::invertedOrthogonal();                               \
        }                                                                                   \
                                                                                            \
        /** Get the underlying array */                                                     \
        T* data() { return MagnumImplType<T>::data(); }                                     \
                                                                                            \
        /** Get the underlying array */                                                     \
        constexpr const T* data() const { return MagnumImplType<T>::data(); }               \

    #define SWIGPYTHON_MAGNUM_MATRIX_SUBCLASS_IMPLEMENTATION(size, Type, VectorType)        \
        VectorType<T>& _getitem(int i) {                                                    \
            if(i < 0) return _getitem(size - 1 + i);                                        \
            return this->operator[](i);                                                     \
        }                                                                                   \
        void _setitem(int i, const VectorType<T> &val) {                                    \
            if(i < 0) return _setitem(size - 1 + i, val);                                   \
            VectorType<T> &item = this->operator[](i);                                      \
            item = val;                                                                     \
        }                                                                                   \
        int __len__() { return size; }                                                      \
        std::vector<std::vector<T> >& asVectors() {                                         \
            std::vector<std::vector<T> > *result = new std::vector<std::vector<T> >(size);  \
            for(int i = 0; i < size; ++i) {                                                 \
                std::vector<T> &item = (*result)[i];                                        \
                item = this->operator[](i);                                                 \
            }                                                                               \
            return *result;                                                                 \
        }                                                                                   \

    #define MAGNUM_BASE_MATRIX_CAST_METHODS(size, Type, MagnumImplType)                     \
        template<class U = T>                                                               \
        Type(const MagnumImplType<U>& other) : MagnumImplType<T>(other) {}                  \
        template<class U = T>                                                               \
        operator MagnumImplType<U>*() { return static_cast<MagnumImplType<U>*>(this); }     \
        template<class U = T>                                                               \
        operator const MagnumImplType<U>*() {                                               \
            return static_cast<const MagnumImplType<U>*>(this);                             \
        }                                                                                   \
                                                                                            \
        template<class U = T>                                                               \
        Type(const MatrixBase<size, U>& other) : MagnumImplType<T>(other) {}                \
                                                                                            \
        operator std::vector<std::vector<T> >&() const {                                    \
            std::vector<T> *result = new std::vector<T>(size);                              \
            for(int i = 0; i < size; ++i) {                                                 \
                std::vector<T> *c = new std::vector<T>(this->operator[](i));                \
                *result[i] = *c;                                                            \
            }                                                                               \
            return *result;                                                                 \
        }                                                                                   \

    }

template<std::size_t size, class T>
inline std::ostream& operator<<(std::ostream& os, const TissueForge::types::TMatrixS<size, T>& m)
{
    os << std::string("{");
    for(int i = 0; i < size; ++i) os << m.row(i) << std::string(",") << std::endl;
    os << std::string("}");
    return os;
}

#define TF_MATRIX_IMPL_OSTREAM(type)                                                    \
    template<class T>                                                                   \
    inline std::ostream& operator<<(std::ostream& os, const type<T>& m)                 \
    {                                                                                   \
        os << std::string("{");                                                         \
        for(int i = 0; i < m.Size; ++i) os << m.row(i) << std::string(",") << std::endl;\
        os << std::string("}");                                                         \
        return os;                                                                      \
    }                                                                                   \

#endif // _SOURCE_TYPES_TFMATRIX_H_