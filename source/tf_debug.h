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

#ifndef _SOURCE_TF_DEBUG_H_
#define _SOURCE_TF_DEBUG_H_

#include <iostream>
#include <Magnum/Magnum.h>
#include <Magnum/Math/Vector3.h>
#include <Magnum/Math/Matrix3.h>
#include <Corrade/Utility/Debug.h>
#include <sstream>
#include <array>


inline std::ostream& operator<<(std::ostream& os, const Magnum::Vector3& vec)
{
    os << "{" << vec[0] << "," << vec[1] << "," << vec[2] << "}";
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const Magnum::Vector3ui& vec)
{
    os << "{" << vec[0] << "," << vec[1] << "," << vec[2] << "}";
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const Magnum::Math::Matrix3<float>& m)
{
    os << "{" << m.row(0) << "," << std::endl
       << "  " << m.row(1) << "," << std::endl
       << "  " << m.row(2) << "}" << std::endl;
    return os;
}


template <typename ArrayType, size_t Length>
std::ostream& operator << (std::ostream& stream, const std::array<ArrayType, Length>);


template <typename ArrayType>
std::ostream& operator << (std::ostream& stream, const std::array<ArrayType, 2> a) {
    stream << "{"  << a[0] << ", " << a[1] << "}";
    return stream;
}

template <typename ArrayType>
std::ostream& operator << (std::ostream& stream, const std::array<ArrayType, 3> a) {
    stream << "{"  << a[0] << ", " << a[1] << ", " << a[2] << "}";
    return stream;
}

template <typename ArrayType>
std::ostream& operator << (std::ostream& stream, const std::array<ArrayType, 4> a) {
    stream << "{"  << a[0] << ", " << a[1] << ", " << a[2] << ", " << a[3] <<"}";
    return stream;
}

template<typename MagnumType>
std::string to_string(const MagnumType &val) {
    std::ostringstream ss;
    ss << std::fixed;
    ss.precision(4);
    ss << val;
    return ss.str();
}

#endif // _SOURCE_TF_DEBUG_H_