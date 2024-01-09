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

#ifndef _SOURCE_TYPES_TF_CAST_H_
#define _SOURCE_TYPES_TF_CAST_H_

#include <stdexcept>
#include <string>
#include <vector>


namespace TissueForge {


    template<typename T, typename S> 
    S cast(const T&);
    
    template<typename T, typename S> 
    S cast(T*);

    template<> std::string cast(const int &t);

    template<> std::string cast(const long &t);

    template<> std::string cast(const long long &t);

    template<> std::string cast(const unsigned int &t);

    template<> std::string cast(const unsigned long &t);

    template<> std::string cast(const unsigned long long &t);

    template<> std::string cast(const bool &t);

    template<> std::string cast(const float &t);

    template<> std::string cast(const double &t);

    template<> std::string cast(const long double &t);

    template<> int cast(const std::string &s);

    template<> long cast(const std::string &s);

    template<> long long cast(const std::string &s);

    template<> unsigned int cast(const std::string &s);

    template<> unsigned long cast(const std::string &s);

    template<> unsigned long long cast(const std::string &s);

    template<> bool cast(const std::string &s);

    template<> float cast(const std::string &s);

    template<> double cast(const std::string &s);

    template<> long double cast(const std::string &s);

    template<typename T, typename S>
    bool check(const T&);

    template<typename T>
    bool check(const std::string&);

    template<> bool check<int>(const std::string &s);

    template<> bool check<long>(const std::string &s);

    template<> bool check<long long>(const std::string &s);

    template<> bool check<unsigned int>(const std::string &s);

    template<> bool check<unsigned long>(const std::string &s);

    template<> bool check<unsigned long long>(const std::string &s);

    template<> bool check<bool>(const std::string &s);

    template<> bool check<float>(const std::string &s);

    template<> bool check<double>(const std::string &s);

    template<> bool check<long double>(const std::string &s);

}

#endif // _SOURCE_TYPES_TF_CAST_H_