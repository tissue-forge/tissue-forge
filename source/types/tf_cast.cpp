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

#include "tf_cast.h"


namespace TissueForge {

    template<>
    std::string cast(const int &t) {
        return std::to_string(t);
    }

    template<>
    std::string cast(const long &t) {
        return std::to_string(t);
    }

    template<>
    std::string cast(const long long &t) {
        return std::to_string(t);
    }

    template<>
    std::string cast(const unsigned int &t) {
        return std::to_string(t);
    }

    template<>
    std::string cast(const unsigned long &t) {
        return std::to_string(t);
    }

    template<>
    std::string cast(const unsigned long long &t) {
        return std::to_string(t);
    }

    template<> 
    std::string cast(const bool &t) {
        return std::to_string(t);
    }

    template<>
    std::string cast(const float &t) {
        return std::to_string(t);
    }

    template<>
    std::string cast(const double &t) {
        return std::to_string(t);
    }

    template<>
    std::string cast(const long double &t) {
        return std::to_string(t);
    }

    template<>
    int cast(const std::string &s) {
        return std::stoi(s);
    }

    template<>
    long cast(const std::string &s) {
        return std::stol(s);
    }

    template<>
    long long cast(const std::string &s) {
        return std::stoll(s);
    }

    template<>
    unsigned int cast(const std::string &s) {
        return std::stoul(s);
    }

    template<>
    unsigned long cast(const std::string &s) {
        return std::stoul(s);
    }

    template<>
    unsigned long long cast(const std::string &s) {
        return std::stoull(s);
    }

    template<>
    bool cast(const std::string &s) {
        return (bool)cast<std::string, int>(s);
    }

    template<>
    float cast(const std::string &s) {
        return std::stof(s);
    }

    template<>
    double cast(const std::string &s) {
        return std::stod(s);
    }

    template<>
    long double cast(const std::string &s) {
        return std::stold(s);
    }

    template<>
    bool check<int>(const std::string &s) {
        try {
            cast<std::string, int>(s);
            return true;
        }
        catch (const std::invalid_argument &) {
            return false;
        }
    }

    template<>
    bool check<long>(const std::string &s) {
        try {
            cast<std::string, long>(s);
            return true;
        }
        catch (const std::invalid_argument &) {
            return false;
        }
    }

    template<>
    bool check<long long>(const std::string &s) {
        try {
            cast<std::string, long long>(s);
            return true;
        }
        catch (const std::invalid_argument &) {
            return false;
        }
    }

    template<>
    bool check<unsigned int>(const std::string &s) {
        try {
            cast<std::string, unsigned int>(s);
            return true;
        }
        catch (const std::invalid_argument &) {
            return false;
        }
    }

    template<>
    bool check<unsigned long>(const std::string &s) {
        try {
            cast<std::string, unsigned long>(s);
            return true;
        }
        catch (const std::invalid_argument &) {
            return false;
        }
    }

    template<>
    bool check<unsigned long long>(const std::string &s) {
        try {
            cast<std::string, unsigned long long>(s);
            return true;
        }
        catch (const std::invalid_argument &) {
            return false;
        }
    }

    template<>
    bool check<bool>(const std::string &s) {
        try {
            cast<std::string, bool>(s);
            return true;
        }
        catch (const std::invalid_argument &) {
            return false;
        }
    }

    template<>
    bool check<float>(const std::string &s) {
        try {
            cast<std::string, float>(s);
            return true;
        }
        catch (const std::invalid_argument &) {
            return false;
        }
    }

    template<>
    bool check<double>(const std::string &s) {
        try {
            cast<std::string, double>(s);
            return true;
        }
        catch (const std::invalid_argument &) {
            return false;
        }
    }

    template<>
    bool check<long double>(const std::string &s) {
        try {
            cast<std::string, long double>(s);
            return true;
        }
        catch (const std::invalid_argument &) {
            return false;
        }
    }

}