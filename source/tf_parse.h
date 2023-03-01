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

#ifndef _SOURCE_TF_PARSE_H_
#define _SOURCE_TF_PARSE_H_

#include <types/tf_cast.h>

#include <sstream>
#include <string>
#include <vector>


namespace TissueForge::parse {


    /**
     * @brief Log that no keyword was found
     */
    void logMessageNoKwarg();

    /**
     * @brief Log that a keyword was not found
     * 
     * @param s keyword
     */
    void logMessageNoKwargFound(const std::string &s);

    /**
     * @brief Check whether an argument has a keyword of the form X=Y. 
     * 
     * @param arg argument
     * @return true if the argument has a keyword
     */
    bool has_kwarg(const std::string &arg);

    /**
     * @brief Check whether an argument has a specified keyword of the form X=Y. 
     * 
     * @param arg argument
     * @param kwarg keyword to check
     * @return true if the argument has the specified keyword
     */
    bool has_kwarg(const std::string &arg, const std::string &kwarg);

    /**
     * @brief Check whether one of multiple arguments has a specified keyword of the form X=Y. 
     * 
     * @param args arguments
     * @param kwarg keyword to check
     * @return true if one of the arguments has the specified keyword
     */
    bool has_kwarg(const std::vector<std::string> &args, const std::string &kwarg);

    /**
     * @brief Get the keyword of an argument of the form X=Y. 
     * 
     * Return an empty string if the keyword is not found. 
     * 
     * @param arg argument
     * @return X for the form X=Y 
     */
    std::string get_kwarg(const std::string &arg);

    /**
     * @brief Get the argument that has a specified keyword of the form X=Y. 
     * 
     * Return an empty string if the keyword is not found. 
     * 
     * @param args multiple arguments
     * @param kwarg keyword to search by
     * @return argument with the specified keyword
     */
    std::string kwarg_findStr(const std::vector<std::string> &args, const std::string &kwarg);

    /**
     * @brief Get the value of an argument with a specified keyword of the form X=Y. 
     * 
     * Return an empty string if the keyword is not found. 
     * 
     * @param arg argument X=Y
     * @param kwarg keyword X
     * @return Y
     */
    std::string kwarg_strVal(const std::string &arg, const std::string &kwarg);

    /**
     * @brief Get the value of an argument with a specified keyword of the form X=Y. 
     * 
     * Return 0 if the keyword is not found. 
     * 
     * @tparam T type of return value
     * @param arg argument X=Y
     * @param kwarg keyword X
     * @return Y
     */
    template<typename T>
    T kwarg_getVal(const std::string &arg, const std::string &kwarg) {
        if(!has_kwarg(arg, kwarg)) {
            logMessageNoKwargFound(kwarg);
            return 0;
        }

        return TissueForge::cast<std::string, T>(kwarg_strVal(arg, kwarg));
    }

    /**
     * @brief Get the value of a keyword from arguments with a specified keyword of the form X=Y. 
     * 
     * Return 0 if the keyword is not found. 
     * 
     * @tparam T type of return value
     * @param args arguments of the form X=Y
     * @param kwarg keyword X
     * @return Y
     */
    template<typename T>
    T kwarg_getVal(const std::vector<std::string> &args, const std::string &kwarg) {
        for(auto &s : args)
            if(has_kwarg(s, kwarg)) return kwarg_getVal<T>(s, kwarg);
        
        logMessageNoKwargFound(kwarg);
        return 0;
    }

    /**
     * @brief Get the keyword and value of an argument of the form X=Y. 
     * 
     * Return an empty string if the keyword is not found. 
     * 
     * @param arg argument X=Y
     * @return X, Y
     */
    std::pair<std::string, std::string> kwarg_getNameVal(const std::string &arg);

    /**
     * @brief Get the argument X=Y for a specified keyword and value. 
     * 
     * @tparam T type of the value
     * @param kwarg keyword X
     * @param t value Y
     * @return X=Y
     */
    template<typename T>
    std::string kwarg_valSetStr(const std::string &kwarg, const T &t) {
        return kwarg + "=" + TissueForge::cast<T, std::string>(t);
    }

    /**
     * @brief Get the value of an argument with a specified keyword of the form X=Y. 
     * 
     * Return an empty string if the keyword is not found. 
     * 
     * @param kwarg argument X=Y
     * @param keyword keyword X
     * @return Y
     */
    std::string kwargVal(const std::string &kwarg, const std::string &keyword);

    /**
     * @brief Get the value of a keyword from arguments with a specified keyword of the form X=Y. 
     * 
     * @param kwargs arguments of the form X=Y
     * @param keyword X
     * @return Y
     */
    std::string kwargVal(const std::vector<std::string> &kwargs, const std::string &keyword);

    // Vector format: vec{a, b, c} <-> str=a,b,c

    /**
     * @brief Generate a vector from a value of the form a,b,c,...
     * 
     * @tparam T type of vector elements
     * @param s comma-separated string of vector values
     * @return vector
     */
    template<typename T>
    std::vector<T> strToVec(const std::string &s) {
        std::vector<T> result;

        std::stringstream ss(s);
        std::string str;
        while(getline(ss, str, ',')) result.push_back(TissueForge::cast<std::string, T>(str));

        return result;
    }

    /**
     * @brief Generate a string value from a vector.
     * 
     * @tparam T type of vector elements
     * @param v vector
     * @return comma-separated string of vector values
     */
    template<typename T>
    std::string vecToStr(const std::vector<T> &v) {
        std::string result;
        if(v.size() == 0) return result;

        result = TissueForge::cast<T, std::string>(v[0]);
        if(v.size() == 1) return result;

        for(int i = 1; i < v.size(); ++i) result += "," + TissueForge::cast<T, std::string>(v[i]);

        return result;
    }

    // Heterogeneous map: {"A", {a, b, c}}, {"B", {1, 2, 3}} <-> {A=a,b,c;B=1,2,3}

    /**
     * @brief Check whether a string argument X=Y has a keyword with a map value.
     * 
     * @param arg argument X=Y
     * @param kwarg X
     * @return true if Y is a map
     */
    bool has_mapKwarg(const std::string &arg, const std::string &kwarg);

    /**
     * @brief Check whether string arguments of the form X=Y have a keyword with a map value.
     * 
     * @param args arguments of the form X=Y
     * @param kwarg X
     * @return true if the arguments have X and Y is a map
     */
    bool has_mapKwarg(const std::vector<std::string> &args, const std::string &kwarg);

    /**
     * @brief Find an argument with a keyword that has a map value. 
     * 
     * Return an empty string if the keyword is not found. 
     * 
     * @param args arguments of the form X=Y to search
     * @param kwarg X
     * @return Y
     */
    std::string kwarg_findMapStr(const std::vector<std::string> &args, const std::string &kwarg);

    /**
     * @brief Strip a map of its enclosing formatting. 
     * 
     * @param arg map value {A=B;C=D;...}
     * @return A=B;C=D;...
     */
    std::string mapStrip(const std::string &arg);

    /**
     * @brief Get the stripped value of a map from an argument of the form X={A=B;C=D;...}.
     * 
     * @param arg X={A=B;C=D;...}
     * @param kwarg X
     * @return A=B;C=D;... 
     */
    std::string mapStrip(const std::string &arg, const std::string &kwarg);

    /**
     * @brief Get the arguments of a stripped map value A=B;C=D;...
     * 
     * @param s stripped map value A=B;C=D;...
     * @return vector with elements {A=B,C=D,...}
     */
    std::vector<std::string> mapStrToStrVec(const std::string &s);

    /**
     * @brief Get the formatted map from a vector of arguments of the form {A=B,C=D,...}.
     * 
     * @param v vector of arguments of the form {A=B,C=D,...}
     * @return formatted map {A=B;C=D;...} 
     */
    std::string mapVecToStr(const std::vector<std::string> &v);

    /**
     * @brief Get the stripped map value of a keyword from an argument of the form X={Y}.
     * 
     * @param arg argument of the form X={Y}
     * @param kwarg X
     * @return Y
     */
    std::string kwarg_strMapVal(const std::string &arg, const std::string &kwarg);

    /**
     * @brief Get the stripped map value of a keyword from arguments of the form X={Y}.
     * 
     * @param args arguments of the form X={Y}
     * @param kwarg X
     * @return Y 
     */
    std::string kwarg_strMapVal(const std::vector<std::string> &args, const std::string &kwarg);

    /**
     * @brief Get the keyword and stripped map arguments of an argument of the form X={A=B;C=D;...}.
     * 
     * @param arg argument
     * @return X, {A=B,C=D,...}
     */
    std::pair<std::string, std::vector<std::string>> kwarg_getNameMapVals(const std::string &arg);

};

#endif // _SOURCE_TF_PARSE_H_