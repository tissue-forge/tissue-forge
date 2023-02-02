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

#ifndef _SOURCE_IO_TF_IO_H_
#define _SOURCE_IO_TF_IO_H_

#include <tf_port.h>
#include <tf_config.h>
#include <types/tf_types.h>

#include <map>
#include <unordered_map>
#include <set>
#include <unordered_set>
#include <string>


namespace TissueForge::io {


    using IOChildMap = std::unordered_map<std::string, struct IOElement*>;

    /**
     * @brief Intermediate I/O class for reading/writing 
     * Tissue Forge objects to/from file/string. 
     * 
     */
    struct IOElement {
        std::string type;
        std::string value;
        IOElement *parent = NULL;
        IOChildMap children;
    };

    /**
     * @brief Tissue Forge meta data. 
     * 
     * An instance is always stored in any object export. 
     * 
     */
    struct MetaData {
        unsigned int versionMajor = TF_VERSION_MAJOR;
        unsigned int versionMinor = TF_VERSION_MINOR;
        unsigned int versionPatch = TF_VERSION_PATCH;
    };

    /**
     * @brief Convert an object to an intermediate I/O object
     * 
     * @tparam T type of object to convert
     * @param dataElement object to convert
     * @param metaData meta data of target installation
     * @param fileElement resulting I/O object
     * @return HRESULT 
     */
    template <typename T>
    HRESULT toFile(const T &dataElement, const MetaData &metaData, IOElement *fileElement);

    /**
     * @brief Convert an object to an intermediate I/O object
     * 
     * @tparam T type of object to convert
     * @param dataElement object to convert
     * @param metaData meta data of target installation
     * @param fileElement resulting I/O object
     * @return HRESULT 
     */
    template <typename T>
    HRESULT toFile(T *dataElement, const MetaData &metaData, IOElement *fileElement);

    /**
     * @brief Instantiate an object from an intermediate I/O object
     * 
     * @tparam T type of object to instantiate
     * @param fileElement source I/O object
     * @param metaData meta data of exporting installation
     * @param dataElement resulting object
     * @return HRESULT 
     */
    template <typename T>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, T *dataElement);


    // Tissue Forge types


    // MetaData

    template <>
    HRESULT toFile(const MetaData &dataElement, const MetaData &metaData, IOElement *fileElement);

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, MetaData *dataElement);

    // TissueForge::types::TVector2<T>

    template <typename T>
    HRESULT toFile(const TissueForge::types::TVector2<T> &dataElement, const MetaData &metaData, IOElement *fileElement) {
        
        fileElement->type = "Vector2";
        IOElement *xfe = new IOElement();
        IOElement *yfe = new IOElement();

        if(toFile(dataElement.x(), metaData, xfe) != S_OK || toFile(dataElement.y(), metaData, yfe) != S_OK) 
            return E_FAIL;
        
        xfe->parent = fileElement;
        yfe->parent = fileElement;
        fileElement->children["x"] = xfe;
        fileElement->children["y"] = yfe;

        return S_OK;
    }

    template <typename T>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, TissueForge::types::TVector2<T> *dataElement) {

        T de;
        auto feItr = fileElement.children.find("x");
        if(feItr == fileElement.children.end() || fromFile(*feItr->second, metaData, &de) != S_OK) 
            return E_FAIL;
        
        (*dataElement)[0] = de;

        feItr = fileElement.children.find("y");
        if(feItr == fileElement.children.end() || fromFile(*feItr->second, metaData, &de) != S_OK) 
            return E_FAIL;
        
        (*dataElement)[1] = de;
        
        return S_OK;
    }

    // TissueForge::types::TVector3<T>

    template <typename T>
    HRESULT toFile(const TissueForge::types::TVector3<T> &dataElement, const MetaData &metaData, IOElement *fileElement) {

        fileElement->type = "Vector3";
        IOElement *xfe = new IOElement();
        IOElement *yfe = new IOElement();
        IOElement *zfe = new IOElement();

        if(toFile(dataElement.x(), metaData, xfe) != S_OK || 
        toFile(dataElement.y(), metaData, yfe) != S_OK || 
        toFile(dataElement.z(), metaData, zfe) != S_OK) 
            return E_FAIL;
        
        xfe->parent = fileElement;
        yfe->parent = fileElement;
        zfe->parent = fileElement;
        fileElement->children["x"] = xfe;
        fileElement->children["y"] = yfe;
        fileElement->children["z"] = zfe;

        return S_OK;
    }

    template <typename T>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, TissueForge::types::TVector3<T> *dataElement) {

        T de;
        auto feItr = fileElement.children.find("x");
        if(feItr == fileElement.children.end() || fromFile(*feItr->second, metaData, &de) != S_OK) 
            return E_FAIL;
        
        (*dataElement)[0] = de;

        feItr = fileElement.children.find("y");
        if(feItr == fileElement.children.end() || fromFile(*feItr->second, metaData, &de) != S_OK) 
            return E_FAIL;
        
        (*dataElement)[1] = de;

        feItr = fileElement.children.find("z");
        if(feItr == fileElement.children.end() || fromFile(*feItr->second, metaData, &de) != S_OK) 
            return E_FAIL;
        
        (*dataElement)[2] = de;
        
        return S_OK;
    }

    // TissueForge::types::TVector4<T>

    template <typename T>
    HRESULT toFile(const TissueForge::types::TVector4<T> &dataElement, const MetaData &metaData, IOElement *fileElement) {

        fileElement->type = "Vector4";
        IOElement *xfe = new IOElement();
        IOElement *yfe = new IOElement();
        IOElement *zfe = new IOElement();
        IOElement *wfe = new IOElement();

        if(toFile(dataElement.x(), metaData, xfe) != S_OK || 
        toFile(dataElement.y(), metaData, yfe) != S_OK || 
        toFile(dataElement.z(), metaData, zfe) != S_OK || 
        toFile(dataElement.w(), metaData, zfe) != S_OK) 
            return E_FAIL;
        
        xfe->parent = fileElement;
        yfe->parent = fileElement;
        zfe->parent = fileElement;
        wfe->parent = fileElement;
        fileElement->children["x"] = xfe;
        fileElement->children["y"] = yfe;
        fileElement->children["z"] = zfe;
        fileElement->children["w"] = wfe;

        return S_OK;
    }

    template <typename T>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, TissueForge::types::TVector4<T> *dataElement) {

        T de;
        auto feItr = fileElement.children.find("x");
        if(feItr == fileElement.children.end() || fromFile(*feItr->second, metaData, &de) != S_OK) 
            return E_FAIL;
        
        (*dataElement)[0] = de;

        feItr = fileElement.children.find("y");
        if(feItr == fileElement.children.end() || fromFile(*feItr->second, metaData, &de) != S_OK) 
            return E_FAIL;
        
        (*dataElement)[1] = de;

        feItr = fileElement.children.find("z");
        if(feItr == fileElement.children.end() || fromFile(*feItr->second, metaData, &de) != S_OK) 
            return E_FAIL;
        
        (*dataElement)[2] = de;

        feItr = fileElement.children.find("w");
        if(feItr == fileElement.children.end() || fromFile(*feItr->second, metaData, &de) != S_OK) 
            return E_FAIL;
        
        (*dataElement)[3] = de;
        
        return S_OK;
    }

    // TissueForge::types::TMatrix3<T>

    template <typename T>
    HRESULT toFile(const TissueForge::types::TMatrix3<T> &dataElement, const MetaData &metaData, IOElement *fileElement) {
        
        fileElement->type = "Matrix3";

        for(unsigned int i = 0; i < 3; i++) {
            for (unsigned int j = 0; j < 3; j++) { 
                std::string key = std::to_string(i) + std::to_string(j);
                IOElement *fe = new IOElement();
                if(toFile(dataElement[i][j], metaData, fe) != S_OK) 
                    return E_FAIL;
                fe->parent = fileElement;
                fileElement->children[key] = fe;
            }
        }

        return S_OK;
    }

    template <typename T>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, TissueForge::types::TMatrix3<T> *dataElement) {

        T de;

        for(unsigned int i = 0; i < 3; i++) {
            for(unsigned int j = 0; j < 3; j++) { 
                std::string key = std::to_string(i) + std::to_string(j);
                auto feItr = fileElement.children.find(key);
                if(feItr == fileElement.children.end() || fromFile(*feItr->second, metaData, &de) != S_OK) 
                    return E_FAIL;
                (*dataElement)[i][j] = de;
            }
        }

        return S_OK;
    }

    // TissueForge::types::TMatrix4<T>

    template <typename T>
    HRESULT toFile(const TissueForge::types::TMatrix4<T> &dataElement, const MetaData &metaData, IOElement *fileElement) {
        
        fileElement->type = "Matrix4";

        for(unsigned int i = 0; i < 4; i++) {
            for (unsigned int j = 0; j < 4; j++) { 
                std::string key = std::to_string(i) + std::to_string(j);
                IOElement *fe = new IOElement();
                if(toFile(dataElement[i][j], metaData, fe) != S_OK) 
                    return E_FAIL;
                fe->parent = fileElement;
                fileElement->children[key] = fe;
            }
        }

        return S_OK;
    }

    template <typename T>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, TissueForge::types::TMatrix4<T> *dataElement) {

        T de;

        for(unsigned int i = 0; i < 4; i++) {
            for(unsigned int j = 0; j < 4; j++) { 
                std::string key = std::to_string(i) + std::to_string(j);
                auto feItr = fileElement.children.find(key);
                if(feItr == fileElement.children.end() || fromFile(*feItr->second, metaData, &de) != S_OK) 
                    return E_FAIL;
                (*dataElement)[i][j] = de;
            }
        }

        return S_OK;
    }

    // TissueForge::types::TQuaternion<T>

    template <typename T>
    HRESULT toFile(const TissueForge::types::TQuaternion<T> &dataElement, const MetaData &metaData, IOElement *fileElement) {

        fileElement->type = "Quaternion";
        IOElement *vfe = new IOElement();
        IOElement *sfe = new IOElement();
        
        if(toFile(dataElement.vector(), metaData, vfe) != S_OK || toFile(dataElement.scalar(), metaData, sfe) != S_OK) 
            return E_FAIL;

        vfe->parent = fileElement;
        sfe->parent = fileElement;
        fileElement->children["vector"] = vfe;
        fileElement->children["scalar"] = sfe;

        return S_OK;
    }

    template <typename T>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, TissueForge::types::TQuaternion<T> *dataElement) { 

        std::vector<T> vde;
        T sde;

        auto feItr = fileElement.children.find("vector");
        if(feItr == fileElement.children.end() || fromFile(*feItr->second, metaData, &vde) != S_OK) 
            return E_FAIL;
        dataElement->vector() = vde;

        feItr = fileElement.children.find("scalar");
        if(feItr == fileElement.children.end() || fromFile(*feItr->second, metaData, &sde) != S_OK) 
            return E_FAIL;
        dataElement->scalar() = sde;

        return S_OK;
    }


    // Built-in implementations


    // char

    template <>
    HRESULT toFile(const char &dataElement, const MetaData &metaData, IOElement *fileElement);

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, char *dataElement);

    // signed char

    template <>
    HRESULT toFile(const signed char &dataElement, const MetaData &metaData, IOElement *fileElement);

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, signed char *dataElement);

    // unsigned char

    template <>
    HRESULT toFile(const unsigned char &dataElement, const MetaData &metaData, IOElement *fileElement);

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, unsigned char *dataElement);

    // short

    template <>
    HRESULT toFile(const short &dataElement, const MetaData &metaData, IOElement *fileElement);

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, short *dataElement);

    // unsigned short

    template <>
    HRESULT toFile(const unsigned short &dataElement, const MetaData &metaData, IOElement *fileElement);

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, unsigned short *dataElement);

    // int

    template <>
    HRESULT toFile(const int &dataElement, const MetaData &metaData, IOElement *fileElement);

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, int *dataElement);

    // unsigned int

    template <>
    HRESULT toFile(const unsigned int &dataElement, const MetaData &metaData, IOElement *fileElement);

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, unsigned int *dataElement);

    // bool

    template <>
    HRESULT toFile(const bool &dataElement, const MetaData &metaData, IOElement *fileElement);

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, bool *dataElement);

    // long

    template <>
    HRESULT toFile(const long &dataElement, const MetaData &metaData, IOElement *fileElement);

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, long *dataElement);

    // unsigned long

    template <>
    HRESULT toFile(const unsigned long &dataElement, const MetaData &metaData, IOElement *fileElement);

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, unsigned long *dataElement);

    // long long

    template <>
    HRESULT toFile(const long long &dataElement, const MetaData &metaData, IOElement *fileElement);

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, long long *dataElement);

    // unsigned long long

    template <>
    HRESULT toFile(const unsigned long long &dataElement, const MetaData &metaData, IOElement *fileElement);

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, unsigned long long *dataElement);

    // float

    template <>
    HRESULT toFile(const float &dataElement, const MetaData &metaData, IOElement *fileElement);

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, float *dataElement);

    // double

    template <>
    HRESULT toFile(const double &dataElement, const MetaData &metaData, IOElement *fileElement);

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, double *dataElement);

    // string

    template <>
    HRESULT toFile(const std::string &dataElement, const MetaData &metaData, IOElement *fileElement);

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, std::string *dataElement);

    // Containers

    // set

    template <typename T>
    HRESULT toFile(const std::set<T> &dataElement, const MetaData &metaData, IOElement *fileElement) {
        fileElement->type = "set";
        fileElement->children.reserve(dataElement.size());
        unsigned int i = 0;
        for(auto de : dataElement) {
            IOElement *fe = new IOElement();
            if(toFile(de, metaData, fe) != S_OK) 
                return E_FAIL;
            
            fe->parent = fileElement;
            fileElement->children[std::to_string(i)] = fe;
            i++;
        }
        return S_OK;
    }

    template <typename T>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, std::set<T> *dataElement) {
        unsigned int numEls = fileElement.children.size();
        for(unsigned int i = 0; i < numEls; i++) {
            T de;
            auto itr = fileElement.children.find(std::to_string(i));
            if(itr == fileElement.children.end()) 
                return E_FAIL;
            if(fromFile(*itr->second, metaData, &de) != S_OK) 
                return E_FAIL;
            dataElement->insert(de);
        }
        return S_OK;
    }

    // unordered_set

    template <typename T>
    HRESULT toFile(const std::unordered_set<T> &dataElement, const MetaData &metaData, IOElement *fileElement) {
        fileElement->type = "unordered_set";
        fileElement->children.reserve(dataElement.size());
        unsigned int i = 0;
        for(auto de : dataElement) {
            IOElement *fe = new IOElement();
            if(toFile(de, metaData, fe) != S_OK) 
                return E_FAIL;
            
            fe->parent = fileElement;
            fileElement->children[std::to_string(i)] = fe;
            i++;
        }
        return S_OK;
    }

    template <typename T>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, std::unordered_set<T> *dataElement) {
        unsigned int numEls = fileElement.children.size();
        for(unsigned int i = 0; i < numEls; i++) {
            T de;
            auto itr = fileElement.children.find(std::to_string(i));
            if(itr == fileElement.children.end()) 
                return E_FAIL;
            if(fromFile(*itr->second, metaData, &de) != S_OK) 
                return E_FAIL;
            dataElement->insert(de);
        }
        return S_OK;
    }

    // vector

    template <typename T>
    HRESULT toFile(const std::vector<T> &dataElement, const MetaData &metaData, IOElement *fileElement) {
        fileElement->type = "vector";
        fileElement->children.reserve(dataElement.size());
        for(unsigned int i = 0; i < dataElement.size(); i++) {
            IOElement *fe = new IOElement();
            if(toFile(dataElement[i], metaData, fe) != S_OK) 
                return E_FAIL;
            
            fe->parent = fileElement;
            fileElement->children[std::to_string(i)] = fe;
        }
        return S_OK;
    }

    template <typename T>
    HRESULT toFile(std::vector<T*> dataElement, const MetaData &metaData, IOElement *fileElement) {
        fileElement->type = "vector";
        fileElement->children.reserve(dataElement.size());
        for(unsigned int i = 0; i < dataElement.size(); i++) {
            IOElement *fe = new IOElement();
            if(toFile<T>(dataElement[i], metaData, fe) != S_OK) 
                return E_FAIL;
            
            fe->parent = fileElement;
            fileElement->children[std::to_string(i)] = fe;
        }
        return S_OK;
    }

    template <typename T>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, std::vector<T> *dataElement) {
        unsigned int numEls = fileElement.children.size();
        dataElement->reserve(numEls);
        for(unsigned int i = 0; i < numEls; i++) {
            T de;
            auto itr = fileElement.children.find(std::to_string(i));
            if(itr == fileElement.children.end()) 
                return E_FAIL;
            if(fromFile(*itr->second, metaData, &de) != S_OK) 
                return E_FAIL;
            dataElement->push_back(de);
        }
        return S_OK;
    }

    // map

    template <typename S, typename T>
    HRESULT toFile(const std::map<S, T> &dataElement, const MetaData &metaData, IOElement *fileElement) {
        fileElement->type = "map";
        
        std::vector<S> keysde;
        std::vector<T> valsde;

        for(typename std::map<S, T>::iterator de = dataElement.begin(); de != dataElement.end(); de++) {
            keysde.push_back(de->first);
            valsde.push_back(de->second);
        }

        IOElement *keysfe = new IOElement();
        IOElement *valsfe = new IOElement();
        if(toFile(keysde, metaData, keysfe) != S_OK || toFile(valsde, metaData, valsfe) != S_OK) 
            return E_FAIL;
        
        keysfe->parent = fileElement;
        valsfe->parent = fileElement;
        fileElement->children["keys"] = keysfe;
        fileElement->children["values"] = valsfe;

        return S_OK;
    }

    template <typename S, typename T>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, std::map<S, T> *dataElement) {
        
        auto keysfeItr = fileElement.children.find("keys");
        if(keysfeItr == fileElement.children.end())
            return E_FAIL;
        IOElement *keysfe = keysfeItr->second;

        auto valsfeItr = fileElement.children.find("values");
        if(valsfeItr == fileElement.children.end())
            return E_FAIL;
        IOElement *valsfe = valsfeItr->second;

        std::vector<S> keysde;
        std::vector<T> valsde;
        if(fromFile(*keysfe, metaData, &keysde) != S_OK || fromFile(*valsfe, metaData, &valsde) != S_OK) 
            return E_FAIL;
        
        for(unsigned int i = 0; i < keysde.size(); i++) {
            (*dataElement)[keysde[i]] = valsde[i];
        }

        return S_OK;
    }

    // unordered_map

    template <typename S, typename T>
    HRESULT toFile(const std::unordered_map<S, T> &dataElement, const MetaData &metaData, IOElement *fileElement) {
        fileElement->type = "unordered_map";
        
        std::vector<S> keysde;
        std::vector<T> valsde;
        
        for(auto de = dataElement.begin(); de != dataElement.end(); de++) {
            keysde.push_back(de->first);
            valsde.push_back(de->second);
        }

        IOElement *keysfe = new IOElement();
        IOElement *valsfe = new IOElement();
        if(toFile(keysde, metaData, keysfe) != S_OK || toFile(valsde, metaData, valsfe) != S_OK) 
            return E_FAIL;
        
        keysfe->parent = fileElement;
        valsfe->parent = fileElement;
        fileElement->children["keys"] = keysfe;
        fileElement->children["values"] = valsfe;

        return S_OK;
    }

    template <typename S, typename T>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, std::unordered_map<S, T> *dataElement) {
        
        auto keysfeItr = fileElement.children.find("keys");
        if(keysfeItr == fileElement.children.end())
            return E_FAIL;
        IOElement *keysfe = keysfeItr->second;

        auto valsfeItr = fileElement.children.find("values");
        if(valsfeItr == fileElement.children.end())
            return E_FAIL;
        IOElement *valsfe = valsfeItr->second;

        std::vector<S> keysde;
        std::vector<T> valsde;
        if(fromFile(*keysfe, metaData, &keysde) != S_OK || fromFile(*valsfe, metaData, &valsde) != S_OK) 
            return E_FAIL;
        
        for(unsigned int i = 0; i < keysde.size(); i++) {
            (*dataElement)[keysde[i]] = valsde[i];
        }

        return S_OK;
    }

};

#endif // _SOURCE_IO_TF_IO_H_