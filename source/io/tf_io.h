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
#include <memory>


#define TF_IOTOEASY(fileElement, metaData, key, member) {\
    ::TissueForge::io::IOElement _fe = ::TissueForge::io::IOElement::create(); \
    if(::TissueForge::io::toFile(member, metaData, _fe) != S_OK)  \
        return E_FAIL; \
    fileElement.addChild(_fe, key);}

#define TF_IOFROMEASY(fileElement, metaData, key, member_p) {\
    ::TissueForge::io::IOChildMap _children = ::TissueForge::io::IOElement::children(fileElement); \
    ::TissueForge::io::IOChildMap::const_iterator _feItr = _children.find(key); \
    if(_feItr == _children.end() || ::TissueForge::io::fromFile(_feItr->second, metaData, member_p) != S_OK) \
        return E_FAIL; \
    }


namespace TissueForge::io {


    /**
     * @brief Intermediate I/O class for reading/writing 
     * Tissue Forge objects to/from file/string. 
     * 
     */
    template <typename T> 
    struct _IOElementT {
        std::string type;
        std::string value;
        T parent;
        std::unordered_map<std::string, T> children;
    };
    using _IOElement = _IOElementT<struct IOElement>;
    using IOChildMap = std::unordered_map<std::string, struct IOElement>;

    /**
     * @brief Container for _IOElement. 
     * 
     */
    struct IOElement {
        std::weak_ptr<_IOElement> el;

        IOElement() {}
        IOElement(const IOElement &other) {
            el = other.el;
        }
        
        static IOElement create() {
            IOElement result;
            result._init();
            return result;
        }

        IOElement clone() {
            IOElement result = IOElement::create();
            result._el = this->_el;
            result.el = this->_el;
            return result;
        }

        std::shared_ptr<_IOElement> get() { 
            if(el.expired()) 
                _init();
            return el.lock();
        }

        void addChild(IOElement &child, const std::string &key) {
            get()->children[key] = IOElement(child);
            child.get()->parent = IOElement(*this);
        }

        void reset() {
            _init();
        }

        bool isEmpty() { return get()->type.size() == 0; }

        static std::string type(const IOElement &_e) { return const_cast<IOElement&>(_e).get()->type; };
        static std::string value(const IOElement &_e) { return const_cast<IOElement&>(_e).get()->value; };
        static IOElement parent(const IOElement &_e) { return const_cast<IOElement&>(_e).get()->parent; };
        static std::unordered_map<std::string, IOElement> children(const IOElement &_e) { return const_cast<IOElement&>(_e).get()->children; };

    private:

        void _init() {
            _el = std::make_shared<_IOElement>();
            el = _el;
        }
        
        std::shared_ptr<_IOElement> _el;

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
    HRESULT toFile(const T &dataElement, const MetaData &metaData, IOElement &fileElement);

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
    HRESULT toFile(T *dataElement, const MetaData &metaData, IOElement &fileElement);

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
    HRESULT toFile(const MetaData &dataElement, const MetaData &metaData, IOElement &fileElement);

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, MetaData *dataElement);

    // TissueForge::types::TVector2<T>

    template <typename T>
    HRESULT toFile(const TissueForge::types::TVector2<T> &dataElement, const MetaData &metaData, IOElement &fileElement) {
        
        fileElement.get()->type = "Vector2";
        TF_IOTOEASY(fileElement, metaData, "x", dataElement.x());
        TF_IOTOEASY(fileElement, metaData, "y", dataElement.y());

        return S_OK;
    }

    template <typename T>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, TissueForge::types::TVector2<T> *dataElement) {

        T de;
        
        TF_IOFROMEASY(fileElement, metaData, "x", &de);
        (*dataElement)[0] = de;

        TF_IOFROMEASY(fileElement, metaData, "y", &de);
        (*dataElement)[1] = de;
        
        return S_OK;
    }

    // TissueForge::types::TVector3<T>

    template <typename T>
    HRESULT toFile(const TissueForge::types::TVector3<T> &dataElement, const MetaData &metaData, IOElement &fileElement) {

        fileElement.get()->type = "Vector3";
        TF_IOTOEASY(fileElement, metaData, "x", dataElement.x());
        TF_IOTOEASY(fileElement, metaData, "y", dataElement.y());
        TF_IOTOEASY(fileElement, metaData, "z", dataElement.z());

        return S_OK;
    }

    template <typename T>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, TissueForge::types::TVector3<T> *dataElement) {

        T de;

        TF_IOFROMEASY(fileElement, metaData, "x", &de);
        (*dataElement)[0] = de;

        TF_IOFROMEASY(fileElement, metaData, "y", &de);
        (*dataElement)[1] = de;

        TF_IOFROMEASY(fileElement, metaData, "z", &de);
        (*dataElement)[2] = de;
        
        return S_OK;
    }

    // TissueForge::types::TVector4<T>

    template <typename T>
    HRESULT toFile(const TissueForge::types::TVector4<T> &dataElement, const MetaData &metaData, IOElement &fileElement) {

        fileElement.get()->type = "Vector4";
        TF_IOTOEASY(fileElement, metaData, "x", dataElement.x());
        TF_IOTOEASY(fileElement, metaData, "y", dataElement.y());
        TF_IOTOEASY(fileElement, metaData, "z", dataElement.z());
        TF_IOTOEASY(fileElement, metaData, "w", dataElement.w());

        return S_OK;
    }

    template <typename T>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, TissueForge::types::TVector4<T> *dataElement) {

        T de;

        TF_IOFROMEASY(fileElement, metaData, "x", &de);
        (*dataElement)[0] = de;

        TF_IOFROMEASY(fileElement, metaData, "y", &de);
        (*dataElement)[1] = de;

        TF_IOFROMEASY(fileElement, metaData, "z", &de);
        (*dataElement)[2] = de;

        TF_IOFROMEASY(fileElement, metaData, "w", &de);
        (*dataElement)[3] = de;
        
        return S_OK;
    }

    // TissueForge::types::TMatrix3<T>

    template <typename T>
    HRESULT toFile(const TissueForge::types::TMatrix3<T> &dataElement, const MetaData &metaData, IOElement &fileElement) {
        
        fileElement.get()->type = "Matrix3";

        for(unsigned int i = 0; i < 3; i++) {
            for (unsigned int j = 0; j < 3; j++) { 
                std::string key = std::to_string(i) + std::to_string(j);
                TF_IOTOEASY(fileElement, metaData, key, dataElement[i][j]);
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
                TF_IOFROMEASY(fileElement, metaData, key, &de);
                (*dataElement)[i][j] = de;
            }
        }

        return S_OK;
    }

    // TissueForge::types::TMatrix4<T>

    template <typename T>
    HRESULT toFile(const TissueForge::types::TMatrix4<T> &dataElement, const MetaData &metaData, IOElement &fileElement) {
        
        fileElement.get()->type = "Matrix4";

        for(unsigned int i = 0; i < 4; i++) {
            for (unsigned int j = 0; j < 4; j++) { 
                std::string key = std::to_string(i) + std::to_string(j);
                TF_IOTOEASY(fileElement, metaData, key, dataElement[i][j]);
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
                TF_IOFROMEASY(fileElement, metaData, key, &de);
                (*dataElement)[i][j] = de;
            }
        }

        return S_OK;
    }

    // TissueForge::types::TQuaternion<T>

    template <typename T>
    HRESULT toFile(const TissueForge::types::TQuaternion<T> &dataElement, const MetaData &metaData, IOElement &fileElement) {

        fileElement.get()->type = "Quaternion";
        TF_IOTOEASY(fileElement, metaData, "vector", dataElement.vector());
        TF_IOTOEASY(fileElement, metaData, "scalar", dataElement.scalar());

        return S_OK;
    }

    template <typename T>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, TissueForge::types::TQuaternion<T> *dataElement) { 

        std::vector<T> vde;
        T sde;

        TF_IOFROMEASY(fileElement, metaData, "vector", &dataElement->vector());
        TF_IOFROMEASY(fileElement, metaData, "scalar", &dataElement->scalar());

        return S_OK;
    }


    // Built-in implementations


    // char

    template <>
    HRESULT toFile(const char &dataElement, const MetaData &metaData, IOElement &fileElement);

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, char *dataElement);

    // signed char

    template <>
    HRESULT toFile(const signed char &dataElement, const MetaData &metaData, IOElement &fileElement);

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, signed char *dataElement);

    // unsigned char

    template <>
    HRESULT toFile(const unsigned char &dataElement, const MetaData &metaData, IOElement &fileElement);

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, unsigned char *dataElement);

    // short

    template <>
    HRESULT toFile(const short &dataElement, const MetaData &metaData, IOElement &fileElement);

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, short *dataElement);

    // unsigned short

    template <>
    HRESULT toFile(const unsigned short &dataElement, const MetaData &metaData, IOElement &fileElement);

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, unsigned short *dataElement);

    // int

    template <>
    HRESULT toFile(const int &dataElement, const MetaData &metaData, IOElement &fileElement);

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, int *dataElement);

    // unsigned int

    template <>
    HRESULT toFile(const unsigned int &dataElement, const MetaData &metaData, IOElement &fileElement);

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, unsigned int *dataElement);

    // bool

    template <>
    HRESULT toFile(const bool &dataElement, const MetaData &metaData, IOElement &fileElement);

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, bool *dataElement);

    // long

    template <>
    HRESULT toFile(const long &dataElement, const MetaData &metaData, IOElement &fileElement);

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, long *dataElement);

    // unsigned long

    template <>
    HRESULT toFile(const unsigned long &dataElement, const MetaData &metaData, IOElement &fileElement);

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, unsigned long *dataElement);

    // long long

    template <>
    HRESULT toFile(const long long &dataElement, const MetaData &metaData, IOElement &fileElement);

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, long long *dataElement);

    // unsigned long long

    template <>
    HRESULT toFile(const unsigned long long &dataElement, const MetaData &metaData, IOElement &fileElement);

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, unsigned long long *dataElement);

    // float

    template <>
    HRESULT toFile(const float &dataElement, const MetaData &metaData, IOElement &fileElement);

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, float *dataElement);

    // double

    template <>
    HRESULT toFile(const double &dataElement, const MetaData &metaData, IOElement &fileElement);

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, double *dataElement);

    // string

    template <>
    HRESULT toFile(const std::string &dataElement, const MetaData &metaData, IOElement &fileElement);

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, std::string *dataElement);

    // Containers

    // set

    template <typename T>
    HRESULT toFile(const std::set<T> &dataElement, const MetaData &metaData, IOElement &fileElement) {
        fileElement.get()->type = "set";
        fileElement.get()->children.reserve(dataElement.size());
        unsigned int i = 0;
        for(auto de : dataElement) {
            TF_IOTOEASY(fileElement, metaData, std::to_string(i), de);
            i++;
        }
        return S_OK;
    }

    template <typename T>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, std::set<T> *dataElement) {
        unsigned int numEls = IOElement::children(fileElement).size();
        for(unsigned int i = 0; i < numEls; i++) {
            T de;
            TF_IOFROMEASY(fileElement, metaData, std::to_string(i), &de);
            dataElement->insert(de);
        }
        return S_OK;
    }

    // unordered_set

    template <typename T>
    HRESULT toFile(const std::unordered_set<T> &dataElement, const MetaData &metaData, IOElement &fileElement) {
        fileElement.get()->type = "unordered_set";
        fileElement.get()->children.reserve(dataElement.size());
        unsigned int i = 0;
        for(auto de : dataElement) {
            TF_IOTOEASY(fileElement, metaData, std::to_string(i), de);
            i++;
        }
        return S_OK;
    }

    template <typename T>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, std::unordered_set<T> *dataElement) {
        unsigned int numEls = IOElement::children(fileElement).size();
        for(unsigned int i = 0; i < numEls; i++) {
            T de;
            TF_IOFROMEASY(fileElement, metaData, std::to_string(i), &de);
            dataElement->insert(de);
        }
        return S_OK;
    }

    // vector

    template <typename T>
    HRESULT toFile(const std::vector<T> &dataElement, const MetaData &metaData, IOElement &fileElement) {
        fileElement.get()->type = "vector";
        fileElement.get()->children.reserve(dataElement.size());
        for(unsigned int i = 0; i < dataElement.size(); i++) {
            TF_IOTOEASY(fileElement, metaData, std::to_string(i), dataElement[i]);
        }
        return S_OK;
    }

    template <typename T>
    HRESULT toFile(std::vector<T*> dataElement, const MetaData &metaData, IOElement &fileElement) {
        fileElement.get()->type = "vector";
        fileElement.get()->children.reserve(dataElement.size());
        for(unsigned int i = 0; i < dataElement.size(); i++) {
            TF_IOTOEASY(fileElement, metaData, std::to_string(i), dataElement[i]);
        }
        return S_OK;
    }

    template <typename T>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, std::vector<T> *dataElement) {
        unsigned int numEls = IOElement::children(fileElement).size();
        dataElement->reserve(numEls);
        for(unsigned int i = 0; i < numEls; i++) {
            T de;
            TF_IOFROMEASY(fileElement, metaData, std::to_string(i), &de);
            dataElement->push_back(de);
        }
        return S_OK;
    }

    // map

    template <typename S, typename T>
    HRESULT toFile(const std::map<S, T> &dataElement, const MetaData &metaData, IOElement &fileElement) {
        fileElement.get()->type = "map";
        
        std::vector<S> keysde;
        std::vector<T> valsde;

        for(typename std::map<S, T>::iterator de = dataElement.begin(); de != dataElement.end(); de++) {
            keysde.push_back(de->first);
            valsde.push_back(de->second);
        }

        TF_IOTOEASY(fileElement, metaData, "keys", keysde);
        TF_IOTOEASY(fileElement, metaData, "vals", valsde);

        return S_OK;
    }

    template <typename S, typename T>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, std::map<S, T> *dataElement) {
        
        std::vector<S> keysde;
        std::vector<T> valsde;

        TF_IOFROMEASY(fileElement, metaData, "keys", &keysde);
        TF_IOFROMEASY(fileElement, metaData, "vals", &valsde);

        for(unsigned int i = 0; i < keysde.size(); i++) {
            (*dataElement)[keysde[i]] = valsde[i];
        }

        return S_OK;
    }

    // unordered_map

    template <typename S, typename T>
    HRESULT toFile(const std::unordered_map<S, T> &dataElement, const MetaData &metaData, IOElement &fileElement) {
        fileElement.get()->type = "unordered_map";
        
        std::vector<S> keysde;
        std::vector<T> valsde;
        
        for(auto de = dataElement.begin(); de != dataElement.end(); de++) {
            keysde.push_back(de->first);
            valsde.push_back(de->second);
        }

        TF_IOTOEASY(fileElement, metaData, "keys", keysde);
        TF_IOTOEASY(fileElement, metaData, "vals", valsde);

        return S_OK;
    }

    template <typename S, typename T>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, std::unordered_map<S, T> *dataElement) {
        
        std::vector<S> keysde;
        std::vector<T> valsde;

        TF_IOFROMEASY(fileElement, metaData, "keys", &keysde);
        TF_IOFROMEASY(fileElement, metaData, "vals", &valsde);
        
        for(unsigned int i = 0; i < keysde.size(); i++) {
            (*dataElement)[keysde[i]] = valsde[i];
        }

        return S_OK;
    }

};

#endif // _SOURCE_IO_TF_IO_H_