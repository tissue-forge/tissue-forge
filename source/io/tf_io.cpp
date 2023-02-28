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

#include <tfParticle.h>

#include "tf_io.h"

#include <limits>


using namespace TissueForge;


#define TF_IOEASYTOFILE(dataElement, typeName) \
    fileElement.get()->value = std::to_string(dataElement); \
    fileElement.get()->type = typeName; \
    return S_OK;

#define TF_IOFINDSAFE(fileElement, itrName, keyName, valObj) \
    auto itrName = fileElement.el->children.find(keyName); \
    if(itrName == fileElement.el->children.end()) \
        return E_FAIL; \
    valObj = itrName->second;


namespace TissueForge::io {


    // built-in types


    // char

    template <>
    HRESULT toFile(const char &dataElement, const MetaData &metaData, IOElement &fileElement) {
        TF_IOEASYTOFILE(dataElement, "char");
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, char *dataElement) {
        *dataElement = IOElement::value(fileElement)[0];
        return S_OK;
    }

    // signed char

    template <>
    HRESULT toFile(const signed char &dataElement, const MetaData &metaData, IOElement &fileElement) {
        TF_IOEASYTOFILE(dataElement, "signed_char");
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, signed char *dataElement) {
        *dataElement = IOElement::value(fileElement)[0];
        return S_OK;
    }

    // unsigned char

    template <>
    HRESULT toFile(const unsigned char &dataElement, const MetaData &metaData, IOElement &fileElement) {
        TF_IOEASYTOFILE(dataElement, "unsigned_char");
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, unsigned char *dataElement) {
        *dataElement = IOElement::value(fileElement)[0];
        return S_OK;
    }

    // short

    template <>
    HRESULT toFile(const short &dataElement, const MetaData &metaData, IOElement &fileElement) {
        if(toFile((int)dataElement, metaData, fileElement) != S_OK) 
            return E_FAIL;
        fileElement.get()->type = "short";
        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, short *dataElement) {
        int i;

        if(fromFile(fileElement, metaData, &i) != S_OK) 
            return E_FAIL;

        if(std::abs(i) >= std::numeric_limits<short>::max()) {
            tf_exp(std::range_error("Value exceeds numerical limits"));
            return E_FAIL;
        }

        *dataElement = i;

        return S_OK;
    }

    // unsigned short

    template <>
    HRESULT toFile(const unsigned short &dataElement, const MetaData &metaData, IOElement &fileElement) {
        if(toFile((unsigned int)dataElement, metaData, fileElement) != S_OK) 
            return E_FAIL;
        fileElement.get()->type = "unsigned_short";
        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, unsigned short *dataElement) {
        unsigned int i;
        
        if(fromFile(fileElement, metaData, &i) != S_OK) 
            return E_FAIL;

        if(i >= std::numeric_limits<unsigned short>::max()) {
            tf_exp(std::range_error("Value exceeds numerical limits"));
            return E_FAIL;
        }

        *dataElement = i;

        return S_OK;
    }

    // int

    template <>
    HRESULT toFile(const int &dataElement, const MetaData &metaData, IOElement &fileElement) {
        TF_IOEASYTOFILE(dataElement, "int");
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, int *dataElement) {
        *dataElement = std::stoi(IOElement::value(fileElement));
        return S_OK;
    }

    // unsigned int

    template <>
    HRESULT toFile(const unsigned int &dataElement, const MetaData &metaData, IOElement &fileElement) {
        TF_IOEASYTOFILE(dataElement, "unsigned_int");
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, unsigned int *dataElement) {
        *dataElement = std::stoul(IOElement::value(fileElement));
        return S_OK;
    }

    // bool

    template <>
    HRESULT toFile(const bool &dataElement, const MetaData &metaData, IOElement &fileElement) {
        TF_IOEASYTOFILE(dataElement, "bool");
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, bool *dataElement) {
        unsigned int i;
        if(fromFile(fileElement, metaData, &i) != S_OK) 
            return E_FAIL;
        *dataElement = i;
        return S_OK;
    }

    // long

    template <>
    HRESULT toFile(const long &dataElement, const MetaData &metaData, IOElement &fileElement) {
        TF_IOEASYTOFILE(dataElement, "long");
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, long *dataElement) {
        *dataElement = std::stol(IOElement::value(fileElement));
        return S_OK;
    }

    // unsigned long

    template <>
    HRESULT toFile(const unsigned long &dataElement, const MetaData &metaData, IOElement &fileElement) {
        TF_IOEASYTOFILE(dataElement, "unsigned_long");
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, unsigned long *dataElement) {
        *dataElement = std::stoul(IOElement::value(fileElement));
        return S_OK;
    }

    // long long

    template <>
    HRESULT toFile(const long long &dataElement, const MetaData &metaData, IOElement &fileElement) {
        TF_IOEASYTOFILE(dataElement, "long_long");
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, long long *dataElement) {
        *dataElement = std::stoll(IOElement::value(fileElement));
        return S_OK;
    }

    // unsigned long long

    template <>
    HRESULT toFile(const unsigned long long &dataElement, const MetaData &metaData, IOElement &fileElement) {
        TF_IOEASYTOFILE(dataElement, "unsigned_long_long");
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, unsigned long long *dataElement) {
        *dataElement = std::stoull(IOElement::value(fileElement));
        return S_OK;
    }

    // float

    template <>
    HRESULT toFile(const float &dataElement, const MetaData &metaData, IOElement &fileElement) {
        TF_IOEASYTOFILE(dataElement, "float");
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, float *dataElement) {
        *dataElement = std::stof(IOElement::value(fileElement));
        return S_OK;
    }

    // double

    template <>
    HRESULT toFile(const double &dataElement, const MetaData &metaData, IOElement &fileElement) {
        TF_IOEASYTOFILE(dataElement, "double");
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, double *dataElement) {
        *dataElement = std::stod(IOElement::value(fileElement));
        return S_OK;
    }

    // string

    template <>
    HRESULT toFile(const std::string &dataElement, const MetaData &metaData, IOElement &fileElement) {
        fileElement.get()->value = dataElement;
        fileElement.get()->type = "string";
        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, std::string *dataElement) {
        *dataElement = std::string(IOElement::value(fileElement));
        return S_OK;
    }

    // Containers


    // Tissue Forge types


    // MetaData

    template <>
    HRESULT toFile(const MetaData &dataElement, const MetaData &metaData, IOElement &fileElement) {
        TF_IOTOEASY(fileElement, metaData, "versionMajor", dataElement.versionMajor);
        TF_IOTOEASY(fileElement, metaData, "versionMinor", dataElement.versionMinor);
        TF_IOTOEASY(fileElement, metaData, "versionPatch", dataElement.versionPatch);

        fileElement.get()->type = "MetaData";

        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, MetaData *dataElement) { 

        TF_IOFROMEASY(fileElement, metaData, "versionMajor", &dataElement->versionMajor);
        TF_IOFROMEASY(fileElement, metaData, "versionMinor", &dataElement->versionMinor);
        TF_IOFROMEASY(fileElement, metaData, "versionPatch", &dataElement->versionPatch);

        return S_OK;
    }

};
