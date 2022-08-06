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

#include <nlohmann/json.hpp>

#include <tfSimulator.h>
#include <tfLogger.h>
#include <tf_util.h>
#include <tfError.h>

#include <fstream>
#include <iostream>

#include "tfFIO.h"


using json = nlohmann::json;


namespace TissueForge::io {


    const std::string FIO::KEY_TYPE = "IOType";
    const std::string FIO::KEY_VALUE = "IOValue";

    const std::string FIO::KEY_METADATA = "MetaData";
    const std::string FIO::KEY_SIMULATOR = "Simulator";
    const std::string FIO::KEY_UNIVERSE = "Universe";
    const std::string FIO::KEY_MODULES = "Modules";


    template <>
    HRESULT toFile(const json &dataElement, const MetaData &metaData, IOElement *fileElement) {

        IOElement *fe;

        for(auto &el : dataElement.items()) {
            auto key = el.key();
            
            if(key == FIO::KEY_TYPE) {
                fileElement->type = el.value().get<std::string>();
            }
            else if(key == FIO::KEY_VALUE) {
                fileElement->value = el.value().get<std::string>();
            }
            else {
            
                fe = new IOElement();
                if(toFile(el.value(), metaData, fe) != S_OK) 
                    return E_FAIL;
                
                fe->parent = fileElement;
                fileElement->children[key] = fe;

            }
        }

        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, json *dataElement) { 

        try {

            (*dataElement)[FIO::KEY_TYPE] = fileElement.type;
            (*dataElement)[FIO::KEY_VALUE] = fileElement.value;
            for(IOChildMap::const_iterator feItr = fileElement.children.begin(); feItr != fileElement.children.end(); feItr++) {
                json &jv = (*dataElement)[feItr->first.c_str()];
                if(fromFile(*feItr->second, metaData, &jv) != S_OK) 
                    return E_FAIL;
            }

        }
        catch (...) {
            TF_Log(LOG_CRITICAL) << "Could not generate JSON data: " << fileElement.type << ", " << fileElement.value;
            return E_FAIL;
        }

        return S_OK;
    }

    std::string toStr(IOElement *fileElement, const MetaData &metaData) {

        json jroot;

        json jmetadata;
        IOElement femetadata;
        toFile(metaData, MetaData(), &femetadata);

        if(fromFile(femetadata, metaData, &jmetadata) != S_OK) 
            tf_exp(std::runtime_error("Could not translate meta data"));

        jroot[FIO::KEY_METADATA] = jmetadata;

        json jvalue;
        
        if(fromFile(*fileElement, metaData, &jvalue) != S_OK) 
            tf_exp(std::runtime_error("Could not translate data"));

        jroot[FIO::KEY_VALUE] = jvalue;

        return jroot.dump(4);
    }

    std::string toStr(IOElement *fileElement) {

        return toStr(fileElement, MetaData());

    }

    IOElement *fromStr(const std::string &str, const MetaData &metaData) {

        json jroot = json::parse(str);

        IOElement *fe = new IOElement();

        if(toFile(jroot[FIO::KEY_VALUE], metaData, fe) != S_OK) 
            tf_exp(std::runtime_error("Could not translate data"));

        return fe;
    }

    IOElement *fromStr(const std::string &str) {

        json jroot = json::parse(str);

        IOElement *fevalue = new IOElement(), femetadata;
        MetaData strMetaData, metaData;

        if(toFile(jroot[FIO::KEY_METADATA], metaData, &femetadata) != S_OK) 
            tf_exp(std::runtime_error("Could not parse meta data"));
        
        if(fromFile(femetadata, metaData, &strMetaData) != S_OK) 
            tf_exp(std::runtime_error("Could not translate meta data"));

        if(toFile(jroot[FIO::KEY_VALUE], strMetaData, fevalue) != S_OK) 
            tf_exp(std::runtime_error("Could not translate data"));

        return fevalue;
    }

    static HRESULT gatherElements(std::vector<IOElement*> &elements, IOElement *fileElement) {
        HRESULT ret;

        elements.push_back(fileElement);
        for(auto &e : fileElement->children) 
            if((ret = gatherElements(elements, e.second)) != S_OK) 
                return ret;

        return S_OK;
    }

    HRESULT deleteElement(IOElement **fileElement) { 
        HRESULT result;

        if(fileElement) {

            std::vector<IOElement*> elementsFlat;
            if((result = gatherElements(elementsFlat, *fileElement)) != S_OK) 
                return result;
            elementsFlat = util::unique(elementsFlat);

            for(auto &e : elementsFlat) {
                if(e) {
                    e->parent = NULL;
                    e->children.clear();
                    delete e;
                    e = NULL;
                }
            }

            *fileElement = NULL;

        }

        return S_OK;
    }


    // FIOModule

    void FIOModule::registerIOModule() {
        FIO::registerModule(this->moduleName(), this);
    }

    void FIOModule::load() {
        FIO::loadModule(this->moduleName());
    }


    // FIO


    IOElement *FIO::fromFile(const std::string &loadFilePath) { 

        // Build root node from file contents

        std::ifstream fileContents_ifs(loadFilePath, std::ifstream::binary);
        if(!fileContents_ifs || !fileContents_ifs.good() || fileContents_ifs.fail()) 
            tf_exp(std::runtime_error(std::string("Error loading file: ") + loadFilePath));

        // Create a reader and get root node

        json jroot;

        fileContents_ifs >> jroot;

        fileContents_ifs.close();
        
        TF_Log(LOG_INFORMATION) << "Loaded source: " << loadFilePath;

        // Create root io element and populate from root node

        MetaData metaData, metaDataFile;
        IOElement feMetaData;
        if(::TissueForge::io::toFile(jroot[FIO::KEY_METADATA], metaData, &feMetaData) != S_OK) 
            tf_exp(std::runtime_error("Could not unpack metadata"));
        if(::TissueForge::io::fromFile(feMetaData, metaData, &metaDataFile) != S_OK) 
            tf_exp(std::runtime_error("Could not load metadata"));

        TF_Log(LOG_INFORMATION) << "Got file metadata: " << metaDataFile.versionMajor << "." << metaDataFile.versionMinor << "." << metaDataFile.versionPatch;

        FIO::currentRootElement = new IOElement();
        if(::TissueForge::io::toFile(jroot, metaDataFile, FIO::currentRootElement)) 
            tf_exp(std::runtime_error("Could not load simulation data"));
        
        TF_Log(LOG_INFORMATION) << "Generated i/o from source: " << loadFilePath;
        
        return FIO::currentRootElement;
    }

    IOElement *FIO::generateIORootElement() {

        if(FIO::currentRootElement != NULL) 
            if(FIO::releaseIORootElement() != S_OK) 
                return NULL;

        IOElement *tfData = new IOElement();

        // Add metadata

        MetaData metaData;
        IOElement *feMetaData = new IOElement();
        if(::TissueForge::io::toFile(metaData, metaData, feMetaData) != S_OK) 
            tf_exp(std::runtime_error("Could not store metadata"));
        tfData->children[FIO::KEY_METADATA] = feMetaData;
        feMetaData->parent = tfData;

        // Add simulator
        
        auto simulator = Simulator::get();
        IOElement *feSimulator = new IOElement();
        if(::TissueForge::io::toFile(*simulator, metaData, feSimulator) != S_OK) 
            tf_exp(std::runtime_error("Could not store simulator"));
        tfData->children[FIO::KEY_SIMULATOR] = feSimulator;
        feSimulator->parent = tfData;

        // Add universe
        
        auto universe = Universe::get();
        IOElement *feUniverse = new IOElement();
        if(::TissueForge::io::toFile(*universe, metaData, feUniverse) != S_OK) 
            tf_exp(std::runtime_error("Could not store universe"));
        tfData->children[FIO::KEY_UNIVERSE] = feUniverse;
        feUniverse->parent = tfData;

        // Add modules

        if(FIO::modules == NULL) 
            FIO::modules = new std::unordered_map<std::string, FIOModule*>();

        if(FIO::modules->size() > 0) {
            IOElement *feModules = new IOElement();
            for(auto &itr : *FIO::modules) {
                IOElement *feModule = new IOElement();
                if(itr.second->toFile(metaData, feModule) != S_OK) 
                    tf_exp(std::runtime_error("Could not store module: " + itr.first));
                feModules->children[itr.first] = feModule;
                feModule->parent = feModules;
            }

            tfData->children[FIO::KEY_MODULES] = feModules;
            feModules->parent = tfData;
        }

        FIO::currentRootElement = tfData;

        return tfData;
    }

    HRESULT FIO::releaseIORootElement() { 

        if(FIO::currentRootElement == NULL) 
            return S_OK;

        return deleteElement(&FIO::currentRootElement);
    }

    HRESULT FIO::toFile(const std::string &saveFilePath) { 

        MetaData metaData;
        IOElement *tfData = generateIORootElement();

        // Create root node

        json jroot;

        if(::TissueForge::io::fromFile(*tfData, metaData, &jroot) != S_OK) 
            tf_exp(std::runtime_error("Could not translate final data"));

        // Write

        std::ofstream saveFile(saveFilePath);
        
        saveFile << jroot.dump(4);

        saveFile.close();

        return releaseIORootElement();
    }

    std::string FIO::toString() {

        MetaData metaData;
        IOElement *tfData = generateIORootElement();

        // Create root node

        json jroot;

        if(::TissueForge::io::fromFile(*tfData, metaData, &jroot) != S_OK) 
            tf_exp(std::runtime_error("Could not translate final data"));

        // Write

        std::string result = jroot.dump(4);

        if(releaseIORootElement() != S_OK) 
            tf_exp(std::runtime_error("Could not close root element"));

        return result;
    }

    void FIO::registerModule(const std::string moduleName, FIOModule *module) {
        if(FIO::modules == NULL) 
            FIO::modules = new std::unordered_map<std::string, FIOModule*>();
        
        auto itr = FIO::modules->find(moduleName);
        if(itr != FIO::modules->end()) 
            tf_exp(std::runtime_error("I/O module already registered: " + moduleName));

        (*FIO::modules)[moduleName] = module;
    }

    void FIO::loadModule(const std::string moduleName) {

        if(FIO::modules == NULL) 
            FIO::modules = new std::unordered_map<std::string, FIOModule*>();

        // Get registered module

        auto itr = FIO::modules->find(moduleName);
        if(itr == FIO::modules->end()) 
            tf_exp(std::runtime_error("I/O module not registered: " + moduleName));

        // Validate previous main import
        
        if(FIO::currentRootElement == NULL) 
            tf_exp(std::runtime_error("No import state"));
        
        // Get file metadata
        
        IOElement *feMetaData = FIO::currentRootElement->children[FIO::KEY_METADATA];
        MetaData metaData, metaDataFile;
        if(::TissueForge::io::fromFile(*feMetaData, metaData, &metaDataFile) != S_OK) 
            tf_exp(std::runtime_error("Could not load metadata"));

        // Get modules element
        
        auto mItr = FIO::currentRootElement->children.find(FIO::KEY_MODULES);
        if(mItr == FIO::currentRootElement->children.end()) 
            tf_exp(std::runtime_error("No loaded modules"));
        auto feModules = mItr->second;

        // Get module element
        
        mItr = feModules->children.find(moduleName);
        if(mItr == feModules->children.end()) 
            tf_exp(std::runtime_error("Module data not available: " + moduleName));

        // Issue module import
        
        if((*FIO::modules)[moduleName]->fromFile(metaDataFile, *mItr->second) != S_OK) 
            tf_exp(std::runtime_error("Module import failed: " + moduleName));
    }

    bool FIO::hasImport() {
        return FIO::importSummary != NULL;
    }

};
