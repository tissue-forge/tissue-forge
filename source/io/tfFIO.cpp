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

    const std::string FIO::KEY_ROOT = "Root";
    const std::string FIO::KEY_METADATA = "MetaData";
    const std::string FIO::KEY_SIMULATOR = "Simulator";
    const std::string FIO::KEY_UNIVERSE = "Universe";
    const std::string FIO::KEY_MODULES = "Modules";

    static IOElement _currentRootElement;
    static bool _hasCurrentRootElement = false;

    template <>
    HRESULT toFile(const json &dataElement, const MetaData &metaData, IOElement &fileElement) {

        for(auto &el : dataElement.items()) {
            auto key = el.key();
            
            if(key == FIO::KEY_TYPE) {
                fileElement.get()->type = el.value().get<std::string>();
            }
            else if(key == FIO::KEY_VALUE) {
                fileElement.get()->value = el.value().get<std::string>();
            }
            else {
            
                IOElement fe = IOElement::create();
                if(toFile(el.value(), metaData, fe) != S_OK) 
                    return E_FAIL;
                
                fileElement.addChild(fe, key);

            }
        }

        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, json *dataElement) { 

        std::string fet = IOElement::type(fileElement);
        std::string fev = IOElement::value(fileElement);

        try {

            (*dataElement)[FIO::KEY_TYPE] = fet;
            (*dataElement)[FIO::KEY_VALUE] = fev;
            IOChildMap fec = IOElement::children(fileElement);
            for(IOChildMap::const_iterator feItr = fec.begin(); feItr != fec.end(); feItr++) {
                json &jv = (*dataElement)[feItr->first.c_str()];
                if(fromFile(feItr->second, metaData, &jv) != S_OK) 
                    return E_FAIL;
            }

        }
        catch (...) {
            TF_Log(LOG_CRITICAL) << "Could not generate JSON data: " << fet << ", " << fev;
            return E_FAIL;
        }

        return S_OK;
    }

    std::string toStr(IOElement &fileElement, const MetaData &metaData) {

        json jroot;

        json jmetadata;
        IOElement femetadata;
        toFile(metaData, MetaData(), femetadata);

        if(fromFile(femetadata, metaData, &jmetadata) != S_OK) 
            tf_exp(std::runtime_error("Could not translate meta data"));

        jroot[FIO::KEY_METADATA] = jmetadata;
        
        if(fromFile(fileElement, metaData, &jroot) != S_OK) 
            tf_exp(std::runtime_error("Could not translate data"));

        std::string result = jroot.dump(4);

        jroot.clear();
        jmetadata.clear();

        return result;
    }

    std::string toStr(IOElement &fileElement) {

        return toStr(fileElement, MetaData());

    }

    IOElement fromStr(const std::string &str, const MetaData &metaData) {

        json jroot = json::parse(str);

        IOElement fe = IOElement::create();

        if(toFile(jroot, metaData, fe) != S_OK) 
            tf_exp(std::runtime_error("Could not translate data"));

        jroot.clear();

        return fe;
    }

    IOElement fromStr(const std::string &str) {
        return fromStr(str, MetaData());
    }


    // FIOModule

    void FIOModule::registerIOModule() {
        FIO::registerModule(this->moduleName(), this);
    }

    void FIOModule::load() {
        FIO::loadModule(this->moduleName());
    }


    // FIO


    IOElement FIO::fromFile(const std::string &loadFilePath) { 

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

        releaseIORootElement();

        MetaData metaData, metaDataFile;
        IOElement feMetaData = IOElement::create();
        if(::TissueForge::io::toFile(jroot[FIO::KEY_METADATA], metaData, feMetaData) != S_OK) 
            tf_exp(std::runtime_error("Could not unpack metadata"));
        if(::TissueForge::io::fromFile(feMetaData, metaData, &metaDataFile) != S_OK) 
            tf_exp(std::runtime_error("Could not load metadata"));

        TF_Log(LOG_INFORMATION) << "Got file metadata: " << metaDataFile.versionMajor << "." << metaDataFile.versionMinor << "." << metaDataFile.versionPatch;

        if(::TissueForge::io::toFile(jroot, metaDataFile, _currentRootElement)) 
            tf_exp(std::runtime_error("Could not load simulation data"));
        
        jroot.clear();
        TF_Log(LOG_INFORMATION) << "Generated i/o from source: " << loadFilePath;
        _hasCurrentRootElement = true;
        
        return _currentRootElement.clone();
    }

    HRESULT FIO::fromFile(const std::string &loadFilePath, IOElement &el) {
        try {
            el = FIO::fromFile(loadFilePath);
        }
        catch(const std::exception &e) {
            return tf_exp(e);
        }
        return S_OK;
    }

    IOElement FIO::generateIORootElement() {

        FIO::releaseIORootElement();

        IOElement tfData = IOElement::create();
        tfData.get()->type = FIO::KEY_ROOT;

        // Add metadata

        MetaData metaData;
        IOElement feMetaData = IOElement::create();
        if(::TissueForge::io::toFile(metaData, metaData, feMetaData) != S_OK) 
            tf_exp(std::runtime_error("Could not store metadata"));
        tfData.addChild(feMetaData, FIO::KEY_METADATA);

        // Add simulator
        
        auto simulator = Simulator::get();
        IOElement feSimulator = IOElement::create();
        if(::TissueForge::io::toFile(*simulator, metaData, feSimulator) != S_OK) 
            tf_exp(std::runtime_error("Could not store simulator"));
        tfData.addChild(feSimulator, FIO::KEY_SIMULATOR);

        // Add universe
        
        auto universe = Universe::get();
        IOElement feUniverse = IOElement::create();
        if(::TissueForge::io::toFile(*universe, metaData, feUniverse) != S_OK) 
            tf_exp(std::runtime_error("Could not store universe"));
        tfData.addChild(feUniverse, FIO::KEY_UNIVERSE);

        // Add modules

        if(FIO::modules == NULL) 
            FIO::modules = new std::unordered_map<std::string, FIOModule*>();

        if(FIO::modules->size() > 0) {
            IOElement feModules = IOElement::create();
            for(auto &itr : *FIO::modules) {
                IOElement feModule = IOElement::create();
                if(itr.second->toFile(metaData, feModule) != S_OK) 
                    tf_exp(std::runtime_error("Could not store module: " + itr.first));
                feModules.addChild(feModule, itr.first);
            }

            tfData.addChild(feModules, FIO::KEY_MODULES);
        }

        _currentRootElement = tfData.clone();
        _hasCurrentRootElement = true;

        return tfData;
    }

    HRESULT FIO::releaseIORootElement() { 

        if(!_hasCurrentRootElement) 
            return S_OK;

        _currentRootElement.reset();
        _hasCurrentRootElement = false;

        return S_OK;
    }

    HRESULT FIO::getCurrentIORootElement(IOElement *el) {
        if(!_hasCurrentRootElement) {
            return tf_error(E_FAIL, "No current import");
        }

        *el = _currentRootElement.clone();
        return S_OK;
    }

    HRESULT FIO::toFile(const std::string &saveFilePath) { 

        MetaData metaData;
        IOElement tfData = generateIORootElement();

        // Create root node

        json jroot;

        if(::TissueForge::io::fromFile(tfData, metaData, &jroot) != S_OK) 
            tf_exp(std::runtime_error("Could not translate final data"));

        // Write

        std::ofstream saveFile(saveFilePath);
        
        saveFile << jroot.dump(4);

        saveFile.close();

        jroot.clear();

        return releaseIORootElement();
    }

    std::string FIO::toString() {

        MetaData metaData;
        IOElement tfData = generateIORootElement();

        // Create root node

        json jroot;

        if(::TissueForge::io::fromFile(tfData, metaData, &jroot) != S_OK) 
            tf_exp(std::runtime_error("Could not translate final data"));

        // Write

        std::string result = jroot.dump(4);

        if(releaseIORootElement() != S_OK) 
            tf_exp(std::runtime_error("Could not close root element"));

        jroot.clear();

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
        
        if(!_hasCurrentRootElement) 
            tf_exp(std::runtime_error("No import state"));
        
        // Get file metadata
        
        IOElement feMetaData = _currentRootElement.get()->children[FIO::KEY_METADATA];
        MetaData metaData, metaDataFile;
        if(::TissueForge::io::fromFile(feMetaData, metaData, &metaDataFile) != S_OK) 
            tf_exp(std::runtime_error("Could not load metadata"));

        // Get modules element
        
        auto mItr = _currentRootElement.get()->children.find(FIO::KEY_MODULES);
        if(mItr == _currentRootElement.get()->children.end()) 
            tf_exp(std::runtime_error("No loaded modules"));
        auto feModules = mItr->second;

        // Get module element
        
        mItr = feModules.get()->children.find(moduleName);
        if(mItr == feModules.get()->children.end()) 
            tf_exp(std::runtime_error("Module data not available: " + moduleName));

        // Issue module import
        
        if((*FIO::modules)[moduleName]->fromFile(metaDataFile, mItr->second) != S_OK) 
            tf_exp(std::runtime_error("Module import failed: " + moduleName));
    }

    bool FIO::hasImport() {
        return _hasCurrentRootElement;
    }

};
