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

#ifndef _SOURCE_IO_TFFIO_H_
#define _SOURCE_IO_TFFIO_H_


#include "tf_io.h"


namespace TissueForge::io {


    /**
     * @brief Generate a JSON string representation of an intermediate I/O object. 
     * 
     * @param fileElement object to convert
     * @param metaData meta data of target installation
     * @return std::string 
     */
    std::string toStr(IOElement &fileElement, const MetaData &metaData);

    /**
     * @brief Generate a JSON string representation of an intermediate I/O object. 
     * 
     * Current installation is target installation.
     * 
     * @param fileElement object to convert
     * @return std::string 
     */
    std::string toStr(IOElement &fileElement);

    /**
     * @brief Generate an intermediate I/O object from a JSON string. 
     * 
     * @param str JSON string
     * @param metaData meta data of target installation
     * @return IOElement 
     */
    IOElement fromStr(const std::string &str, const MetaData &metaData);

    /**
     * @brief Generate an intermediate I/O object from a JSON string. 
     * 
     * Installation during string export is target installation.
     * 
     * @param str JSON string
     * @return IOElement 
     */
    IOElement fromStr(const std::string &str);

    /**
     * @brief Generate a JSON string representation of an object. 
     * 
     * @tparam T type of source object
     * @param dataElement source object
     * @param metaData meta data of target installation
     * @return std::string 
     */
    template <typename T>
    std::string toString(const T &dataElement, const MetaData &metaData) {
        IOElement fe = IOElement::create();
        if(toFile<T>(dataElement, metaData, fe) != S_OK) 
            return "";
        return toStr(fe, metaData);
    }

    /**
     * @brief Generate a JSON string representation of an object. 
     * 
     * Current installation is target installation. 
     * 
     * @tparam T type of source object
     * @param dataElement source object
     * @return std::string 
     */
    template <typename T>
    std::string toString(const T &dataElement) {
        return toString(dataElement, MetaData());
    }

    /**
     * @brief Generate an object from a JSON string. 
     * 
     * @tparam T type of object
     * @param str JSON string
     * @param metaData meta data of target installation
     * @return T 
     */
    template <typename T>
    T fromString(const std::string &str, const MetaData &metaData) { 
        IOElement fe = fromStr(str, metaData);
        T de;
        fromFile<T>(fe, metaData, &de);
        return de;
    }

    /**
     * @brief Generate an object from a JSON string. 
     * 
     * Current installation is target installation. 
     * 
     * @tparam T type of object
     * @param str JSON string
     * @return T 
     */
    template <typename T>
    T fromString(const std::string &str) {
        return fromString<T>(str, MetaData());
    }

    /**
     * @brief Tissue Forge data import summary. 
     * 
     * Not every datum of an imported Tissue Forge simulation state is conserved 
     * (e.g., particle id). This class provides the information necessary 
     * to exactly translate the imported data of a previously exported simulation. 
     * 
     */
    struct CAPI_EXPORT FIOImportSummary {

        /** Map of imported particle ids to loaded particle ids */
        std::unordered_map<unsigned int, unsigned int> particleIdMap;

        /** Map of imported particle type ids to loaded particle ids */
        std::unordered_map<unsigned int, unsigned int> particleTypeIdMap;

    };


    /**
     * @brief Interface for Tissue Forge peripheral module I/O (e.g., models)
     * 
     */
    struct FIOModule {

        /**
         * @brief Name of module. Used as a storage key. 
         * 
         * @return std::string 
         */
        virtual std::string moduleName() = 0;

        /**
         * @brief Export module data. 
         * 
         * @param metaData metadata of current installation
         * @param fileElement container to store serialized data
         * @return HRESULT 
         */
        virtual HRESULT toFile(const MetaData &metaData, IOElement &fileElement) = 0;

        /**
         * @brief Import module data. 
         * 
         * @param metaData metadata of import file. 
         * @param fileElement container of stored serialized data
         * @return HRESULT 
         */
        virtual HRESULT fromFile(const MetaData &metaData, const IOElement &fileElement) = 0;

        /**
         * @brief Register this module for I/O events
         * 
         */
        void registerIOModule();

        /**
         * @brief User-facing function to load module data from main import. 
         * 
         * Must only be called after main import. 
         * 
         */
        void load();

    };


    /**
     * @brief Tissue Forge data import/export interface. 
     * 
     * This interface provides methods for serializing/deserializing 
     * the state of a Tissue Forge simulation. 
     * 
     */
    struct CAPI_EXPORT FIO {

        /** Key for basic element type storage */
        static const std::string KEY_TYPE;

        /** Key for basic element value storage */
        static const std::string KEY_VALUE;

        /** Key for simulation metadata storage */
        static const std::string KEY_METADATA;

        /** Key for simulation simulator storage */
        static const std::string KEY_SIMULATOR;

        /** Key for simulation universe storage */
        static const std::string KEY_UNIVERSE;

        /** Key for module i/o storage */
        static const std::string KEY_MODULES;

        /** Import summary of most recent import */
        inline static FIOImportSummary *importSummary = NULL;

        /**
        * @brief Generate root element from current simulation state
        * 
        * @return IOElement 
        */
        static IOElement generateIORootElement();

        /**
        * @brief Release current root element
        * 
        * @return HRESULT 
        */
        static HRESULT releaseIORootElement();

        /**
        * @brief Get the current root element, if any
        * 
        * @param el root element
        * @return HRESULT 
        */
        static HRESULT getCurrentIORootElement(IOElement *el);

        /**
        * @brief Load a simulation from file
        * 
        * @param loadFilePath absolute path to file
        * @return IOElement 
        */
        static IOElement fromFile(const std::string &loadFilePath);

        /**
        * @brief Load a simulation from file
        * 
        * @param loadFilePath absolute path to file
        * @param el resulting file element
        * @return HRESULT 
        */
        static HRESULT fromFile(const std::string &loadFilePath, IOElement &el);

        /**
        * @brief Save a simulation to file
        * 
        * @param saveFilePath absolute path to file
        * @return HRESULT 
        */
        static HRESULT toFile(const std::string &saveFilePath);

        /**
        * @brief Return a simulation state as a JSON string
        * 
        * @return std::string 
        */
        static std::string toString();

        /**
        * @brief Register a module for I/O events
        * 
        * @param moduleName name of module
        * @param module borrowed pointer to module
        */
        static void registerModule(const std::string moduleName, FIOModule *module);

        /**
        * @brief Load a module from imported data. 
        * 
        * Can only be called after main initial import. 
        * 
        * @param moduleName 
        */
        static void loadModule(const std::string moduleName);

        /**
        * @brief Test whether imported data is available. 
        * 
        * @return true 
        * @return false 
        */
        static bool hasImport();

    private:
        
        inline static std::unordered_map<std::string, FIOModule*> *modules = NULL;

    };

};

#endif // _SOURCE_IO_TFFIO_H_