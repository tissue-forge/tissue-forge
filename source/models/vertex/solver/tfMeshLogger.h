/*******************************************************************************
 * This file is part of Tissue Forge.
 * Copyright (c) 2022, 2023 T.J. Sego and Tien Comlekoglu
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

/**
 * @file tfMeshLogger.h
 * 
 */

#ifndef _MODELS_VERTEX_SOLVER_TFMESHLOGGER_H_
#define _MODELS_VERTEX_SOLVER_TFMESHLOGGER_H_

#include <tf_port.h>
#include <tfLogger.h>
#include <types/tf_cast.h>

#include "tfMeshObj.h"

#include <string>
#include <vector>


namespace TissueForge::models::vertex { 


    /**
     * @brief Types of log events
     * 
     */
    enum MeshLogEventType {
        None = 0,
        Create,
        Destroy, 
        Operation
    };

    /**
     * @brief An event for the logger
     * 
     */
    struct CAPI_EXPORT MeshLogEvent {
        MeshLogEventType type;
        std::vector<int> objIDs;
        std::vector<MeshObjTypeLabel> objTypes;
        std::string name;
    };

    /**
     * @brief The Tissue Forge vertex model solver logger. 
     * 
     */
    struct CAPI_EXPORT MeshLogger {

        /**
         * @brief Clear the log
         */
        static HRESULT clear();

        /**
         * @brief Add a log event to the log
         * 
         * @param event a log event
         */
        static HRESULT log(const MeshLogEvent &event);

        /**
         * @brief Get the list of log events
         */
        static std::vector<MeshLogEvent> events();

        /**
         * @brief Test whether the logger is fowarding log events to the main Tissue Forge logger
         */
        static bool getForwardLogging();

        /**
         * @brief Set whether to foward log events to the main Tissue Forge logger
         * 
         * @param _forward flag indicating whether to foward log events
         */
        static HRESULT setForwardLogging(const bool &_forward);

        /**
         * @brief Get the current log level
         */
        static LogLevel getLogLevel();

        /**
         * @brief Set the current log level
         * 
         * @param _level the current log level
         */
        static HRESULT setLogLevel(const LogLevel &_level);

    };

}


inline std::ostream& operator<<(std::ostream& os, const TissueForge::models::vertex::MeshLogEvent &logEvent)
{
    os << "{";

    switch (logEvent.type)
    {
    case TissueForge::models::vertex::MeshLogEventType::Create:
        os << "Create";
        break;
    
    case TissueForge::models::vertex::MeshLogEventType::Destroy:
        os << "Destroy";
        break;

    case TissueForge::models::vertex::MeshLogEventType::Operation:
        os << "Operation";
        break;

    default:
        os << "None";
        break;
    }

    os << " (" << logEvent.name << "): (";
    if(logEvent.objIDs.size() > 0) {
        os << logEvent.objIDs[0];
        for(int i = 1; i < logEvent.objIDs.size(); i++) os << ", " << logEvent.objIDs[i];
    }
    os << "), (";
    if(logEvent.objTypes.size() > 0) {
        os << logEvent.objTypes[0];
        for(int i = 1; i < logEvent.objTypes.size(); i++) os << ", " << logEvent.objTypes[i];
    }
    os << ")}";
    return os;
}

template<> 
inline std::string TissueForge::cast(const TissueForge::models::vertex::MeshLogEvent &logEvent) {
    std::stringstream ss;
    ss << logEvent;
    return ss.str();
}

#endif // _MODELS_VERTEX_SOLVER_TFMESHLOGGER_H_