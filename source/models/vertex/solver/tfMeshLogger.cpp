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

#include "tfMeshLogger.h"


using namespace TissueForge::models::vertex;


static std::vector<MeshLogEvent> logEvents;
static bool _isLogging = false;
static TissueForge::LogLevel _logLvl;


HRESULT MeshLogger::clear() {
    logEvents.clear();
    return S_OK;
}

HRESULT MeshLogger::log(const MeshLogEvent &event) {
    logEvents.push_back(event);
    if(_isLogging) 
        Logger::log(_logLvl, TissueForge::cast<MeshLogEvent, std::string>(event));
    return S_OK;
}

std::vector<MeshLogEvent> MeshLogger::events() {
    return logEvents;
}

bool MeshLogger::getForwardLogging() {
    return _isLogging;
}

HRESULT MeshLogger::setForwardLogging(const bool &_forward) {
    _isLogging = _forward;
    return S_OK;
}

TissueForge::LogLevel MeshLogger::getLogLevel() {
    return _logLvl;
}

HRESULT MeshLogger::setLogLevel(const LogLevel &_level) {
    _logLvl = _level;
    return S_OK;
}
