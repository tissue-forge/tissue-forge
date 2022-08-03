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

#include <cstdio>

#include "tfError.h"

#include <iostream>
#include <sstream>
#include <tfLogger.h>


using namespace TissueForge;


static Error _Error;
static Error *_ErrorPtr = NULL;

HRESULT TissueForge::errSet(HRESULT code, const char* msg, int line,
		const char* file, const char* func) {

	_Error.err = code;
	_Error.fname = file;
	_Error.func = func;
	_Error.msg = msg;
    
    std::string logstr = "Code: ";
    logstr += std::to_string(code);
    logstr += ", Msg: ";
    logstr += msg;
    logstr += ", File: " + std::string(file);
    logstr += ", Line: " + std::to_string(line);
    logstr += ", Function: " + std::string(func);
    Logger::log(LOG_ERROR, logstr);

	_ErrorPtr = &_Error;
	return code;
}

HRESULT TissueForge::expSet(const std::exception& e, const char* msg, int line, const char* file, const char* func) {
    std::cerr << "error: " << e.what() << ", " << msg << ", " << line << ", " << func << std::endl;
    
    _Error.err = E_FAIL;
    _Error.fname = file;
    _Error.func = func;
    _Error.msg = msg;
    
    _ErrorPtr = &_Error;
    return E_FAIL;
}

Error* TissueForge::errOccurred() {
    return _ErrorPtr;
}

void TissueForge::errClear() {
    _ErrorPtr = NULL;
}
