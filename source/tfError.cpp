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
#include <memory>
#include <tfLogger.h>
#include <tf_errs.h>


using namespace TissueForge;

static std::vector<std::shared_ptr<Error> > error_registry;

static HRESULT engineError(const unsigned int &id, Error &err) {
	char *msg, *func, *fname;
	if(errs_get((int)id, &msg, &err.lineno, &func, &fname) != errs_err_ok) 
		return E_FAIL;

	err.msg = msg;
	err.func = func;
	err.fname = fname;
	return S_OK;
}

static HRESULT engineErrors(std::vector<Error> &result) {
	result = std::vector<Error>(errs_num());
	for(unsigned int i = 0; i < result.size(); i++) 
		if(engineError(i, result[i]) != S_OK) 
			return E_FAIL;
	return S_OK;
}

HRESULT TissueForge::errSet(HRESULT code, const char* msg, int line,
		const char* file, const char* func) {

	auto err = std::make_shared<Error>();
	err->err = code;
	err->fname = file;
	err->func = func;
	err->msg = msg;
    
    TF_Log(LOG_ERROR) << *err;
	error_registry.push_back(err);

	return code;
}

HRESULT TissueForge::expSet(const std::exception& e, const char* msg, int line, const char* file, const char* func) {
    return errSet(E_FAIL, e.what(), line, file, func);
}

bool TissueForge::errOccurred() {
    return !error_registry.empty() && errs_num() == 0;
}

void TissueForge::errClear() {
    error_registry.clear();
	errs_clear();
}

std::string TissueForge::errStr(const Error &err) {
	std::stringstream ss;
	ss << err;
	return ss.str();
}

std::vector<Error> TissueForge::errGetAll() {
	std::vector<Error> result;
	result.reserve(error_registry.size() + errs_num());
	for(auto &e : error_registry) 
		result.push_back(*e);

	std::vector<Error> eng_result;
	engineErrors(eng_result);
	for(auto &e : eng_result) 
		result.push_back(e);
	return result;
}

Error TissueForge::errGetFirst() {
	Error result;
	if(!errOccurred()) {
		tf_error(E_FAIL, "Requested errors, but no errors to report");
		result = *error_registry.front();
		error_registry.clear();
	} 
	else {
		if(error_registry.empty()) engineError(0, result);
		else result = *error_registry.front();
	}
	
	return result;
}

void TissueForge::errClearFirst() {
	if(error_registry.empty()) 
		errs_clear();
	else 
		error_registry.erase(error_registry.begin());
}

Error TissueForge::errPopFirst() {
	Error result = errGetFirst();
	errClearFirst();
	return result;
}
