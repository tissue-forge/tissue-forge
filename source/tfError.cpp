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
#include <unordered_map>


using namespace TissueForge;

static std::vector<std::shared_ptr<Error> > error_registry;
static std::unordered_map<unsigned int, ErrorCallback&> cb_registry;


const unsigned int TissueForge::addErrorCallback(ErrorCallback &cb) {
	unsigned int result = cb_registry.size();
	for(unsigned int i = 0; i < cb_registry.size(); i++) 
		if(cb_registry.find(i) == cb_registry.end()) {
			result = i;
			break;
		}

	cb_registry.insert({result, cb});
	return result;
}

HRESULT TissueForge::removeErrorCallback(const unsigned int &cb_id) {
	auto itr = cb_registry.find(cb_id);
	if(itr == cb_registry.end()) 
		return E_FAIL;

	cb_registry.erase(itr);
	return S_OK;
}

HRESULT TissueForge::clearErrorCallbacks() {
	cb_registry.clear();
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
	for(auto &cb_pair : cb_registry) 
		cb_pair.second(*err);

	return code;
}

HRESULT TissueForge::expSet(const std::exception& e, const char* msg, int line, const char* file, const char* func) {
    return errSet(E_FAIL, e.what(), line, file, func);
}

bool TissueForge::errOccurred() {
    return !error_registry.empty();
}

void TissueForge::errClear() {
    error_registry.clear();
}

std::string TissueForge::errStr(const Error &err) {
	std::stringstream ss;
	ss << err;
	return ss.str();
}

std::vector<Error> TissueForge::errGetAll() {
	std::vector<Error> result;
	result.reserve(error_registry.size());
	for(auto &e : error_registry) 
		result.push_back(*e);
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
		result = *error_registry.front();
	}
	
	return result;
}

void TissueForge::errClearFirst() {
	if(!error_registry.empty()) 
		error_registry.erase(error_registry.begin());
}

Error TissueForge::errPopFirst() {
	Error result = errGetFirst();
	errClearFirst();
	return result;
}
