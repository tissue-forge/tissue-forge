/*******************************************************************************
 * This file is part of mdcore.
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

#include "tfSubEngine.h"

#include <tf_errs.h>
#include <tfError.h>
#include <tfEngine.h>


using namespace TissueForge;


/* the error macro. */
#define error(id)   (tf_error(E_FAIL, errs_err_msg[id]))


HRESULT SubEngine::registerEngine() {
    for(auto &se : _Engine.subengines) 
        if(strcmp(this->name, se->name) == 0) 
            return error(MDCERR_subengine);
    
    _Engine.subengines.push_back(this);
    return S_OK;
}
