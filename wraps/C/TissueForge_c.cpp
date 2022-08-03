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

#include "TissueForge_c.h"

#include "TissueForge_c_private.h"

#include <tf_config.h>


using namespace TissueForge;


//////////////////////
// Module functions //
//////////////////////


HRESULT tfVersionStr(char **str, unsigned int *numChars) {
    return capi::str2Char(TF_VERSION, str, numChars);
}

HRESULT tfSystemNameStr(char **str, unsigned int *numChars) {
    return capi::str2Char(TF_SYSTEM_NAME, str, numChars);
}

HRESULT tfSystemVersionStr(char **str, unsigned int *numChars) {
    return capi::str2Char(TF_SYSTEM_VERSION, str, numChars);
}

HRESULT tfCompilerIDStr(char **str, unsigned int *numChars) {
    return capi::str2Char(TF_COMPILER_ID, str, numChars);
}

HRESULT tfCompilerVersionStr(char **str, unsigned int *numChars) {
    return capi::str2Char(TF_COMPILER_VERSION, str, numChars);
}

HRESULT tfBuildDateStr(char **str, unsigned int *numChars) {
    return capi::str2Char(tfBuildDate(), str, numChars);
}

HRESULT tfBuildTimeStr(char **str, unsigned int *numChars) {
    return capi::str2Char(tfBuildTime(), str, numChars);
}

HRESULT tfVersionMajorStr(char **str, unsigned int *numChars) {
    return capi::str2Char(std::to_string(TF_VERSION_MAJOR), str, numChars);
}

HRESULT tfVersionMinorStr(char **str, unsigned int *numChars) {
    return capi::str2Char(std::to_string(TF_VERSION_MINOR), str, numChars);
}

HRESULT tfVersionPatchStr(char **str, unsigned int *numChars) {
    return capi::str2Char(std::to_string(TF_VERSION_PATCH), str, numChars);
}

HRESULT tfClose() {
    return tfSimulator_close();
}

HRESULT tfShow() {
    return tfSimulator_show();
}

HRESULT tfInit(char **argv, unsigned int nargs) {
    return tfSimulator_init(argv, nargs);
}

HRESULT tfInitC(struct tfSimulatorConfigHandle *conf, char **appArgv, unsigned int nargs) {
    return tfSimulator_initC(conf, appArgv, nargs);
}

HRESULT tfStep(tfFloatP_t until, tfFloatP_t dt) {
    return tfUniverse_step(until, dt);
}

HRESULT tfStop() {
    return tfUniverse_stop();
}

HRESULT tfStart() {
    return tfUniverse_start();
}

HRESULT tfRun(tfFloatP_t et) {
    return tfSimulator_run(et);
}
