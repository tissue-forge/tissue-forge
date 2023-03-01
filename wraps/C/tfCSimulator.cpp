/*******************************************************************************
 * This file is part of Tissue Forge.
 * Copyright (c) 2022, 2023 T.J. Sego
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

#include "tfCSimulator.h"

#include "TissueForge_c_private.h"
#include "tfCUniverse.h"

#include <tfSimulator.h>


using namespace TissueForge;


namespace TissueForge {
    

    Simulator::Config *cast(struct tfSimulatorConfigHandle *handle) {
        return castC<Simulator::Config, tfSimulatorConfigHandle>(handle);
    }

    Simulator *cast(struct tfSimulatorHandle *handle) {
        return castC<Simulator, tfSimulatorHandle>(handle);
    }

}

#define TFC_SIMCONFIG_GET(handle) \
    Simulator::Config *conf = TissueForge::castC<Simulator::Config, tfSimulatorConfigHandle>(handle); \
    TFC_PTRCHECK(conf);

#define TFC_SIM_GET(handle) \
    Simulator *sim = TissueForge::castC<Simulator, tfSimulatorHandle>(handle); \
    TFC_PTRCHECK(sim);

/////////////////////////////////
// Simulator::EngineIntegrator //
/////////////////////////////////

HRESULT tfSimulatorEngineIntegrator_init(struct tfSimulatorEngineIntegratorHandle *handle) {
    TFC_PTRCHECK(handle);
    handle->FORWARD_EULER = (int)Simulator::EngineIntegrator::FORWARD_EULER;
    handle->RUNGE_KUTTA_4 = (int)Simulator::EngineIntegrator::RUNGE_KUTTA_4;
    return S_OK;
}

///////////////////////
// Simulator::Config //
///////////////////////

HRESULT tfSimulatorConfig_init(struct tfSimulatorConfigHandle *handle) {
    TFC_PTRCHECK(handle);
    handle->tfObj = new Simulator::Config();
    return S_OK;
}

HRESULT tfSimulatorConfig_getTitle(struct tfSimulatorConfigHandle *handle, char **title, unsigned int *numChars) {
    TFC_SIMCONFIG_GET(handle)
    return TissueForge::capi::str2Char(conf->title(), title, numChars);
}

HRESULT tfSimulatorConfig_setTitle(struct tfSimulatorConfigHandle *handle, char *title) {
    TFC_SIMCONFIG_GET(handle)
    conf->setTitle(title);
    return S_OK;
}

HRESULT tfSimulatorConfig_getWindowSize(struct tfSimulatorConfigHandle *handle, unsigned int *x, unsigned int *y) {
    TFC_SIMCONFIG_GET(handle)
    TFC_PTRCHECK(x);
    TFC_PTRCHECK(y);
    auto ws = conf->windowSize();
    *x = ws.x();
    *y = ws.y();
    return S_OK;
}

HRESULT tfSimulatorConfig_setWindowSize(struct tfSimulatorConfigHandle *handle, unsigned int x, unsigned int y) {
    TFC_SIMCONFIG_GET(handle)
    conf->setWindowSize(iVector2(x, y));
    return S_OK;
}

HRESULT tfSimulatorConfig_getSeed(struct tfSimulatorConfigHandle *handle, unsigned int *seed) {
    TFC_SIMCONFIG_GET(handle)
    TFC_PTRCHECK(seed);
    unsigned int *_seed = conf->seed();
    TFC_PTRCHECK(_seed);
    *seed = *_seed;
    return S_OK;
}

HRESULT tfSimulatorConfig_setSeed(struct tfSimulatorConfigHandle *handle, unsigned int seed) {
    TFC_SIMCONFIG_GET(handle)
    conf->setSeed(seed);
    return S_OK;
}

HRESULT tfSimulatorConfig_getWindowless(struct tfSimulatorConfigHandle *handle, bool *windowless) {
    TFC_SIMCONFIG_GET(handle);
    TFC_PTRCHECK(windowless);
    *windowless = conf->windowless();
    return S_OK;
}

HRESULT tfSimulatorConfig_setWindowless(struct tfSimulatorConfigHandle *handle, bool windowless) {
    TFC_SIMCONFIG_GET(handle);
    conf->setWindowless(windowless);
    return S_OK;
}

HRESULT tfSimulatorConfig_getImportDataFilePath(struct tfSimulatorConfigHandle *handle, char **filePath, unsigned int *numChars) {
    TFC_SIMCONFIG_GET(handle)
    std::string *fp = conf->importDataFilePath;
    if(!fp) {
        numChars = 0;
        return S_OK;
    }
    else return TissueForge::capi::str2Char(*fp, filePath, numChars);
}

HRESULT tfSimulatorConfig_getClipPlanes(struct tfSimulatorConfigHandle *handle, float **clipPlanes, unsigned int *numClipPlanes) {
    TFC_SIMCONFIG_GET(handle)
    TFC_PTRCHECK(clipPlanes);
    TFC_PTRCHECK(numClipPlanes);
    *numClipPlanes = conf->clipPlanes.size();
    if(*numClipPlanes > 0) {
        float *cps = (float*)malloc(*numClipPlanes * 4 * sizeof(float));
        if(!cps) 
            return E_OUTOFMEMORY;
        for(unsigned int i = 0; i < *numClipPlanes; i++) {
            auto _cp = conf->clipPlanes[i];
            for(unsigned int j = 0; j < 4; j++) 
                cps[4 * i + j] = _cp[j];
        }
        *clipPlanes = cps;
    }
    return S_OK;
}

HRESULT tfSimulatorConfig_setClipPlanes(struct tfSimulatorConfigHandle *handle, float *clipPlanes, unsigned int numClipPlanes) {
    TFC_SIMCONFIG_GET(handle);
    TFC_PTRCHECK(clipPlanes);
    conf->clipPlanes.clear();
    for(unsigned int i = 0; i < numClipPlanes; i++) {
        float *b = &clipPlanes[4 * i];
        conf->clipPlanes.push_back(fVector4(b[0], b[1], b[2], b[3]));
    }
    return S_OK;
}

HRESULT tfSimulatorConfig_getUniverseConfig(struct tfSimulatorConfigHandle *handle, struct tfUniverseConfigHandle *confHandle) {
    TFC_SIMCONFIG_GET(handle)
    TFC_PTRCHECK(confHandle);
    confHandle->tfObj = (void*)&conf->universeConfig;
    return S_OK;
}

HRESULT tfSimulatorConfig_destroy(struct tfSimulatorConfigHandle *handle) {
    return TissueForge::capi::destroyHandle<Simulator::Config, tfSimulatorConfigHandle>(handle) ? S_OK : E_FAIL;
}


///////////////
// Simulator //
///////////////


HRESULT tfSimulator_init(char **argv, unsigned int nargs) {

    return Simulator_init(TissueForge::capi::charA2StrV((const char**)argv, nargs));
}

HRESULT tfSimulator_initC(struct tfSimulatorConfigHandle *handle, char **appArgv, unsigned int nargs) {

    TFC_SIMCONFIG_GET(handle)

    return Simulator_init(*conf, TissueForge::capi::charA2StrV((const char**)appArgv, nargs));
}

HRESULT tfSimulator_get(struct tfSimulatorHandle *handle) {
    Simulator *sim = Simulator::get();
    if(!sim) 
        return E_FAIL;
    handle->tfObj = sim;
    return S_OK;
}

HRESULT tfSimulator_makeCurrent(struct tfSimulatorHandle *handle) {
    TFC_SIM_GET(handle)
    return sim->makeCurrent();
}

HRESULT tfSimulator_run(tfFloatP_t et) {
    Simulator *sim = Simulator::get();
    TFC_PTRCHECK(sim);
    return sim->run(et);
}

HRESULT tfSimulator_show() {
    Simulator *sim = Simulator::get();
    TFC_PTRCHECK(sim);
    return sim->show();
}

HRESULT tfSimulator_close() {
    Simulator *sim = Simulator::get();
    TFC_PTRCHECK(sim);
    return sim->close();
}

HRESULT tfSimulator_destroy() {
    Simulator *sim = Simulator::get();
    TFC_PTRCHECK(sim);
    return sim->destroy();
}

HRESULT tfSimulator_redraw() {
    Simulator *sim = Simulator::get();
    TFC_PTRCHECK(sim);
    return sim->redraw();
}

HRESULT tfSimulator_getNumThreads(unsigned int *numThreads) {
    Simulator *sim = Simulator::get();
    TFC_PTRCHECK(sim);
    TFC_PTRCHECK(numThreads);
    *numThreads = sim->getNumThreads();
    return S_OK;
}

bool tfIsTerminalInteractiveShell() {
    return isTerminalInteractiveShell();
}

HRESULT tfSetIsTerminalInteractiveShell(bool _interactive) {
    return setIsTerminalInteractiveShell(_interactive);
}