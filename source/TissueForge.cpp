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

#include "TissueForge_private.h"
#include <TissueForge.h>
#include <tfSimulator.h>
#include <tfError.h>
#include <tfLogger.h>

#include <Magnum/GL/Context.h>
#include <string>


namespace TissueForge {


    std::string version_str() { return std::string(TF_VERSION); }

    std::string systemNameStr() { return std::string(TF_SYSTEM_NAME); }

    std::string systemVersionStr() { return std::string(TF_SYSTEM_VERSION); }

    std::string compilerIdStr() { return std::string(TF_COMPILER_ID); }

    std::string compilerVersionStr() { return std::string(TF_COMPILER_VERSION); }

    std::string buildDate() { return std::string(tfBuildDate()); }

    std::string buildTime() { return std::string(tfBuildTime()); }

    bool hasCuda() { return tfHasCuda(); }

    HRESULT initialize(int args) {

        TF_Log(LOG_TRACE);
        
        // GL symbols are globals in each shared library address space,
        // if the app already initialized gl, we need to get the symbols here
        if(Magnum::GL::Context::hasCurrent() && !glCreateProgram) {
            flextGLInit(Magnum::GL::Context::current());
        }

        return modules_init();
    }

    HRESULT close() { return Simulator::close(); }

    HRESULT show() { return Simulator::show(); }

    HRESULT init(const std::vector<std::string> &argv) { return Simulator_init(argv); }

    HRESULT init(Simulator::Config &conf, const std::vector<std::string> &appArgv) { return Simulator_init(conf, appArgv); }

    HRESULT step(const FloatP_t &until, const FloatP_t &dt) { return Universe::step(until, dt); }

    HRESULT stop() { return Universe::stop(); }
    
    HRESULT start() { return Universe::start(); }

    HRESULT run(FloatP_t et) { return Simulator::run(et); }

};
