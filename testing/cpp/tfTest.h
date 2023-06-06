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

#ifndef _TESTING_CPP_TFTEST_H_
#define _TESTING_CPP_TFTEST_H_

#include <tf_port.h>

#include <TissueForge.h>
#include <tfLogger.h>

#define TF_TEST_REPORTERR() { std::cerr << "Error: " << __LINE__ << ", " << TF_FUNCTION << ", " << __FILE__ << std::endl; }
#define TF_TEST_CHECK(code) { if((code) != S_OK) { TF_TEST_REPORTERR(); return E_FAIL; } }


HRESULT tfTest_init(TissueForge::Simulator::Config &conf) {
    #ifdef TFTEST_LOG
    TissueForge::Logger::enableConsoleLogging(TissueForge::LogLevel::LOG_DEBUG);
    #endif
    return TissueForge::init(conf);
}

#endif // _TESTING_CPP_TFTEST_H_