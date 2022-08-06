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

/*
 Based on Magnum example
 
 Original authors — credit is appreciated but not required:
 
 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019 —
 Vladimír Vondruš <mosra@centrum.cz>
 2019 — Nghia Truong <nghiatruong.vn@gmail.com>
 
 This library is free software; you can redistribute it and/or
 modify it under the terms of the GNU Lesser General Public
 License as published by the Free Software Foundation; either
 version 2.1 of the License, or (at your option) any later version.
 
 This library is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 Lesser General Public License for more details.
 */

#ifndef _SOURCE_TFTASKSCHEDULER_H_
#define _SOURCE_TFTASKSCHEDULER_H_

#ifdef USE_TBB
#include <tbb/tbb.h>
#endif

#include "tfThreadPool.h"


namespace TissueForge {


    template<class IndexType, class Function> void parallel_for(IndexType endIdx, Function&& func) {
    #ifdef USE_TBB
        tbb::parallel_for(tbb::blocked_range<IndexType>(IndexType(0), endIdx),
                        [&](const tbb::blocked_range<IndexType>& r) {
            for(IndexType i = r.begin(), iEnd = r.end(); i < iEnd; ++i) {
                func(i);
            }
        });
    #else
        ThreadPool::getUniqueInstance().parallel_for(endIdx, std::forward<Function>(func));
    #endif
    }

};

#endif // _SOURCE_TFTASKSCHEDULER_H_