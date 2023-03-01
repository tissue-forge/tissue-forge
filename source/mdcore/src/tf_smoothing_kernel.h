/*******************************************************************************
 * This file is part of mdcore.
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

#ifndef _MDCORE_SOURCE_TF_SMOOTHING_KERNEL_H_
#define _MDCORE_SOURCE_TF_SMOOTHING_KERNEL_H_

#include <mdcore_config.h>
#include <cmath>

#include <tf_fptype.h>


namespace TissueForge { 


    #if defined(__x86_64__) || defined(_M_X64)
    // faster than  1.0f/std::sqrt, but with little accuracy.
    TF_ALWAYS_INLINE float qsqrt(const float f)
    {
        __m128 temp = _mm_set_ss(f);
        temp = _mm_rsqrt_ss(temp);
        return 1.0 / _mm_cvtss_f32(temp);
    }
    #endif

    #if defined(__ARM_NEON)
    TF_ALWAYS_INLINE float qsqrt(const float f)
    {
        return 1.0f/std::sqrt(f);
    }
    #endif

    TF_ALWAYS_INLINE FPTYPE w_cubic_spline(FPTYPE r2, FPTYPE h) {
        FPTYPE r = (FPTYPE)qsqrt(r2);
        FPTYPE x = r/h;
        FPTYPE y;
        
        if(x < 1.f) {
            FPTYPE x2 = x * x;
            y = 1.f - (3.f / 2.f) * x2 + (3.f / 4.f) * x2 * x;
        }
        else if(x >= 1.f && x < 2.f) {
            FPTYPE arg = 2.f - x;
            y = (1.f / 4.f) * arg * arg * arg;
        }
        else {
            y = 0.f;
        }
        
        return y / (M_PI * h * h * h);
    }

    TF_ALWAYS_INLINE FPTYPE grad_w_cubic_spline(FPTYPE r2, FPTYPE h) {
        FPTYPE r = (FPTYPE)qsqrt(r2);
        FPTYPE x = r/h;
        FPTYPE y;
        
        if(x < 1.f) {
            y = (9.f / 4.f) * x * x  - (3.f) * x;
        }
        else if(x >= 1.f && x < 2.f) {
            FPTYPE arg = 2.f - x;
            y = -(3.f / 4.f) * arg * arg;
        }
        else {
            y = 0.f;
        }
        
        return y / (M_PI * h * h * h * h);
    }

    TF_ALWAYS_INLINE FPTYPE W(FPTYPE r2, FPTYPE h) { return w_cubic_spline(r2, h); };

    TF_ALWAYS_INLINE FPTYPE grad_W(FPTYPE r2, FPTYPE h) { return grad_w_cubic_spline(r2, h); };

};

#endif // _MDCORE_SOURCE_TF_SMOOTHING_KERNEL_H_