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

#ifndef _WRAPS_C_TISSUEFORGE_C_PRIVATE_H_
#define _WRAPS_C_TISSUEFORGE_C_PRIVATE_H_

#include "tf_port_c.h"
#include <vector>

typedef tfFloatP_t FloatP_t;

#include <types/tf_types.h>

#define TFC_PTRCHECK(varname) if(!varname) return E_FAIL;

// Convenience macros for array operations

#define TFC_VECTOR2_COPYFROM(vec, arr) arr[0] = vec.x(); arr[1] = vec.y();
#define TFC_VECTOR2_COPYTO(arr, vec)   vec.x() = arr[0]; vec.y() = arr[1];
#define TFC_VECTOR3_COPYFROM(vec, arr) arr[0] = vec.x(); arr[1] = vec.y(); arr[2] = vec.z();
#define TFC_VECTOR3_COPYTO(arr, vec)   vec.x() = arr[0]; vec.y() = arr[1]; vec.z() = arr[2];
#define TFC_VECTOR4_COPYFROM(vec, arr) arr[0] = vec.x(); arr[1] = vec.y(); arr[2] = vec.z();; arr[3] = vec.w();
#define TFC_VECTOR4_COPYTO(arr, vec)   vec.x() = arr[0]; vec.y() = arr[1]; vec.z() = arr[2]; vec.w() = arr[3];
#define TFC_MATRIX2_COPYFROM(mat, arr) arr[0] = mat[0][0]; arr[1] = mat[0][1]; \
                                       arr[2] = mat[1][0]; arr[3] = mat[1][1];
#define TFC_MATRIX2_COPYTO(arr, mat)   mat[0][0] = arr[0]; mat[0][1] = arr[1]; \
                                       mat[1][0] = arr[2]; mat[1][1] = arr[2];
#define TFC_MATRIX3_COPYFROM(mat, arr) arr[0] = mat[0][0]; arr[1] = mat[0][1]; arr[2] = mat[0][2]; \
                                       arr[3] = mat[1][0]; arr[4] = mat[1][1]; arr[5] = mat[1][2]; \
                                       arr[6] = mat[2][0]; arr[7] = mat[2][1]; arr[8] = mat[2][2];
#define TFC_MATRIX3_COPYTO(arr, mat)   mat[0][0] = arr[0]; mat[0][1] = arr[1]; mat[0][2] = arr[2]; \
                                       mat[1][0] = arr[3]; mat[1][1] = arr[4]; mat[1][2] = arr[5]; \
                                       mat[2][0] = arr[6]; mat[2][1] = arr[7]; mat[2][2] = arr[8];
#define TFC_MATRIX4_COPYFROM(mat, arr) arr[0]  = mat[0][0]; arr[1]  = mat[0][1]; arr[2]  = mat[0][2]; arr[3]  = mat[0][3]; \
                                       arr[4]  = mat[1][0]; arr[5]  = mat[1][1]; arr[6]  = mat[1][2]; arr[7]  = mat[1][3]; \
                                       arr[8]  = mat[2][0]; arr[9]  = mat[2][1]; arr[10] = mat[2][2]; arr[11] = mat[2][3]; \
                                       arr[12] = mat[3][0]; arr[13] = mat[3][1]; arr[14] = mat[3][2]; arr[15] = mat[3][3];
#define TFC_MATRIX4_COPYTO(arr, mat)   mat[0][0] = arr[0];  mat[0][1] = arr[1];  mat[0][2] = arr[2];  mat[0][3] = arr[3];  \
                                       mat[1][0] = arr[4];  mat[1][1] = arr[5];  mat[1][2] = arr[6];  mat[1][3] = arr[7];  \
                                       mat[2][0] = arr[8];  mat[2][1] = arr[9];  mat[2][2] = arr[10]; mat[2][3] = arr[11]; \
                                       mat[3][0] = arr[12]; mat[3][1] = arr[13]; mat[3][2] = arr[14]; mat[3][3] = arr[15];


// Standard template handle casts


namespace TissueForge { 


    template <typename O, typename H>
    O *castC(H *h) {
        if(!h || !h->tfObj) 
            return NULL;
        return (O*)h->tfObj;
    }

    template <typename O, typename H>
    HRESULT castC(O &obj, H *handle) {
        TFC_PTRCHECK(handle);
        handle->tfObj = (void*)&obj;
        return S_OK;
    }


    namespace capi {


        HRESULT str2Char(const std::string s, char **c, unsigned int *n);

        std::vector<std::string> charA2StrV(const char **c, const unsigned int &n);

        template <typename O, typename H>
        bool destroyHandle(H *h) {
            if(!h || !h->tfObj) {
                delete (O*)h->tfObj;
                h->tfObj = NULL;
                return true;
            }
            return false;
        }

        template<typename T> 
        HRESULT copyVecVecs2_2Arr(const std::vector<types::TVector2<T> > &vecsV, T **vecsA) {
            unsigned int n = 2;
            TFC_PTRCHECK(vecsA);
            auto nr_parts = vecsV.size();
            T *_vecsA = (T*)malloc(n * nr_parts * sizeof(T));
            if(!_vecsA) 
                return E_OUTOFMEMORY;
            for(unsigned int i = 0; i < nr_parts; i++) {
                auto _aV = vecsV[i];
                TFC_VECTOR2_COPYFROM(_aV, (_vecsA + n * i));
            }
            *vecsA = _vecsA;
            return S_OK;
        }

        template<typename T> 
        HRESULT copyVecVecs3_2Arr(const std::vector<types::TVector3<T> > &vecsV, T **vecsA) {
            unsigned int n = 3;
            TFC_PTRCHECK(vecsA);
            auto nr_parts = vecsV.size();
            T *_vecsA = (T*)malloc(n * nr_parts * sizeof(T));
            if(!_vecsA) 
                return E_OUTOFMEMORY;
            for(unsigned int i = 0; i < nr_parts; i++) {
                auto _aV = vecsV[i];
                TFC_VECTOR3_COPYFROM(_aV, (_vecsA + n * i));
            }
            *vecsA = _vecsA;
            return S_OK;
        }

        template<typename T> 
        HRESULT copyVecVecs4_2Arr(const std::vector<types::TVector4<T> > &vecsV, T **vecsA) {
            unsigned int n = 4;
            TFC_PTRCHECK(vecsA);
            auto nr_parts = vecsV.size();
            T *_vecsA = (T*)malloc(n * nr_parts * sizeof(T));
            if(!_vecsA) 
                return E_OUTOFMEMORY;
            for(unsigned int i = 0; i < nr_parts; i++) {
                auto _aV = vecsV[i];
                TFC_VECTOR4_COPYFROM(_aV, (_vecsA + n * i));
            }
            *vecsA = _vecsA;
            return S_OK;
        }

    }

}

#endif // _WRAPS_C_TISSUEFORGE_C_PRIVATE_H_