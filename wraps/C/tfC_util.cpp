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

#include "tfC_util.h"

#include "TissueForge_c_private.h"

#include <tf_util.h>


using namespace TissueForge;


////////////////
// PointsType //
////////////////


HRESULT tfPointsType_init(struct tfPointsTypeHandle *handle) {
    TFC_PTRCHECK(handle);
    handle->Sphere = (unsigned int)PointsType::Sphere;
    handle->SolidSphere = (unsigned int)PointsType::SolidSphere;
    handle->Disk = (unsigned int)PointsType::Disk;
    handle->SolidCube = (unsigned int)PointsType::SolidCube;
    handle->Cube = (unsigned int)PointsType::Cube;
    handle->Ring = (unsigned int)PointsType::Ring;
    return S_OK;
}


//////////////////////
// Module functions //
//////////////////////


HRESULT tfGetSeed(unsigned int *seed) {
    TFC_PTRCHECK(seed);
    *seed = getSeed();
    return S_OK;
}

HRESULT tfSetSeed(unsigned int seed) {
    return setSeed(&seed);
}

HRESULT tfColor3Names(char ***names, unsigned int *numNames) {
    TFC_PTRCHECK(names);
    TFC_PTRCHECK(numNames);
    
    std::vector<std::string> _namesV = color3Names();
    *numNames = _namesV.size();
    if(*numNames > 0) {
        char **_names = (char**)malloc(*numNames * sizeof(char*));
        for(unsigned int i = 0; i < *numNames; i++) {
            std::string _s = _namesV[i];
            char *_c = new char[_s.size() + 1];
            std::strcpy(_c, _s.c_str());
            _names[i] = _c;
        }
        *names = _names;
    }
    return S_OK;
}

HRESULT tfPlaneEquationFPN(tfFloatP_t *point, tfFloatP_t *normal, tfFloatP_t **planeEq) {
    TFC_PTRCHECK(point);
    TFC_PTRCHECK(normal);
    TFC_PTRCHECK(planeEq);
    FVector4 _planeEq = planeEquation(FVector3::from(normal), FVector3::from(point));
    TFC_VECTOR4_COPYFROM(_planeEq, (*planeEq));
    return S_OK;
}

HRESULT tfPlaneEquationTPN(tfFloatP_t *planeEq, tfFloatP_t **point, tfFloatP_t **normal) {
    TFC_PTRCHECK(planeEq);
    TFC_PTRCHECK(point);
    TFC_PTRCHECK(normal);
    FVector3 _point, _normal;
    std::tie(_normal, _point) = planeEquation(FVector4::from(planeEq));
    TFC_VECTOR3_COPYFROM(_point, (*point));
    TFC_VECTOR3_COPYFROM(_normal, (*normal));
    return S_OK;
}

HRESULT tfRandomPoint(unsigned int kind, tfFloatP_t dr, tfFloatP_t phi0, tfFloatP_t phi1, tfFloatP_t *x, tfFloatP_t *y, tfFloatP_t *z) {
    TFC_PTRCHECK(x); TFC_PTRCHECK(y); TFC_PTRCHECK(z);
    auto p = randomPoint((PointsType)kind, dr, phi0, phi1);
    *x = p.x(); *y = p.y(); *z = p.z();
    return S_OK;
}

HRESULT tfRandomPoints(unsigned int kind, int n, tfFloatP_t dr, tfFloatP_t phi0, tfFloatP_t phi1, tfFloatP_t **x) {
    auto pv = randomPoints((PointsType)kind, n, dr, phi0, phi1);
    return pv.size() > 0 ? TissueForge::capi::copyVecVecs3_2Arr(pv, x) : S_OK;
}

HRESULT tfPoints(unsigned int kind, int n, tfFloatP_t **x) {
    auto pv = points((PointsType)kind, n);
    return pv.size() > 0 ? TissueForge::capi::copyVecVecs3_2Arr(pv, x) : S_OK;
}

HRESULT tfFilledCubeUniform(
    tfFloatP_t *corner1, 
    tfFloatP_t *corner2, 
    unsigned int nParticlesX, 
    unsigned int nParticlesY, 
    unsigned int nParticlesZ, 
    tfFloatP_t **x) 
{
    TFC_PTRCHECK(corner1);
    TFC_PTRCHECK(corner2);
    auto pv = filledCubeUniform(FVector3::from(corner1), FVector3::from(corner2), nParticlesX, nParticlesY, nParticlesZ);
    return pv.size() > 0 ? TissueForge::capi::copyVecVecs3_2Arr(pv, x) : S_OK;
}

HRESULT tfFilledCubeRandom(tfFloatP_t *corner1, tfFloatP_t *corner2, int nParticles, tfFloatP_t **x) {
    TFC_PTRCHECK(corner1);
    TFC_PTRCHECK(corner2);
    auto pv = filledCubeRandom(FVector3::from(corner1), FVector3::from(corner2), nParticles);
    return pv.size() > 0 ? TissueForge::capi::copyVecVecs3_2Arr(pv, x) : S_OK;
}

HRESULT tfIcosphere(
    unsigned int subdivisions, 
    tfFloatP_t phi0, 
    tfFloatP_t phi1,
    tfFloatP_t **verts, unsigned int *numVerts,
    int **inds, unsigned int *numInds) 
{
    TFC_PTRCHECK(verts);
    TFC_PTRCHECK(numVerts);
    TFC_PTRCHECK(inds);
    TFC_PTRCHECK(numInds);

    std::vector<FVector3> _verts;
    std::vector<int> _indsV;
    HRESULT result = icosphere(subdivisions, phi0, phi1, _verts, _indsV);
    if(result != S_OK) 
        return result;

    *numInds = _indsV.size();
    *numVerts = _verts.size();
    int *_inds = (int*)malloc(*numInds * sizeof(int));
    if(!_inds) 
        return E_OUTOFMEMORY;
    for(unsigned int i = 0; i < _indsV.size(); i++) 
        _inds[i] = _indsV[i];
    *inds = _inds;
    return TissueForge::capi::copyVecVecs3_2Arr(_verts, verts);
}

HRESULT tfRandomVector(tfFloatP_t mean, tfFloatP_t std, tfFloatP_t *x, tfFloatP_t *y, tfFloatP_t *z) {
    TFC_PTRCHECK(x); TFC_PTRCHECK(y); TFC_PTRCHECK(z);
    auto pv = randomVector(mean, std);
    *x = pv.x(); *y = pv.y(); *z = pv.z();
    return S_OK;
}

HRESULT tfRandomUnitVector(tfFloatP_t *x, tfFloatP_t *y, tfFloatP_t *z) {
    TFC_PTRCHECK(x); TFC_PTRCHECK(y); TFC_PTRCHECK(z);
    auto pv = randomUnitVector();
    *x = pv.x(); *y = pv.y(); *z = pv.z();
    return S_OK;
}

HRESULT tfUtilGetFeaturesMap(char ***names, bool **flags, unsigned int *numFeatures) {
    TFC_PTRCHECK(names);
    TFC_PTRCHECK(flags);
    TFC_PTRCHECK(numFeatures);
    auto fmap = util::getFeaturesMap();
    *numFeatures = fmap.size();
    if(*numFeatures > 0) {
        char **_names = (char**)malloc(*numFeatures * sizeof(char*));
        bool *_flags = (bool*)malloc(*numFeatures * sizeof(bool));
        if(!_names || !_flags) 
            return E_OUTOFMEMORY;
        unsigned int i = 0;
        for(auto &fm : fmap) {
            char *_c = new char[fm.first.size() + 1];
            std::strcpy(_c, fm.first.c_str());
            _names[i] = _c;
            _flags[i] = fm.second;
            i++;
        }
        *names = _names;
        *flags = _flags;
    }
    return S_OK;
}

HRESULT tfUtilWallTime(double *wtime) {
    TFC_PTRCHECK(wtime);
    *wtime = util::wallTime();
    return S_OK;
}

HRESULT tfUtilCPUTime(double *cputime) {
    TFC_PTRCHECK(cputime);
    *cputime = util::CPUTime();
    return S_OK;
}
