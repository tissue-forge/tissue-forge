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

#include <tf_util.h>

#include "tfThreeDFVertexData.h"
#include "tfThreeDFEdgeData.h"
#include "tfThreeDFFaceData.h"
#include "tfThreeDFMeshData.h"


namespace TissueForge::io {


    std::vector<ThreeDFVertexData*> ThreeDFMeshData::getVertices() {
        std::vector<ThreeDFVertexData*> result;
        for(auto e : this->faces) {
            auto v = e->getVertices();
            result.insert(result.end(), v.begin(), v.end());
        }
        return util::unique(result);
    }

    std::vector<ThreeDFEdgeData*> ThreeDFMeshData::getEdges() {
        std::vector<ThreeDFEdgeData*> result;
        for(auto e : this->faces) {
            auto v = e->getEdges();
            result.insert(result.end(), v.begin(), v.end());
        }
        return util::unique(result);
    }

    std::vector<ThreeDFFaceData*> ThreeDFMeshData::getFaces() {
        return this->faces;
    }

    unsigned int ThreeDFMeshData::getNumVertices() {
        return this->getVertices().size();
    }

    unsigned int ThreeDFMeshData::getNumEdges() {
        return this->getEdges().size();
    }

    unsigned int ThreeDFMeshData::getNumFaces() {
        return this->faces.size();
    }

    bool ThreeDFMeshData::has(ThreeDFVertexData *v) {
        for(auto f : this->faces) 
            if(f->has(v)) 
                return true;
        return false;
    }

    bool ThreeDFMeshData::has(ThreeDFEdgeData *e) {
        for(auto f : this->faces) 
            if(f->has(e)) 
                return true;
        return false;
    }

    bool ThreeDFMeshData::has(ThreeDFFaceData *f) {
        auto itr = std::find(this->faces.begin(), this->faces.end(), f);
        return itr != std::end(this->faces);
    }

    bool ThreeDFMeshData::in(ThreeDFStructure *s) {
        return s == this->structure;
    }

    FVector3 ThreeDFMeshData::getCentroid() { 
        auto vertices = this->getVertices();
        auto numV = vertices.size();

        if(numV == 0) 
            tf_error(E_FAIL, "No vertices");

        FVector3 result = {0.f, 0.f, 0.f};

        for(unsigned int i = 0; i < numV; i++) 
            result += vertices[i]->position;

        result /= numV;
        return result;
    }

    HRESULT ThreeDFMeshData::translate(const FVector3 &displacement) {
        for(auto v : this->getVertices()) 
            v->position += displacement;
        
        return S_OK;
    }

    HRESULT ThreeDFMeshData::translateTo(const FVector3 &position) {
        return this->translate(position - this->getCentroid());
    }

    HRESULT ThreeDFMeshData::rotateAt(const FMatrix3 &rotMat, const FVector3 &rotPt) {
        FMatrix4 t = FMatrix4::translation(rotPt) * FMatrix4::from(rotMat, FVector3(0.f)) * FMatrix4::translation(rotPt * -1.f);

        for(auto v : this->getVertices()) { 
            FVector4 p = {v->position.x(), v->position.y(), v->position.z(), 1.f};
            v->position = (t * p).xyz();
        }

        return S_OK;
    }

    HRESULT ThreeDFMeshData::rotate(const FMatrix3 &rotMat) {
        return this->rotateAt(rotMat, this->getCentroid());
    }

    HRESULT ThreeDFMeshData::scaleFrom(const FVector3 &scales, const FVector3 &scalePt) {
        if(scales[0] <= 0 || scales[1] <= 0 || scales[2] <= 0) 
            tf_error(E_FAIL, "Invalid non-positive scale");

        FMatrix4 t = FMatrix4::translation(scalePt) * FMatrix4::scaling(scales) * FMatrix4::translation(scalePt * -1.f);

        for(auto v : this->getVertices()) { 
            FVector4 p = {v->position.x(), v->position.y(), v->position.z(), 1.f};
            v->position = (t * p).xyz();
        }

        return S_OK;
    }

    HRESULT ThreeDFMeshData::scaleFrom(const FloatP_t &scale, const FVector3 &scalePt) {
        return this->scaleFrom(FVector3(scale), scalePt);
    }

    HRESULT ThreeDFMeshData::scale(const FVector3 &scales) {
        return this->scaleFrom(scales, this->getCentroid());
    }

    HRESULT ThreeDFMeshData::scale(const FloatP_t &scale) {
        return this->scale(FVector3(scale));
    }

};
