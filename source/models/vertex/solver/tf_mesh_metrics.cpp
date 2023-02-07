/*******************************************************************************
 * This file is part of Tissue Forge.
 * Copyright (c) 2022 T.J. Sego and Tien Comlekoglu
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

#include "tf_mesh_metrics.h"

#include <tfUniverse.h>
#include <tf_metrics.h>
#include <tfError.h>


using namespace TissueForge;


#define MeshMetrics_INVALIDHANDLERR { tf_error(E_FAIL, "Invalid handle"); }


static FMatrix3 calculateEdgeStrain(const FVector3 &pos_rel, const FVector3 &vel_rel) {
    FMatrix3 result;
    
    FloatP_t dt = Universe::getDt();
    FloatP_t pos_len2 = pos_rel.dot(pos_rel) / dt;
    FloatP_t nonlin_fact = vel_rel.dot(vel_rel) / pos_len2;

    for(size_t i = 0; i < 3; i++) {
        for(size_t j = i; j < 3; j++) {
            result[i][j] = pos_rel[i] * vel_rel[j] + pos_rel[j] * vel_rel[i] + pos_rel[i] * pos_rel[j] * nonlin_fact;
            if(j > i) 
                result[j][i] = result[i][j];
        }
    }

    return result * 0.5 / pos_len2;
}


namespace TissueForge::models::vertex {


static FMatrix3 MeshMetrics_edgeStrain(Vertex *v1, Vertex *v2) {
    ParticleHandle *p1 = v1->particle();
    ParticleHandle *p2 = v2->particle();

    FVector3 pos_rel = metrics::relativePosition(p2->getPosition(), p1->getPosition());
    FVector3 vel_rel = p2->getVelocity() - p1->getVelocity();
    return calculateEdgeStrain(pos_rel, vel_rel);
}

FMatrix3 edgeStrain(const VertexHandle &v1, const VertexHandle &v2) {
    Vertex *_v1 = v1.vertex();
    Vertex *_v2 = v2.vertex();
    if(!_v1 || !_v2) {
        MeshMetrics_INVALIDHANDLERR;
        return FMatrix3();
    }

    return MeshMetrics_edgeStrain(_v1, _v2);
}

FMatrix3 vertexStrain(const VertexHandle &v) {
    
    FMatrix3 result(0);

    Vertex *_v = v.vertex();
    if(!_v) {
        MeshMetrics_INVALIDHANDLERR;
        return result;
    }

    std::vector<Vertex*> nbs_v = _v->connectedVertices();
    if(nbs_v.size() == 0) {
        tf_error(E_FAIL, "Vertex is insufficiently connected");
        return result;
    }

    const FVector3 v_pos = _v->getPosition();

    FloatP_t totLen2 = 0;
    std::vector<FloatP_t> weights;
    weights.reserve(nbs_v.size());

    for(int i = 0; i < nbs_v.size(); i++) { 
        FloatP_t dist2 = metrics::relativePosition(v_pos, nbs_v[i]->getPosition()).dot();
        weights.push_back(dist2);
        totLen2 += dist2;
    }

    const FloatP_t fact = 2.0 / nbs_v.size();
    for(int i = 0; i < nbs_v.size(); i++) {
        const FloatP_t wi = fact - weights[i] / totLen2;
        result += MeshMetrics_edgeStrain(_v, nbs_v[i]) * wi;
    }

    return result;
}

};
