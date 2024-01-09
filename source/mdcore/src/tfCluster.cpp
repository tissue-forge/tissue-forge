/*******************************************************************************
 * This file is part of mdcore.
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

#include <tfCluster.h>

#include <stdlib.h>
#include <math.h>

#include <tfParticle.h>
#include <tf_fptype.h>
#include <iostream>
#include <tf_util.h>
#include <tfLogger.h>
#include <tfError.h>
#include <tfEngine.h>
#include <tfSpace.h>
#include <tfSpace_cell.h>
#include <tf_metrics.h>
#include <tf_errs.h>

#include <rendering/tfStyle.h>


using namespace TissueForge;


/* the error macro. */
#define error(id)( tf_error(E_FAIL, errs_err_msg[id]) )


/**
 * removes a particle from the list at the index.
 * returns null if not found
 */
static Particle *remove_particle_at_index(Cluster *cluster, int index) {
    if(index >= cluster->nr_parts) {
        return NULL;
    }
    
    int pid = cluster->parts[index];
    
    for(int i = index; i + 1 < cluster->nr_parts; ++i) {
        cluster->parts[i] = cluster->parts[i+i];
    }
    
    cluster->nr_parts -= 1;
    cluster->parts[cluster->nr_parts] = -1;
    
    Particle *part = _Engine.s.partlist[pid];
    
    part->clusterId = -1;
    
    return part;
}

static ParticleHandle* cluster_fission_plane(Particle *cluster, const FVector4 &plane) {
    
    TF_Log(LOG_INFORMATION) << ", plane: " << plane;
    
    // particles to move to daughter cluster.
    // only perform a split if the contained particles can be split into
    // two non-empty sets.
    std::vector<int> dparts;
    
    for(int i = 0; i < cluster->nr_parts; ++i) {
        Particle *p = cluster->particle(i);
        FPTYPE dist = plane.distanceScaled(p->global_position());
        
        if(dist < 0) {
            dparts.push_back(p->id);
        }
    }
    
    if(dparts.size() > 0 && dparts.size() < cluster->nr_parts) {
        
        ParticleHandle *_daughter = Particle_New(cluster->handle()->type(),  NULL,  NULL);
        Cluster *daughter = (Cluster*)Particle_Get(_daughter);
        assert(daughter);
        
        TF_Log(LOG_TRACE) << "split cluster "
        << cluster->id << " into ("
        << cluster->id << ":" << (cluster->nr_parts - dparts.size())
        << ", "
        << daughter->id << ": " << dparts.size() << ")" << std::endl;
        
        for(int i = 0; i < dparts.size(); ++i) {
            cluster->removepart(dparts[i]);
            daughter->addpart(dparts[i]);
        }
        
        return _daughter;
    }
    else {
        return NULL;
    }
}

static ParticleHandle* cluster_fission_normal_point(Particle *cluster, const FVector3 &normal, const FVector3 &point) {
    
    TF_Log(LOG_DEBUG) << "normal: " << normal
                   << ", point: " << point << ", cluster center: "
                   << cluster->global_position();
    
    FVector4 plane = FVector4::planeEquation(normal, point);
    
    return cluster_fission_plane(cluster, plane);
}


static ParticleHandle* cluster_fission_axis(Particle *cluster, const FVector3 &axis) {
    
    TF_Log(LOG_DEBUG) << "axis: " << axis;
    
    FVector3 p1 = cluster->global_position();
    
    FVector3 p2 = p1 + axis;
    
    FVector3 p3 = p1 + randomUnitVector();
    
    FVector4 plane = FVector4::planeEquation(p1, p2, p3);
    
    return cluster_fission_plane(cluster, plane);
}

HRESULT TissueForge::Cluster_ComputeAggregateQuantities(struct Cluster *cluster) {
    
    if(cluster->nr_parts <= 0) {
        return S_OK;
    }
    
    FVector3 pos;
    
    // compute in global coordinates, particles can belong to different space cells.
    
    for(int i = 0; i < cluster->nr_parts; ++i) {
        Particle *p = _Engine.s.partlist[cluster->parts[i]];
        pos += p->global_position();
    }
    
    cluster->set_global_position(pos / cluster->nr_parts);
    
    return S_OK;
}


static ParticleHandle* cluster_fission_random(Particle *cluster)
{
    ParticleHandle *_daughter = Particle_New(cluster->handle()->type(),  NULL,  NULL);
    
    Cluster *daughter = (Cluster*)Particle_Get(_daughter);
    assert(daughter);
    
    int halfIndex = cluster->nr_parts / 2;
    
    for(int i = halfIndex; i < cluster->nr_parts; ++i) {
        // adds to new cluster, sets id of contained particle.
        daughter->addpart(cluster->parts[i]);
        cluster->parts[i] = -1;
    }
    
    cluster->nr_parts = halfIndex;
    
    return _daughter;
}

TissueForge::ClusterParticleType::ClusterParticleType(const bool &noReg) : 
    ParticleType(noReg) 
{
    this->particle_flags |= PARTICLE_CLUSTER;
}

std::string TissueForge::ClusterParticleType::str() const {
    std::stringstream ss;

    ss << "ClusterParticleType(id=" << this->id << ", name=" << this->name << ")";

    return ss.str();
}

bool TissueForge::ClusterParticleType::hasType(const ParticleType *type) {
    for (int tid = 0; tid < types.nr_parts; ++tid) {
        if (type->id == types.item(tid)->id) 
            return true;
    }
    return false;
}

bool TissueForge::ClusterParticleType::has(const int32_t &pid) {
    for(int tid = 0; tid < types.nr_parts; tid++) {
        if(types.parts[tid] == pid) 
            return true;

        ParticleType *ptype = types.item(tid);
        if(ptype->isCluster()) {
            ClusterParticleType *ctype = (ClusterParticleType*)ptype;
            if(ctype->has(pid)) 
                return true;
        }
    }

    return false;
}

bool TissueForge::ClusterParticleType::has(ParticleType *ptype) {
    return ptype ? has(ptype->id) : false;
}

bool TissueForge::ClusterParticleType::has(ParticleHandle *part) {
    return part ? parts.has(part) : false;
}

HRESULT TissueForge::ClusterParticleType::registerType() {
    for (int tid = 0; tid < types.nr_parts; ++tid) {
        auto type = types.item(tid);
        if (!type->isRegistered()) {
            auto result = type->registerType();
            if (result != S_OK) return result;
        }
    }

    return ParticleType::registerType();
}

ClusterParticleType *TissueForge::ClusterParticleType::get() {
    return (ClusterParticleType*)ParticleType::get();
}

TissueForge::ClusterParticleHandle::ClusterParticleHandle() : 
    ParticleHandle() 
{}

TissueForge::ClusterParticleHandle::ClusterParticleHandle(const int &id) : 
    ParticleHandle(id) 
{}

std::string TissueForge::ClusterParticleHandle::str() const {
    std::stringstream  ss;
    
    ss << "ClusterParticleHandle(";
    if(this->id >= 0) {
        ClusterParticleHandle ph(this->id);
        ss << "id=" << ph.getId() << ", typeId=" << ph.getTypeId() << ", clusterId=" << ph.getClusterId();
    }
    ss << ")";
    
    return ss.str();
}

Cluster *TissueForge::ClusterParticleHandle::cluster() {
    return (Cluster*)this->part();
}

ParticleHandle *TissueForge::ClusterParticleHandle::operator()(ParticleType *partType, 
                                                               FVector3 *position, 
                                                               FVector3 *velocity) 
{
    auto p = Cluster_CreateParticle((Cluster*)part(), partType, position, velocity);
    if(!p) {
        error(MDCERR_null);
        return NULL;
    }
    return new ParticleHandle(p->id);
}

ParticleHandle *TissueForge::ClusterParticleHandle::operator()(ParticleType *partType, const std::string &str) {
    Cluster *dummy = Cluster_fromString(str);
    ParticleHandle *p = (*this)(partType, &dummy->position, &dummy->velocity);
    delete dummy;
    return p;
}

bool TissueForge::ClusterParticleHandle::has(const int32_t &pid) {
    ParticleList plist = this->items();

    for(size_t i = 0; i < plist.nr_parts; i++) {
        if(plist.parts[i] == pid) 
            return true;

        ParticleHandle *ph = plist.item(i);
        if(ph->getClusterId() >= 0) {
            ClusterParticleHandle *ch = (ClusterParticleHandle*)ph;
            if(ch->has(pid)) 
                return true;
        }
    }

    return false;
}

bool TissueForge::ClusterParticleHandle::has(ParticleHandle *part) {
    return part ? has(part->id) : false;
}

/**
 # split the cell with a cleavage plane, in normal/point form.
 split(normal=[x, y, z], point=[px, py, pz])
 
 # split the cell with a cleavage plane normal, but use the clusters center as the point
 split(normal=[x, y, z])
 
 # if no named arguments are given, split interprets the first argument as a cleavage normal:
 split([x, y, z])
 
 # split using a cleavage *axis*, here, the split will generate a cleavage plane
 # that contains the given axis. This method is designed for the epiboly project,
 # where you’d give it an axis that’s the vector between the yolk center, and the
 # center of the cell. This will split the cell perpendicular to the yolk
 split(axis=[x, y, z])
 
 # default version of split uses a random cleavage plane that intersects the
 # cell center
 split()
*/
ParticleHandle* TissueForge::ClusterParticleHandle::fission(FVector3 *axis, 
                                                            bool *random, 
                                                            FPTYPE *time, 
                                                            FVector3 *normal, 
                                                            FVector3 *point) {

    TF_Log(LOG_TRACE) ;
    
    auto *cluster = part();
    
    if(!(Cluster*)cluster) {
        throw std::runtime_error("ERROR, given object is not a cluster");
        return NULL;
    }
    
    Cluster_ComputeAggregateQuantities((Cluster*)cluster);
    
    if(axis) {
        // use axis form of split
        return cluster_fission_axis(cluster, *axis);
    }
    
    if(random && *random) {
        // use random form of split
        return cluster_fission_random(cluster);
    }
    
    FVector3 _normal;
    FVector3 _point;
    
    // check if being called as an event, with the time number argument
    if(time) {
        FPTYPE t = *time;
        TF_Log(LOG_TRACE) << "cluster split event(cluster id: " << cluster->id
                  << ", count: " << cluster->nr_parts
                  << ", time: " << t << ")" << std::endl;
        _normal = randomUnitVector();
        _point = cluster->global_position();
    }
    else {
        // normal documented usage
        _normal = normal ? *normal : randomUnitVector();
        _point = point ? *point : cluster->global_position();
        
        TF_Log(LOG_TRACE) << "using cleavage plane to split cluster" << std::endl;
    }
    
    return cluster_fission_normal_point(cluster, _normal, _point);
}

ParticleHandle* TissueForge::ClusterParticleHandle::split(FVector3 *axis, 
                                                          bool *random, 
                                                          FPTYPE *time, 
                                                          FVector3 *normal, 
                                                          FVector3 *point) 
{ return fission(axis, random, time, normal, point); }

ParticleList TissueForge::ClusterParticleHandle::items() {
    Particle *self = this->part();
    return ParticleList(self->nr_parts, self->parts);
}

FPTYPE TissueForge::ClusterParticleHandle::getRadiusOfGyration() {
    Particle *self = this->part();
    FPTYPE result;
    metrics::particlesRadiusOfGyration(self->parts, self->nr_parts, &result);
    return result;
}

FVector3 TissueForge::ClusterParticleHandle::getCenterOfMass() {
    Particle *self = this->part();
    FVector3 result;
    metrics::particlesCenterOfMass(self->parts, self->nr_parts, result.data());
    return result;
}

FVector3 TissueForge::ClusterParticleHandle::getCentroid() {
    Particle *self = this->part();
    FVector3 result;
    metrics::particlesCenterOfGeometry(self->parts, self->nr_parts, result.data());
    return result;
}

FMatrix3 TissueForge::ClusterParticleHandle::getMomentOfInertia() {
    Particle *self = this->part();
    FMatrix3 result;
    metrics::particlesMomentOfInertia(self->parts, self->nr_parts, result.data());
    return result;
}

uint16_t TissueForge::ClusterParticleHandle::getNumParts() {
    Particle *self = this->part();
    return self->nr_parts;
}

std::vector<int32_t> TissueForge::ClusterParticleHandle::getPartIds() {
    Particle *self = this->part();
    std::vector<int32_t> result;
    result.reserve(self->nr_parts);
    for(size_t i = 0; i < self->nr_parts; i++) 
        result.push_back(self->parts[i]);
    return result;
}

/**
 * adds an existing particle to the cluster.
 */
HRESULT TissueForge::Cluster_AddParticle(struct Cluster *cluster, struct Particle *part) {
    part->flags |= PARTICLE_BOUND;
    cluster->addpart(part->id);
    return S_OK;
}

/**
 * creates a new particle, and adds it to the cluster.
 */
Particle *TissueForge::Cluster_CreateParticle(Cluster *cluster,
                                              ParticleType* particleType, 
                                              FVector3 *position, 
                                              FVector3 *velocity)
{
    TF_Log(LOG_TRACE);
    
    auto *type = &_Engine.types[cluster->typeId];
    if (!type->isCluster()) {
        error(MDCERR_notcluster);
        return NULL;
    }
    
    auto *clusterType = (ClusterParticleType*)type;
    TF_Log(LOG_TRACE) << type->id << ", " << particleType->id << ", " << clusterType->hasType(particleType);
    if (!clusterType->hasType(particleType)) {
        error(MDCERR_wrongptype);
        return NULL;
    }

    auto handle = Particle_New(particleType, position, velocity, &cluster->id);
    if (!handle) {
        error(MDCERR_null);
        return NULL;
    }

    return handle->part();

}

ClusterParticleType* TissueForge::ClusterParticleType_FindFromName(const char* name) {
    ParticleType *ptype = ParticleType_FindFromName(name);
    if(!ptype) return NULL;
    return (ClusterParticleType*)ptype;
}

HRESULT TissueForge::_Cluster_init() {
    TF_Log(LOG_TRACE);

    if(engine::nr_types != 1) {
        return error(MDCERR_initorder);
    }

    auto type = new ClusterParticleType();
    return S_OK;
}

Cluster *TissueForge::Cluster_fromString(const std::string &str) {
    return (Cluster*)Particle::fromString(str);
}

ClusterParticleType *TissueForge::ClusterParticleType_fromString(const std::string &str) {
    return (ClusterParticleType*)ParticleType::fromString(str);
}
