/*******************************************************************************
 * This file is part of mdcore.
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

#include <tfParticleList.h>
#include <tfEngine.h>
#include <tf_metrics.h>
#include <tfError.h>
#include <io/tfFIO.h>
#include <tf_mdcore_io.h>

#include <cstdarg>
#include <iostream>


using namespace TissueForge;


void TissueForge::ParticleList::free()
{
}

HRESULT TissueForge::ParticleList::reserve(size_t _nr_parts) {
    if(size_parts < _nr_parts) {
        size_parts = _nr_parts;
        int32_t* temp = NULL;
        if ((temp = (int32_t*)malloc(sizeof(int32_t) * size_parts)) == NULL) {
            return tf_error(E_FAIL, "could not allocate space for type particles");
        }
        memcpy(temp, parts, sizeof(int32_t) * nr_parts);
        ::free(parts);
        parts = temp;
    }
    return S_OK;
}

uint16_t TissueForge::ParticleList::insert(int32_t id)
{

    if(nr_parts == size_parts) 
        reserve(size_parts + space_partlist_incr);
    
    parts[nr_parts] = id;

    return nr_parts++;
}

uint16_t TissueForge::ParticleList::insert(const ParticleHandle *particle) {
    if(particle) return insert(particle->id);

    tf_error(E_FAIL, "cannot insert a NULL particle");
    return 0;
}

uint16_t TissueForge::ParticleList::remove(int32_t id)
{
    int i = 0;
    for(; i < nr_parts; i++) {
        if(parts[i] == id)
            break;
    }
    
    if(i == nr_parts) {
        return tf_error(E_FAIL, "type does not contain particle id");
    }
    
    nr_parts--;
    if(i < nr_parts) {
        parts[i] = parts[nr_parts];
    }
    
    return i;
}

void TissueForge::ParticleList::extend(const ParticleList &other) {
    if(other.nr_parts > size_parts) reserve(other.nr_parts);
    for(int i = 0; i < other.nr_parts; ++i) this->insert(other.parts[i]);
}

ParticleHandle *TissueForge::ParticleList::item(const int32_t &i) {
    if(i < nr_parts) {
        Particle *part = _Engine.s.partlist[parts[i]];
        if(part) {
            return part->handle();
        }
    }
    else {
        throw std::runtime_error("index out of range");
    }
    return NULL;
}

TissueForge::ParticleList::ParticleList() : 
    flags(PARTICLELIST_OWNDATA | PARTICLELIST_OWNSELF), 
    size_parts(0), 
    nr_parts(0)
{
    this->parts = (int32_t*)malloc(this->size_parts * sizeof(int32_t));
}

TissueForge::ParticleList::ParticleList(uint16_t init_size, uint16_t flags) : ParticleList() {
    this->flags = flags;
    this->size_parts = init_size;
    ::free(this->parts);
    this->parts = (int32_t*)malloc(init_size * sizeof(int32_t));
}

TissueForge::ParticleList::ParticleList(ParticleHandle *part) : 
    ParticleList(1, PARTICLELIST_OWNDATA | PARTICLELIST_OWNSELF)
{
    if(!part) throw std::runtime_error("Cannot instance a list from NULL handle");
    
    Particle *p = part->part();
    if(!p) throw std::runtime_error("Cannot instance a list from NULL particle");

    this->nr_parts = 1;
    this->parts[0] = p->id;
}

TissueForge::ParticleList::ParticleList(std::vector<ParticleHandle*> particles) : 
    ParticleList(particles.size(), PARTICLELIST_OWNDATA | PARTICLELIST_OWNSELF)
{
    this->nr_parts = particles.size();
    
    for(int i = 0; i < nr_parts; ++i) {
        Particle *p = particles[i]->part();
        if(!p) {
            throw std::runtime_error("Cannot initialize a list with a NULL particle");
        }
        this->parts[i] = p->id;
    }
}

TissueForge::ParticleList::ParticleList(uint16_t nr_parts, int32_t *parts) : 
    ParticleList(nr_parts, PARTICLELIST_OWNDATA | PARTICLELIST_OWNSELF)
{
    this->nr_parts = nr_parts;
    memcpy(this->parts, parts, nr_parts * sizeof(int32_t));
}

TissueForge::ParticleList::ParticleList(const ParticleList &other) : 
    ParticleList(other.nr_parts, other.parts)
{}

TissueForge::ParticleList::~ParticleList() {
    if(this->flags & PARTICLELIST_OWNDATA && size_parts > 0) {
        ::free(this->parts);
    }
}

// TODO: in universe.bind, check keywords are correct, and no extra keyworkds
// TODO: simulator init, turn off periodoc if only single cell.
FMatrix3 TissueForge::ParticleList::getVirial()
{
    try {
        FMatrix3 result;
        if(SUCCEEDED(metrics::particlesVirial(this->parts, this->nr_parts, 0, result.data()))) {
            return result;
        }
    }
    catch(const std::exception &e) {
        tf_exp(e);
    }

    return FMatrix3();
}

FPTYPE TissueForge::ParticleList::getRadiusOfGyration() {
    try {
        FPTYPE result;
        if(SUCCEEDED(metrics::particlesRadiusOfGyration(this->parts, this->nr_parts, &result))) {
            return result;
        }
    }
    catch(const std::exception &e) {
        tf_exp(e);
    }
    return 0.0;
}

FVector3 TissueForge::ParticleList::getCenterOfMass() {
    try {
        FVector3 result;
        if(SUCCEEDED(metrics::particlesCenterOfMass(this->parts, this->nr_parts, result.data()))) {
            return result;
        }
    }
    catch(const std::exception &e) {
        tf_exp(e);
    }

    return FVector3();
}

FVector3 TissueForge::ParticleList::getCentroid() {
    try {
        FVector3 result;
        if(SUCCEEDED(metrics::particlesCenterOfGeometry(this->parts, this->nr_parts, result.data()))) {
            return result;
        }
    }
    catch(const std::exception &e) {
        tf_exp(e);
    }

    return FVector3();
}

FMatrix3 TissueForge::ParticleList::getMomentOfInertia() {
    try {
        FMatrix3 result;
        if(SUCCEEDED(metrics::particlesMomentOfInertia(this->parts, this->nr_parts, result.data()))) {
            return result;
        }
    }
    catch(const std::exception &e) {
        tf_exp(e);
    }

    return FMatrix3();
}

std::vector<FVector3> TissueForge::ParticleList::getPositions() {
    std::vector<FVector3> result(this->nr_parts);
    
    try {
        for(int i = 0; i < this->nr_parts; ++i) {
            Particle *part = _Engine.s.partlist[this->parts[i]];
            FVector3 pos = part->global_position();
            result[i] = FVector3(pos.x(), pos.y(), pos.z());
        }
    }
    catch(const std::exception &e) {
        tf_exp(e);
        result.clear();
    }
    
    return result;
}

std::vector<FVector3> TissueForge::ParticleList::getVelocities() {
    std::vector<FVector3> result(this->nr_parts);
    
    try {
        for(int i = 0; i < this->nr_parts; ++i) {
            Particle *part = _Engine.s.partlist[this->parts[i]];
            result[i] = FVector3(part->velocity[0], part->velocity[1], part->velocity[2]);
        }
    }
    catch(const std::exception &e) {
        tf_exp(e);
        result.clear();
    }
    
    return result;
}

std::vector<FVector3> TissueForge::ParticleList::getForces() {
    std::vector<FVector3> result(this->nr_parts);
    
    try{
        for(int i = 0; i < this->nr_parts; ++i) {
            Particle *part = _Engine.s.partlist[this->parts[i]];
            result[i] = FVector3(part->force[0], part->force[1], part->force[2]);
        }
    }
    catch(const std::exception &e) {
        tf_exp(e);
        result.clear();
    }
    
    return result;
}

std::vector<FVector3> TissueForge::ParticleList::sphericalPositions(FVector3 *origin) {
    std::vector<FVector3> result(this->nr_parts);

    FVector3 _origin;
    if(origin) _origin = *origin;
    else{
        auto center = FVector3(_Engine.s.dim[0], _Engine.s.dim[1], _Engine.s.dim[2]);
        _origin = center / 2;
    }
    
    for(int i = 0; i < this->nr_parts; ++i) {
        Particle *part = _Engine.s.partlist[this->parts[i]];
        FVector3 pos = part->global_position();
        result[i] = metrics::cartesianToSpherical(pos, *origin);
    }
    
    return result;
}

ParticleList *TissueForge::ParticleList::pack(size_t n, ...)
{
    int i;
    ParticleList *result = new ParticleList(n, PARTICLELIST_OWNDATA | PARTICLELIST_OWNSELF);
    va_list vargs;
    
    va_start(vargs, n);
    if (result == NULL) {
        va_end(vargs);
        return NULL;
    }

    for (i = 0; i < n; i++) {
        int o = va_arg(vargs, int);
        result->parts[i] = o;
    }
    va_end(vargs);
    return result;
}

ParticleList* TissueForge::ParticleList::all() {
    ParticleList* list = new ParticleList(_Engine.s.nr_parts);
    
    for (int cid = 0 ; cid < _Engine.s.nr_cells ; cid++) {
        space_cell *cell = &_Engine.s.cells[cid];
        for (int pid = 0 ; pid < cell->count ; pid++) {
            Particle *p  = &cell->parts[pid];
            list->insert(p->id);
        }
    }
    
    for (int pid = 0 ; pid < _Engine.s.largeparts.count ; pid++) {
        Particle *p  = &_Engine.s.largeparts.parts[pid];
        list->insert(p->id);
    }
    
    return list;
}

std::string TissueForge::ParticleList::toString() {
    return io::toString(*this);
}

ParticleList *TissueForge::ParticleList::fromString(const std::string &str) {
    return new ParticleList(io::fromString<ParticleList>(str));
}


namespace TissueForge::io {


    template <>
    HRESULT toFile(const ParticleList &dataElement, const MetaData &metaData, IOElement *fileElement) { 

        IOElement *fe;

        std::vector<int32_t> parts;
        for(unsigned int i = 0; i < dataElement.nr_parts; i++) 
            parts.push_back(dataElement.parts[i]);
        fe = new IOElement();
        if(toFile(parts, metaData, fe) != S_OK) 
            return E_FAIL;
        fe->parent = fileElement;
        fileElement->children["parts"] = fe;

        fileElement->type = "ParticleList";

        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, ParticleList *dataElement) { 

        IOChildMap::const_iterator feItr;
        std::vector<int32_t> parts;

        feItr = fileElement.children.find("parts");
        if(feItr == fileElement.children.end() || fromFile(*feItr->second, metaData, &parts) != S_OK) 
            return E_FAIL;

        for(unsigned int i = 0; i < parts.size(); i++) 
            dataElement->insert(parts[i]);

        dataElement->flags = PARTICLELIST_OWNDATA |PARTICLELIST_MUTABLE | PARTICLELIST_OWNSELF;

        return S_OK;
    }

};
