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

#include <tfParticleTypeList.h>

#include <tfEngine.h>
#include <io/tfFIO.h>
#include <tf_mdcore_io.h>

#include <cstdarg>


using namespace TissueForge;


void TissueForge::ParticleTypeList::free() {}

HRESULT TissueForge::ParticleTypeList::reserve(size_t _nr_parts) {
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

uint16_t TissueForge::ParticleTypeList::insert(int32_t item) {
    
    if(nr_parts == size_parts) 
        reserve(size_parts + space_partlist_incr);
    
    parts[nr_parts] = item;

    return nr_parts++;
}

uint16_t TissueForge::ParticleTypeList::insert(const ParticleType *ptype) {
    if(ptype) return insert(ptype->id);

    tf_error(E_FAIL, "cannot insert a NULL type");
    return 0;
}

uint16_t TissueForge::ParticleTypeList::remove(int32_t id) {
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

void TissueForge::ParticleTypeList::extend(const ParticleTypeList &other) {
    if(other.nr_parts > size_parts) reserve(other.nr_parts);
    for(int i = 0; i < other.nr_parts; ++i) this->insert(other.parts[i]);
}

bool TissueForge::ParticleTypeList::has(const int32_t &pid) {
    for(size_t i = 0; i < nr_parts; i++) 
        if(parts[i] == pid) 
            return true;
    return false;
}

bool TissueForge::ParticleTypeList::has(ParticleType *ptype) {
    return ptype ? has(ptype->id) : false;
}

bool TissueForge::ParticleTypeList::has(ParticleHandle *part) {
    return particles().has(part);
}

ParticleType *TissueForge::ParticleTypeList::item(const int32_t &i) {
    if(i < nr_parts) {
        return &_Engine.types[parts[i]];
    }
    else {
        throw std::runtime_error("index out of range");
    }
    return NULL;
}

int32_t TissueForge::ParticleTypeList::operator[](const size_t &i) {
    if(i < nr_parts) {
        return this->parts[i];
    }
    else {
        throw std::runtime_error("index out of range");
    }
    return -1;
}

std::vector<int32_t> TissueForge::ParticleTypeList::vector() {
    std::vector<int32_t> result;
    result.reserve(this->nr_parts);
    for(size_t i = 0; i < nr_parts; i++) 
        result.push_back(parts[i]);
    return result;
}

ParticleList TissueForge::ParticleTypeList::particles() {
    ParticleList list;

    for(int tid = 0; tid < this->nr_parts; ++tid) list.extend(this->item(tid)->parts);

    return list;
}

ParticleTypeList TissueForge::ParticleTypeList::all() {
    ParticleTypeList list;
    
    for(int tid = 0; tid < _Engine.nr_types; tid++) list.insert(tid);
    
    return list;
}

FMatrix3 TissueForge::ParticleTypeList::getVirial() {
    return this->particles().getVirial();
}

FPTYPE TissueForge::ParticleTypeList::getRadiusOfGyration() {
    return this->particles().getRadiusOfGyration();
}

FVector3 TissueForge::ParticleTypeList::getCenterOfMass() {
    return this->particles().getCenterOfMass();
}

FVector3 TissueForge::ParticleTypeList::getCentroid() {
    return this->particles().getCentroid();
}

FMatrix3 TissueForge::ParticleTypeList::getMomentOfInertia() {
    return this->particles().getMomentOfInertia();
}

std::vector<FVector3> TissueForge::ParticleTypeList::getPositions() {
    return this->particles().getPositions();
}

std::vector<FVector3> TissueForge::ParticleTypeList::getVelocities() {
    return this->particles().getVelocities();
}

std::vector<FVector3> TissueForge::ParticleTypeList::getForces() {
    return this->particles().getForces();
}

std::vector<FVector3> TissueForge::ParticleTypeList::sphericalPositions(FVector3 *origin) {
    return this->particles().sphericalPositions(origin);
}

TissueForge::ParticleTypeList::ParticleTypeList() : 
    flags(PARTICLELIST_OWNDATA | PARTICLELIST_OWNSELF), 
    size_parts(0), 
    nr_parts(0)
{
    this->parts = (int32_t*)malloc(this->size_parts * sizeof(int32_t));
}

TissueForge::ParticleTypeList::ParticleTypeList(uint16_t init_size, uint16_t flags) : ParticleTypeList() {
    this->flags = flags;
    this->size_parts = init_size;
    ::free(this->parts);
    this->parts = (int32_t*)malloc(init_size * sizeof(int32_t));
}

TissueForge::ParticleTypeList::ParticleTypeList(ParticleType *ptype) : 
    ParticleTypeList(1, PARTICLELIST_OWNDATA | PARTICLELIST_OWNSELF)
{
    if(!ptype) throw std::runtime_error("Cannot instance a list from NULL type");
    
    this->nr_parts = 1;
    this->parts[0] = ptype->id;
}

TissueForge::ParticleTypeList::ParticleTypeList(std::vector<ParticleType> ptypes) : 
    ParticleTypeList(ptypes.size(), PARTICLELIST_OWNDATA | PARTICLELIST_OWNSELF)
{
    this->nr_parts = ptypes.size();
    
    for(int i = 0; i < nr_parts; ++i) {
        this->parts[i] = ptypes[i].id;
    }
}

TissueForge::ParticleTypeList::ParticleTypeList(std::vector<ParticleType*> ptypes) : 
    ParticleTypeList(ptypes.size(), PARTICLELIST_OWNDATA | PARTICLELIST_OWNSELF)
{
    this->nr_parts = ptypes.size();
    
    for(int i = 0; i < nr_parts; ++i) {
        ParticleType *t = ptypes[i];
        if(!t) {
            throw std::runtime_error("Cannot initialize a list with a NULL type");
        }
        this->parts[i] = t->id;
    }
}

TissueForge::ParticleTypeList::ParticleTypeList(uint16_t nr_parts, int32_t *ptypes) : 
    ParticleTypeList(nr_parts, PARTICLELIST_OWNDATA | PARTICLELIST_OWNSELF)
{
    this->nr_parts = nr_parts;
    memcpy(this->parts, parts, nr_parts * sizeof(int32_t));
}

TissueForge::ParticleTypeList::ParticleTypeList(const ParticleTypeList &other) : 
    ParticleTypeList(other.nr_parts, other.parts)
{}

TissueForge::ParticleTypeList::ParticleTypeList(const std::vector<int32_t> &pids) : 
    ParticleTypeList(pids.size(), PARTICLELIST_OWNSELF | PARTICLELIST_OWNDATA)
{
    this->nr_parts = pids.size();
    for(size_t i = 0; i < pids.size(); i++) 
        parts[i] = pids[i];
}

TissueForge::ParticleTypeList::~ParticleTypeList() {
    if(this->flags & PARTICLELIST_OWNDATA && size_parts > 0) {
        ::free(this->parts);
    }
}

std::string TissueForge::ParticleTypeList::toString() {
    return io::toString(*this);
}

ParticleTypeList *TissueForge::ParticleTypeList::fromString(const std::string &str) {
    return new ParticleTypeList(io::fromString<ParticleTypeList>(str));
}


namespace TissueForge::io {


    template <>
    HRESULT toFile(const ParticleTypeList &dataElement, const MetaData &metaData, IOElement &fileElement) { 

        std::vector<int32_t> parts;
        for(unsigned int i = 0; i < dataElement.nr_parts; i++) 
            parts.push_back(dataElement.parts[i]);
        TF_IOTOEASY(fileElement, metaData, "parts", parts);

        fileElement.get()->type = "ParticleTypeList";

        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, ParticleTypeList *dataElement) { 

        std::vector<int32_t> parts;
        TF_IOFROMEASY(fileElement, metaData, "parts", &parts);

        for(unsigned int i = 0; i < parts.size(); i++) 
            dataElement->insert(parts[i]);

        dataElement->flags = PARTICLELIST_OWNDATA |PARTICLELIST_MUTABLE | PARTICLELIST_OWNSELF;

        return S_OK;
    }

};
