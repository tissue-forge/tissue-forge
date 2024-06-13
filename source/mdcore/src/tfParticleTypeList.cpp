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

#include <tfParticleTypeList.h>

#include <tfEngine.h>
#include <io/tfFIO.h>
#include <tf_mdcore_io.h>

#include <cstdarg>


using namespace TissueForge;


#define TYPELIST_IMMUTABLE_ERR tf_error(E_FAIL, "List is immutable")
#define TYPELIST_IMMUTABLE_CHECK_HRESULT(list) { if(!(list->flags & PARTICLELIST_MUTABLE)) return TYPELIST_IMMUTABLE_ERR; }
#define TYPELIST_IMMUTABLE_CHECK_LISTSZ(list) { if(!(list->flags & PARTICLELIST_MUTABLE)) { TYPELIST_IMMUTABLE_ERR; return list->nr_parts; } }


void TissueForge::ParticleTypeList::free() {
    if(this->flags & PARTICLELIST_OWNDATA && size_parts > 0 && this->parts) {
        ::free(this->parts);
    }
    this->parts = 0;
    this->nr_parts = 0;
    this->size_parts = 0;
}

HRESULT TissueForge::ParticleTypeList::reserve(size_t _nr_parts) {
    TYPELIST_IMMUTABLE_CHECK_HRESULT(this)

    if(size_parts < _nr_parts) {
        size_parts = _nr_parts;
        int32_t* temp = NULL;
        if ((temp = (int32_t*)malloc(sizeof(int32_t) * size_parts)) == NULL) {
            return tf_error(E_FAIL, "could not allocate space for type particles");
        }
        if(parts) {
            memcpy(temp, parts, sizeof(int32_t) * nr_parts);
            ::free(parts);
        }
        parts = temp;
    }
    return S_OK;
}

uint16_t TissueForge::ParticleTypeList::insert(int32_t item) {
    
    TYPELIST_IMMUTABLE_CHECK_LISTSZ(this)

    if(nr_parts == size_parts) 
        reserve(size_parts + space_partlist_incr);
    
    parts[nr_parts] = item;

    return nr_parts++;
}

uint16_t TissueForge::ParticleTypeList::insert(const ParticleType *ptype) {
    TYPELIST_IMMUTABLE_CHECK_LISTSZ(this)

    if(ptype) return insert(ptype->id);

    tf_error(E_FAIL, "cannot insert a NULL type");
    return this->nr_parts;
}

uint16_t TissueForge::ParticleTypeList::remove(int32_t id) {
    TYPELIST_IMMUTABLE_CHECK_LISTSZ(this)

    int i = 0;
    for(; i < nr_parts; i++) {
        if(parts[i] == id)
            break;
    }
    
    if(i == nr_parts) {
        tf_error(E_FAIL, "type does not contain particle id");
        return this->nr_parts;
    }
    
    nr_parts--;
    if(i < nr_parts) {
        parts[i] = parts[nr_parts];
    }
    
    return i;
}

uint16_t TissueForge::ParticleTypeList::extend(const ParticleTypeList &other) {
    TYPELIST_IMMUTABLE_CHECK_LISTSZ(this)

    if(other.nr_parts > size_parts) reserve(other.nr_parts);
    for(int i = 0; i < other.nr_parts; ++i) this->insert(other.parts[i]);
    return this->nr_parts;
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
        tf_error(E_FAIL, "index out of range");
    }
    return NULL;
}

int32_t TissueForge::ParticleTypeList::operator[](const size_t &i) {
    if(i < nr_parts) {
        return this->parts[i];
    }
    else {
        tf_error(E_FAIL, "index out of range");
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
    ParticleTypeList list(_Engine.nr_types, PARTICLELIST_MUTABLE);
    
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

bool TissueForge::ParticleTypeList::getOwnsData() const {
    return this->flags & PARTICLELIST_OWNDATA;
}

void TissueForge::ParticleTypeList::setOwnsData(const bool &_flag) {
    if(_flag) this->flags |= PARTICLELIST_OWNDATA;
    else this->flags &= ~PARTICLELIST_OWNDATA;
}

bool TissueForge::ParticleTypeList::getMutable() const {
    return this->flags & PARTICLELIST_MUTABLE;
}

void TissueForge::ParticleTypeList::setMutable(const bool &_flag) {
    if(_flag) this->flags |= PARTICLELIST_MUTABLE;
    else this->flags &= ~PARTICLELIST_MUTABLE;
}

TissueForge::ParticleTypeList::ParticleTypeList() : 
    flags(PARTICLELIST_OWNDATA | PARTICLELIST_MUTABLE), 
    size_parts(0), 
    nr_parts(0),
    parts(0)
{}

TissueForge::ParticleTypeList::ParticleTypeList(uint16_t init_size, uint16_t _flags) : ParticleTypeList() {
    this->flags = _flags;
    reserve(init_size);
}

TissueForge::ParticleTypeList::ParticleTypeList(ParticleType *ptype) : 
    ParticleTypeList(1, PARTICLELIST_OWNDATA | PARTICLELIST_MUTABLE)
{
    if(!ptype) {
        tf_error(E_FAIL, "Cannot instance a list from NULL type");
    }
    else {
        this->nr_parts = 1;
        this->parts[0] = ptype->id;
    }
}

TissueForge::ParticleTypeList::ParticleTypeList(std::vector<ParticleType> ptypes) : 
    ParticleTypeList(ptypes.size(), PARTICLELIST_OWNDATA | PARTICLELIST_MUTABLE)
{
    this->nr_parts = ptypes.size();
    
    for(int i = 0; i < nr_parts; ++i) {
        this->parts[i] = ptypes[i].id;
    }
}

TissueForge::ParticleTypeList::ParticleTypeList(std::vector<ParticleType*> ptypes) : 
    ParticleTypeList(ptypes.size(), PARTICLELIST_OWNDATA | PARTICLELIST_MUTABLE)
{
    this->nr_parts = ptypes.size();
    
    for(int i = 0; i < nr_parts; ++i) {
        ParticleType *t = ptypes[i];
        if(!t) {
            tf_error(E_FAIL, "Cannot initialize a list with a NULL type");
        }
        else 
            this->parts[i] = t->id;
    }
}

TissueForge::ParticleTypeList::ParticleTypeList(uint16_t _nr_parts, int32_t *ptypes) : 
    ParticleTypeList(_nr_parts, PARTICLELIST_OWNDATA | PARTICLELIST_MUTABLE)
{
    this->nr_parts = _nr_parts;
    memcpy(this->parts, ptypes, _nr_parts * sizeof(int32_t));
}

TissueForge::ParticleTypeList::ParticleTypeList(const ParticleTypeList &other) : 
    ParticleTypeList(other.nr_parts)
{
    this->nr_parts = other.nr_parts;
    memcpy(this->parts, other.parts, this->nr_parts * sizeof(int32_t));
}

TissueForge::ParticleTypeList::ParticleTypeList(const std::vector<int32_t> &pids) : 
    ParticleTypeList(pids.size(), PARTICLELIST_OWNDATA | PARTICLELIST_MUTABLE)
{
    this->nr_parts = pids.size();
    for(size_t i = 0; i < pids.size(); i++) 
        parts[i] = pids[i];
}

TissueForge::ParticleTypeList::~ParticleTypeList() {
    free();
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

        dataElement->flags = PARTICLELIST_OWNDATA | PARTICLELIST_MUTABLE;

        return S_OK;
    }

};
