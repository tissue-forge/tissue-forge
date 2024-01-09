/*******************************************************************************
 * This file is part of mdcore.
 * Coypright (c) 2010 Pedro Gonnet (pedro.gonnet@durham.ac.uk)
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


/* include some standard header files */
#define _USE_MATH_DEFINES // for C++

#include <tfParticle.h>

#include <tf_errs.h>
#include <tfEngine.h>
#include <tfSpace.h>
#include <rendering/tfStyle.h>
#include <tfCluster.h>
#include <tf_metrics.h>
#include <tfParticleList.h>
#include <tfTaskScheduler.h>
#include <tf_util.h>
#include <tfLogger.h>
#include <tfError.h>
#include <io/tfFIO.h>
#include <state/tfSpeciesList.h>
#include <tf_mdcore_io.h>

#include <sstream>
#include <cstring>
#include <cmath>
#include <stdlib.h>
#include <iostream>
#include <unordered_map>
#include <typeinfo>


using namespace TissueForge;


#define error(id)				(tf_error(E_FAIL, errs_err_msg[id]))


TissueForge::Particle::Particle() {
    bzero(this, sizeof(Particle));
}

static unsigned colors [] = {
    0xCCCCCC,
    0x6D99D3, // Rust Oleum Spa Blue
    0xF65917, // Rust Oleum Pumpkin
    0xF9CB20, // rust oleum yellow
    0x3CB371, // green
    0x6353BB, // SGI purple
    0xf4AC21, // gold
    0xC4A5DF, // light purple
    0xDC143C, // dark red
    0x1E90FF, // blue
    0xFFFF00, // yellow
    0x8A2BE2, // purple
    0x76D7C4, // light green
    0xF08080, // salmon
    0xFF00FF, // fuscia
    0xFF8C00, // orange
    0xFFDAB9, // tan / peach
    0x7F8C8D, // gray
    0x884EA0, // purple
    0x6B8E23,
    0x00FFFF,
    0xAFEEEE,
    0x008080,
    0xB0E0E6,
    0x6495ED,
    0x191970,
    0x0000CD,
    0xD8BFD8,
    0xFF1493,
    0xF0FFF0,
    0xFFFFF0,
    0xFFE4E1,
    0xDCDCDC,
    0x778899,
    0x000000
};

unsigned int *TissueForge::Particle_Colors = colors;

#define TF_PARTICLE_SELF(pypart) \
    Particle *self = pypart->part(); \
    if(self == NULL) { \
        error(MDCERR_id); \
        return NULL; \
    }

#define TF_PARTICLE_PROP_SELF(pypart) \
    Particle *self = pypart->part(); \
    if(self == NULL) { \
        return error(MDCERR_id); \
    }

#define TF_PARTICLE_SELFW(pypart, ret) \
    Particle *self = pypart->part(); \
    if(self == NULL) { \
        error(MDCERR_id); \
        return ret; \
    }

#define TF_PARTICLE_TYPE(pypart) \
    ParticleType *ptype = pypart->type(); \
    if(ptype == NULL) { \
        error(MDCERR_id); \
        return NULL; \
    }

#define TF_PARTICLE_PROP_TYPE(pypart) \
    ParticleType *ptype = pypart->type(); \
    if(ptype == NULL) { \
        return error(MDCERR_id); \
    }

#define TF_PARTICLE_TYPEW(pypart, ret) \
    ParticleType *ptype = pypart->type(); \
    if(ptype == NULL) { \
        error(MDCERR_id); \
        return ret; \
    }

static HRESULT particle_ex_construct(
    ParticleHandle *self,  
    ParticleType *ptype, 
    const FVector3 &position, 
    const FVector3 &velocity, 
    int clusterId, 
    Particle &part
);
static HRESULT particle_ex_load(ParticleHandle *self, Particle &part);
static HRESULT particles_ex_load(std::vector<ParticleHandle*> selfs, std::vector<Particle*> parts);
static HRESULT particle_init(
    ParticleHandle *self, 
    ParticleType *ptype, 
    FVector3 *position=NULL, 
    FVector3 *velocity=NULL, 
    int *cluster=NULL
);
static HRESULT particle_init_ex(
    ParticleHandle *self,  
    ParticleType *ptype, 
    const FVector3 &position,
    const FVector3 &velocity,
    int clusterId
);


static ParticleList *particletype_items(ParticleType *self);

static Particle *particleSelf(ParticleHandle *handle);


struct Offset {
    uint32_t kind;
    uint32_t offset;
};

static_assert(sizeof(Offset) == sizeof(void*), "error, void* must be 64 bit");

static FVector3 particle_posdefault() { 
    RandomType &randEng = randomEngine();
    auto eng_origin = engine_origin();
    auto eng_dims = engine_dimensions();
    std::uniform_real_distribution<FPTYPE> x(eng_origin[0], eng_dims[0]);
    std::uniform_real_distribution<FPTYPE> y(eng_origin[1], eng_dims[1]);
    std::uniform_real_distribution<FPTYPE> z(eng_origin[2], eng_dims[2]);
    return {x(randEng), y(randEng), z(randEng)};
}

static FVector3 particle_dirdefault() {
    std::uniform_real_distribution<FPTYPE> x(-1, 1);
    RandomType &randEng = randomEngine();
    return {x(randEng), x(randEng), x(randEng)};
}

static FVector3 particle_veldefault(const ParticleType &ptype) {
    FVector3 _velocity;

    if(ptype.target_energy <= 0) _velocity = {0.0, 0.0, 0.0};
    else {
        RandomType &randEng = randomEngine();
        // initial velocity, chosen to fit target temperature
        std::uniform_real_distribution<FPTYPE> v(-1.0, 1.0);
        _velocity = {v(randEng), v(randEng), v(randEng)};
        FPTYPE v2 = _velocity.dot();
        FPTYPE x2 = (ptype.target_energy * 2. / (ptype.mass * v2));
        _velocity *= std::sqrt(x2);
    }

    return _velocity;
}

FPTYPE TissueForge::ParticleHandle::getCharge() {
    TF_PARTICLE_SELFW(this, 0)
    return self->q;
}

void TissueForge::ParticleHandle::setCharge(const FPTYPE &charge) {
    TF_PARTICLE_SELFW(this,)
    self->q = charge;
}

FPTYPE TissueForge::ParticleHandle::getMass() {
    TF_PARTICLE_SELFW(this, 0)
    return self->mass;
}

void TissueForge::ParticleHandle::setMass(const FPTYPE &mass) {
    if(mass <= 0.f) {
        error(MDCERR_badprop);
        return;
    }

    TF_PARTICLE_SELFW(this,)
    self->mass = mass;
    self->imass = 1.f / mass;
}

FPTYPE TissueForge::ParticleType::getVolume() {
    return 4.0 / 3.0 * M_PI * radius * radius * radius;
}

bool TissueForge::ParticleType::getFrozen() {
    return particle_flags & PARTICLE_FROZEN;
}

void TissueForge::ParticleType::setFrozen(const bool &frozen) {
    if(frozen) particle_flags |= PARTICLE_FROZEN;
    else particle_flags &= ~PARTICLE_FROZEN;
}

bool TissueForge::ParticleType::getFrozenX() {
    return particle_flags & PARTICLE_FROZEN_X;
}

void TissueForge::ParticleType::setFrozenX(const bool &frozen) {
    if(frozen) particle_flags |= PARTICLE_FROZEN_X;
    else particle_flags &= ~PARTICLE_FROZEN_X;
}

bool TissueForge::ParticleType::getFrozenY() {
    return particle_flags & PARTICLE_FROZEN_Y;
}

void TissueForge::ParticleType::setFrozenY(const bool &frozen) {
    if(frozen) particle_flags |= PARTICLE_FROZEN_Y;
    else particle_flags &= ~PARTICLE_FROZEN_Y;
}

bool TissueForge::ParticleType::getFrozenZ() {
    return particle_flags & PARTICLE_FROZEN_Z;
}

void TissueForge::ParticleType::setFrozenZ(const bool &frozen) {
    if(frozen) particle_flags |= PARTICLE_FROZEN_Z;
    else particle_flags &= ~PARTICLE_FROZEN_Z;
}

bool TissueForge::ParticleHandle::getFrozen() {
    TF_PARTICLE_SELFW(this, 0)
    return self->flags & PARTICLE_FROZEN;
}

void TissueForge::ParticleHandle::setFrozen(const bool frozen) {
    TF_PARTICLE_SELFW(this,)
    if(frozen) self->flags |= PARTICLE_FROZEN;
    else self->flags &= ~PARTICLE_FROZEN;
}

bool TissueForge::ParticleHandle::getFrozenX() {
    TF_PARTICLE_SELFW(this, 0)
    return self->flags & PARTICLE_FROZEN_X;
}

void TissueForge::ParticleHandle::setFrozenX(const bool frozen) {
    TF_PARTICLE_SELFW(this,)
    if(frozen) self->flags |= PARTICLE_FROZEN_X;
    else self->flags &= ~PARTICLE_FROZEN_X;
}

bool TissueForge::ParticleHandle::getFrozenY() {
    TF_PARTICLE_SELFW(this, 0)
    return self->flags & PARTICLE_FROZEN_Y;
}

void TissueForge::ParticleHandle::setFrozenY(const bool frozen) {
    TF_PARTICLE_SELFW(this,)
    if(frozen) self->flags |= PARTICLE_FROZEN_Y;
    else self->flags &= ~PARTICLE_FROZEN_Y;
}

bool TissueForge::ParticleHandle::getFrozenZ() {
    TF_PARTICLE_SELFW(this, 0)
    return self->flags & PARTICLE_FROZEN_Z;
}

void TissueForge::ParticleHandle::setFrozenZ(const bool frozen) {
    TF_PARTICLE_SELFW(this,)
    if(frozen) self->flags |= PARTICLE_FROZEN_Z;
    else self->flags &= ~PARTICLE_FROZEN_Z;
}

rendering::Style *TissueForge::ParticleHandle::getStyle() {
    TF_PARTICLE_SELFW(this, 0)
    return self->style;
}

void TissueForge::ParticleHandle::setStyle(rendering::Style *style) {
    TF_PARTICLE_SELFW(this,)
    self->style = style;
}

FPTYPE TissueForge::ParticleHandle::getAge() {
    TF_PARTICLE_SELFW(this, 0)
    return (_Engine.time - self->creation_time) * _Engine.dt;
}

FPTYPE TissueForge::ParticleHandle::getRadius() {
    TF_PARTICLE_SELFW(this, 0)
    return self->radius;
}

void TissueForge::ParticleHandle::setRadius(const FPTYPE &radius) {
    TF_PARTICLE_SELFW(this,)
    self->radius = radius;
    if((radius > _Engine.s.cutoff && !(self->flags & PARTICLE_LARGE)) || (radius <= _Engine.s.cutoff && self->flags & PARTICLE_LARGE)) 
        error(MDCERR_large_state);
}

FPTYPE TissueForge::ParticleHandle::getVolume() {
    TF_PARTICLE_SELFW(this, 0)
    return 4.0 / 3.0 * M_PI * self->radius * self->radius * self->radius;
}

std::string TissueForge::ParticleHandle::getName() {
    TF_PARTICLE_TYPEW(this, "")
    return ptype->name;
}

std::string TissueForge::ParticleHandle::getName2() {
    TF_PARTICLE_TYPEW(this, "")
    return ptype->name2;
}

FPTYPE TissueForge::ParticleType::getTemperature() {
    return this->kinetic_energy;
}

FPTYPE TissueForge::ParticleType::getTargetTemperature() {
    return this->target_energy;
}

void TissueForge::ParticleType::setTargetTemperature(const FPTYPE &temperature) {
    this->target_energy = temperature;
}

FVector3 TissueForge::ParticleHandle::getPosition() {
    TF_PARTICLE_SELFW(this, FVector3(0.0))
    return self->global_position();
}

void TissueForge::ParticleHandle::setPosition(FVector3 position) {
    TF_PARTICLE_SELFW(this,)
    BoundaryConditions::boundedPosition(position);
    self->set_global_position(position);
}

FVector3 &TissueForge::ParticleHandle::getVelocity() {
    auto self = particleSelf(this);
    return self->velocity;
}

void TissueForge::ParticleHandle::setVelocity(FVector3 velocity) {
    TF_PARTICLE_SELFW(this,)
    self->velocity = velocity;
}

FVector3 TissueForge::ParticleHandle::getForce() {
    TF_PARTICLE_SELFW(this, FVector3(0.0))
    return self->force;
}

FVector3 &TissueForge::ParticleHandle::getForceInit() {
    auto self = particleSelf(this);
    return self->force_i;
}

void TissueForge::ParticleHandle::setForceInit(FVector3 force) {
    TF_PARTICLE_SELFW(this,)
    self->force_i = force;
}

int TissueForge::ParticleHandle::getId() {
    TF_PARTICLE_SELFW(this, 0)
    return self->id;
}

int16_t TissueForge::ParticleHandle::getTypeId() {
    TF_PARTICLE_SELFW(this, 0)
    return self->typeId;
}

int32_t TissueForge::ParticleHandle::getClusterId() {
    TF_PARTICLE_SELFW(this, -1)
    return self->clusterId;
}

uint16_t TissueForge::ParticleHandle::getFlags() {
    TF_PARTICLE_SELFW(this, 0)
    return self->flags;
}

state::StateVector *TissueForge::ParticleHandle::getSpecies() {
    TF_PARTICLE_SELFW(this, 0)
    return self->state_vector;
}

TissueForge::ParticleHandle::operator ClusterParticleHandle*() {
    TF_PARTICLE_TYPEW(this, NULL)
    if (!ptype->isCluster()) return NULL;
    return static_cast<ClusterParticleHandle*>(this);
}

int typeIdByName(const char *_name) {
    for(int i = 0; i < _Engine.nr_types; i++) {
        if(strcmp(_Engine.types[i].name, _name) == 0)
            return _Engine.types[i].id;
    }

    return -1;
}

// Check whether a type name has been registered
bool checkTypeName(const char *_name) {
    return typeIdByName(_name) >= 0;
}

// Check whether a type name has been registered among all registered derived types
bool checkDerivedTypeName(const char *_name) {
    return typeIdByName(_name) >= 2;
}

// If the returned type id is the same as the two base types, 
// then generate a name according to type info. 
// Otherwise, this type presumably has already been registered with a unique name. 
std::string getUniqueName(ParticleType *type) {
    if(checkTypeName(type->name) && !checkDerivedTypeName(type->name)) return typeid(*type).name();
    return type->name;
}

void assignUniqueTypeName(ParticleType *type) {
    if(checkDerivedTypeName(type->name)) return;

    std::strncpy(type->name, getUniqueName(type).c_str(), ParticleType::MAX_NAME);
}

bool TissueForge::ParticleType_checkRegistered(ParticleType *type) {
    if(!type) return 0;
    int typeId = typeIdByName(type->name);
    return typeId > 1;
}

HRESULT TissueForge::_Particle_init()
{
    if(engine::max_type < 3) 
        return error(MDCERR_min_types);

    if(engine::nr_types != 0) 
        return error(MDCERR_initorder);

    if((engine::types = (ParticleType *)malloc(sizeof(ParticleType) * engine::max_type)) == NULL) 
        return error(MDCERR_malloc);
    
    ::memset(engine::types, 0, sizeof(ParticleType) * engine::max_type);

    engine::nr_types = 0;
    auto type = new ParticleType();
    
    return S_OK;
}

ParticleType* TissueForge::ParticleType_New(const char *_name) {
    auto type = new ParticleType(*Particle_GetType());
    std::strncpy(type->name, std::string(_name).c_str(), ParticleType::MAX_NAME);
    return type;
}

ParticleType* TissueForge::ParticleType_ForEngine(struct engine *e, FPTYPE mass,
        FPTYPE charge, const char *name, const char *name2)
{
    ParticleType *result = ParticleType_New(name);
    
    result->mass = mass;
    result->charge = charge;
    std::strncpy(result->name2, std::string(name2).c_str(), ParticleType::MAX_NAME);

    return result;
}

TissueForge::ParticleType::ParticleType(const bool &noReg) {
    radius = 1.0;
    minimum_radius = 0.0;
    mass = 1.0;
    charge = 0.0;
    id = 0;
    dynamics = PARTICLE_NEWTONIAN;
    type_flags = PARTICLE_TYPE_NONE;
    particle_flags = PARTICLE_NONE;
    species = NULL;
    
    fVector3 c = Magnum::Color3::fromSrgb(colors[(_Engine.nr_types - 1) % (sizeof(colors)/sizeof(unsigned))]);
    style = new rendering::Style(&c);

    ::strncpy(name, "Particle", ParticleType::MAX_NAME);
    ::strncpy(name2, "Particle", ParticleType::MAX_NAME);

    if(!noReg) registerType();
}

std::string TissueForge::ParticleType::str() const {
    std::stringstream ss;

    ss << "ParticleType(id=" << this->id << ", name=" << this->name << ")";

    return ss.str();
}

ParticleType* TissueForge::Particle_GetType()
{
    return &engine::types[0];
}

ParticleType* TissueForge::Cluster_GetType()
{
    return &engine::types[1];
}

Particle *particleSelf(ParticleHandle *handle) {
    TF_PARTICLE_SELF(handle)
    return self;
}

HRESULT TissueForge::ParticleHandle::destroy()
{
    TF_PARTICLE_SELFW(this, S_OK)
    return engine_del_particle(&_Engine, self->id);
}

FVector3 TissueForge::ParticleHandle::sphericalPosition(Particle *particle, FVector3 *origin)
{
    FVector3 _origin;

    if (particle) _origin = particle->global_position();
    else if (origin) _origin = *origin;
    else _origin = engine_center();
    return metrics::cartesianToSpherical(part()->global_position(), _origin);
}

FVector3 TissueForge::ParticleHandle::relativePosition(const FVector3 &origin, const bool &comp_bc) {
    return metrics::relativePosition(this->getPosition(), origin, comp_bc);
}

FMatrix3 TissueForge::ParticleHandle::virial(FPTYPE *radius)
{
    TF_PARTICLE_SELFW(this, FMatrix3(0.0))

    FVector3 pos = self->global_position();
    FMatrix3 mat;
    
    FPTYPE _radius = radius ? *radius : self->radius * 10;
    
    std::set<short int> typeIds;
    for(int i = 0; i < _Engine.nr_types; ++i) {
        typeIds.emplace(i);
    }
    
    metrics::calculateVirial(pos.data(), _radius, typeIds, mat.data());

    return mat;
}

HRESULT TissueForge::ParticleType::addpart(int32_t id)
{
    this->parts.insert(id);
    return S_OK;
}

HRESULT TissueForge::ParticleType::del_part(int32_t id) {
    this->parts.remove(id);
    return S_OK;
}

std::set<short int> TissueForge::ParticleType::particleTypeIds() {
	std::set<short int> ids;

	for(int i = 0; i < _Engine.nr_types; ++i) {
		if(!_Engine.types[i].isCluster()) ids.insert(i);
	}

	return ids;
}

bool TissueForge::ParticleType::isCluster() {
    return this->particle_flags & PARTICLE_CLUSTER;
}

TissueForge::ParticleType::operator ClusterParticleType*() {
    if (!this->isCluster()) return NULL;
    return static_cast<ClusterParticleType*>(this);
}

ParticleHandle *TissueForge::ParticleType::operator()(FVector3 *position,
                                             FVector3 *velocity,
                                             int *clusterId) 
{
    return Particle_New(this, position, velocity, clusterId);
}

ParticleHandle *TissueForge::ParticleType::operator()(const std::string &str, int *clusterId) {
    Particle *dummy = Particle::fromString(str);

    ParticleHandle *ph = (*this)(&dummy->position, &dummy->velocity, clusterId);
    auto p = ph->part();

    // copy reamining valid imported data

    p->force = dummy->force;
    p->force_i = dummy->force_i;
    p->inv_number_density = dummy->inv_number_density;
    p->creation_time = dummy->creation_time;
    p->persistent_force = dummy->persistent_force;
    p->radius = dummy->radius;
    p->mass = dummy->mass;
    p->imass = dummy->imass;
    p->q = dummy->q;
    p->p0 = dummy->p0;
    p->v0 = dummy->v0;
    for(unsigned int i = 0; i < 4; i++) {
        p->xk[i] = dummy->xk[i];
        p->vk[i] = dummy->vk[i];
    }
    p->vid = dummy->vid;
    p->flags = dummy->flags;
    if(dummy->state_vector) 
        p->state_vector = dummy->state_vector;

    delete dummy;

    return ph;
}

std::vector<int> TissueForge::ParticleType::factory(
    unsigned int nr_parts, 
    std::vector<FVector3> *positions, 
    std::vector<FVector3> *velocities, 
    std::vector<int> *clusterIds) 
{
    return Particles_New(this, nr_parts, positions, velocities, clusterIds);
}

ParticleType* TissueForge::ParticleType::newType(const char *_name) {
    auto type = new ParticleType(*this);
    std::strncpy(type->name, std::string(_name).c_str(), ParticleType::MAX_NAME);
    return type;
}

bool TissueForge::ParticleType::has(const int32_t &pid) {
    return parts.has(pid);
}

bool TissueForge::ParticleType::has(ParticleHandle *part) {
    return part ? has(part->id) : false;
}

HRESULT TissueForge::ParticleType::registerType() {
    if (isRegistered()) 
        return S_OK;

    if(engine::nr_types >= engine::max_type) 
        return error(MDCERR_malloc);

    if(engine::nr_types >= 2 && !checkDerivedTypeName(this->name)) {
        assignUniqueTypeName(this);
        TF_Log(LOG_INFORMATION) << "Type name not unique. Generating name: " << this->name;
    }

    TF_Log(LOG_DEBUG) << "Creating new particle type " << engine::nr_types;

    if(this->mass > 0.f) this->imass = 1.0f / this->mass;
    
    this->id = engine::nr_types;
    memcpy(&engine::types[engine::nr_types], this, sizeof(ParticleType));
    engine::nr_types++;

    // invoke callbacks
    this->on_register();

    return S_OK;
}

bool TissueForge::ParticleType::isRegistered() { return ParticleType_checkRegistered(this); }

ParticleType *TissueForge::ParticleType::get() {
    return ParticleType_FindFromName(this->name);
}

ParticleHandle* TissueForge::Particle_split(
    Particle* self,
    const FVector3& childDirection,
    const FPTYPE& childRatio,
    const std::vector<FPTYPE>* speciesRatios,
    ParticleType* parentType,
    ParticleType* childType
) {
    TF_Log(LOG_TRACE) << "Executing particle split " << self->id << ", " << (int)self->typeId;

    const FPTYPE parentRatio = FPTYPE_ONE - childRatio;

    if(childRatio <= FPTYPE_ZERO || parentRatio <= FPTYPE_ZERO) {
        error(MDCERR_range);
        return NULL;
    }

    int self_id = self->id;

    ParticleType* _parentType = parentType ? parentType : &_Engine.types[self->typeId];
    ParticleType* _childType = childType ? childType : _parentType;
    
    // volume preserving radii
    FPTYPE rC = self->radius * std::pow(childRatio, 1/3.);
    FPTYPE rP = self->radius * std::pow(parentRatio, 1/3.);
    
    if(rC < _childType->minimum_radius || rP < _parentType->minimum_radius) {
        return NULL;
    }

    Particle part = {};
    part.mass = self->mass;
    part.position = self->position;
    part.velocity = self->velocity;
    part.force = {};
    part.persistent_force = {};
    part.q = self->q;
    part.radius = self->radius;
    part.id = engine_next_partid(&_Engine);
    part.vid = 0;
    part.typeId = _childType->id;
    part.flags = self->flags;
    part._handle = NULL;
    part.parts = NULL;
    part.nr_parts = 0;
    part.size_parts = 0;
    part.creation_time = _Engine.time;
    if(part.radius > _Engine.s.cutoff) {
        part.flags |= PARTICLE_LARGE;
    }
    if(self->state_vector) {
        part.state_vector = new state::StateVector(*self->state_vector);

        std::vector<FPTYPE> _speciesRatios;
        if(speciesRatios) 
            _speciesRatios = *speciesRatios;
        else 
            _speciesRatios = std::vector<FPTYPE>(self->state_vector->size, childRatio);

        if(_speciesRatios.size() != self->state_vector->size) {
            TF_Log(LOG_ERROR) << "Species size do not match specified ratios";
        } 
        else {
            for(unsigned int i = 0; i < _speciesRatios.size(); i++) {
                const FPTYPE ratio_i = _speciesRatios[i];
                if(ratio_i < FPTYPE_ZERO || ratio_i > FPTYPE_ONE) {
                    error(MDCERR_range);
                    continue;
                }
                const FPTYPE fvec_i = self->state_vector->fvec[i];
                part.state_vector->fvec[i] = fvec_i * ratio_i / childRatio;
                self->state_vector->fvec[i] = fvec_i * (FPTYPE_ONE - ratio_i) / parentRatio;
            }
        }
    }

    // create a new particle at the same location as the original particle.
    Particle *p = NULL;
    FVector3 vec(0.0);

    if(space_getpos(&_Engine.s, self->id, vec.data()) != S_OK) {
        TF_Log(LOG_CRITICAL) << part.typeId << ", " << _Engine.nr_types;
        TF_Log(LOG_CRITICAL) << vec;
        TF_Log(LOG_CRITICAL) << part.id << ", " << self->id << ", " << _Engine.s.nr_parts;
        error(MDCERR_space);
        return NULL;
    }

    // Calculate new positions; account for boundaries
    // Particles should be in contact and center of mass should not change
    const FVector3 dir = childDirection.normalized();
    FVector3 posParent = BoundaryConditions::boundedPosition(vec - dir * childRatio * (rC + rP));
    FVector3 posChild = BoundaryConditions::boundedPosition(vec + dir * (FPTYPE_ONE - childRatio) * (rC + rP));

    if(engine_addpart(&_Engine, &part, posChild.data(), &p) != S_OK) {
        TF_Log(LOG_CRITICAL) << part.typeId << ", " << _Engine.nr_types;
        TF_Log(LOG_CRITICAL) << posParent;
        TF_Log(LOG_CRITICAL) << posChild;
        TF_Log(LOG_CRITICAL) << part.id << ", " << _Engine.s.nr_parts;
        error(MDCERR_engine);
        return NULL;
    }
    
    // pointers after engine_addpart could change...
    space_setpos(&_Engine.s, self_id, posParent.data());
    space_setpos(&_Engine.s, part.id, posChild.data());
    self = _Engine.s.partlist[self_id];
    p = _Engine.s.partlist[part.id];
    TF_Log(LOG_DEBUG) << self->position << ", " << p->position;
    
    // all is good, set the new radii
    p->radius = rC;
    self->radius = rP;
    p->mass = self->mass * childRatio;
    self->mass = self->mass * parentRatio;
    p->imass = p->mass > 0 ? 1. / p->mass : 0;
    self->imass = self->mass > 0 ? 1. / self->mass : 0;

    if(parentType) 
        self->handle()->become(parentType);

    TF_Log(LOG_TRACE) << "Simple fission for type " << (int)_Engine.types[self->typeId].id;

    return p->handle();
}

std::string TissueForge::ParticleHandle::str() const {
    std::stringstream  ss;
    
    ss << "ParticleHandle(";
    if(this->id >= 0) {
        ParticleHandle ph(this->id);
        ss << "id=" << ph.getId() << ", typeId=" << ph.getTypeId();
        auto clusterId = ph.getClusterId();
        if(clusterId >= 0) 
            ss << ", clusterId=" << clusterId;
    }
    ss << ")";
    
    return ss.str();
}

ParticleHandle* TissueForge::ParticleHandle::fission()
{
    TF_PARTICLE_SELF(this)
    return Particle_split(self, particle_dirdefault(), FPTYPE_ONE / FPTYPE_TWO);
}

ParticleHandle *TissueForge::ParticleHandle::split() { return fission(); }

ParticleHandle* TissueForge::ParticleHandle::split(const FVector3& direction) {
    TF_PARTICLE_SELF(this)
    return Particle_split(self, direction, FPTYPE_ONE / FPTYPE_TWO);
}

ParticleHandle* TissueForge::ParticleHandle::split(const FPTYPE& childRatio) {
    TF_PARTICLE_SELF(this)
    return Particle_split(self, particle_dirdefault(), childRatio);
}

ParticleHandle* TissueForge::ParticleHandle::split(const std::vector<FPTYPE>& speciesRatios) {
    TF_PARTICLE_SELF(this)
    std::vector<FPTYPE> _speciesRatios(speciesRatios);
    return Particle_split(self, particle_dirdefault(), FPTYPE_ONE / FPTYPE_TWO, &_speciesRatios);
}

ParticleHandle* TissueForge::ParticleHandle::split(
    const FVector3& childDirection,
    const FPTYPE& childRatio,
    ParticleType* parentType,
    ParticleType* childType
) {
    TF_PARTICLE_SELF(this)
    return Particle_split(self, childDirection, childRatio, NULL, parentType, childType);
}

ParticleHandle* TissueForge::ParticleHandle::split(
    const FVector3& childDirection,
    const FPTYPE& childRatio,
    const std::vector<FPTYPE>& speciesRatios,
    ParticleType* parentType,
    ParticleType* childType
) {
    TF_PARTICLE_SELF(this)
    std::vector<FPTYPE> _speciesRatios(speciesRatios);
    return Particle_split(self, childDirection, childRatio, &_speciesRatios, parentType, childType);
}

Particle* TissueForge::Particle_Get(ParticleHandle *pypart) {
    return _Engine.s.partlist[pypart->id];
}

ParticleHandle *TissueForge::Particle::handle() {
    
    if(!this->_handle) this->_handle = new ParticleHandle(this->id);
    
    return this->_handle;
}


HRESULT TissueForge::Particle::addpart(int32_t pid) {

    // only in clusters
    if (!_Engine.types[typeId].isCluster()) return error(MDCERR_notcluster);
    
    /* do we need to extend the partlist? */
    if(nr_parts == size_parts) {
        size_parts += CLUSTER_PARTLIST_INCR;
        int32_t* temp;
        if((temp = (int32_t*)malloc(sizeof(int32_t) * size_parts)) == NULL)
            return error(MDCERR_malloc);
        memcpy(temp, parts, sizeof(int32_t) * nr_parts);
        free(parts);
        parts = temp;
    }
    
    Particle *p = _Engine.s.partlist[pid];
    p->clusterId = this->id;
    
    parts[nr_parts] = pid;
    nr_parts++;
    return S_OK;
}

HRESULT TissueForge::Particle::removepart(int32_t pid) {
    
    int pid_index = -1;
    
    for(int i = 0; i < this->nr_parts; ++i) {
        if(this->particle(i)->id == pid) {
            pid_index = i;
            break;
        }
    }
    
    if(pid_index < 0) {
        return error(MDCERR_index);
    }
    
    Particle *p = _Engine.s.partlist[pid];
    p->clusterId = -1;
    
    for(int i = pid_index; i + 1 < this->nr_parts; ++i) {
        this->parts[i] = this->parts[i+1];
    }
    nr_parts--;
    
    return S_OK;
}

bool TissueForge::Particle::verify() {
    bool gte, lt;
    
    if(this->flags & PARTICLE_LARGE) {
        gte = x[0] >= 0 && x[1] >= 0 && x[2] >= 0;
        auto eng_dims = engine_dimensions();
        lt = x[0] <= eng_dims[0] && x[1] <= eng_dims[1] &&x[2] <= eng_dims[2];
    }
    else {
        gte = x[0] >= 0 && x[1] >= 0 && x[2] >= 0;
        // TODO, make less than
        lt = x[0] <= _Engine.s.h[0] && x[1] <= _Engine.s.h[1] &&x[2] <= _Engine.s.h[2];
    }
    
    bool pindex = this == _Engine.s.partlist[this->id];

    if(!gte || !lt || !pindex) {
        
        TF_Log(LOG_ERROR) << "Verify failed for particle " << this->id;
        TF_Log(LOG_ERROR) << "   Large particle   : " << (this->flags & PARTICLE_LARGE);
        TF_Log(LOG_ERROR) << "   Validated lower  : " << lt;
        TF_Log(LOG_ERROR) << "   Validated upper  : " << gte;
        TF_Log(LOG_ERROR) << "   Validated address: " << pindex;
        TF_Log(LOG_ERROR) << "   Global position  : " << this->global_position();
        TF_Log(LOG_ERROR) << "   Relative position: " << this->position;
        TF_Log(LOG_ERROR) << "   Particle type    : " << this->handle()->type()->name;
        TF_Log(LOG_ERROR) << "   Engine dims      : " << engine_dimensions();

        if(!(this->flags & PARTICLE_LARGE)) {
            space_cell *cell = _Engine.s.celllist[this->id];
            TF_Log(LOG_ERROR) << "   Cell dims        : " << FVector3::from(_Engine.s.h);
            TF_Log(LOG_ERROR) << "   Cell location    : " << iVector3::from(cell->loc);
            TF_Log(LOG_ERROR) << "   Cell origin      : " << FVector3::from(cell->origin);
        }
    }

    assert("particle pos below zero" && gte);
    assert("particle pos over cell size" && lt);
    assert("particle not in correct partlist location" && pindex);
    return gte && lt && pindex;
}

TissueForge::Particle::operator Cluster*() {
    ParticleType *type = &_Engine.types[typeId];
    if (!type->isCluster()) return NULL;
    return static_cast<Cluster*>(static_cast<void*>(this));
}

ParticleHandle* TissueForge::Particle_New(
    ParticleType *type, 
    FVector3 *position,
    FVector3 *velocity,
    int *clusterId) 
{
    
    if(!type) {
        error(MDCERR_null);
        return NULL;
    }
    
    // make a new pyparticle
    auto pyPart = new ParticleHandle();
    
    if(particle_init(pyPart, type, position, velocity, clusterId) != S_OK) {
        error(MDCERR_particle);
        return NULL;
    }
    
    return pyPart;
}

std::vector<int> TissueForge::Particles_New(
    std::vector<ParticleType*> types, 
    std::vector<FVector3> *positions, 
    std::vector<FVector3> *velocities, 
    std::vector<int> *clusterIds) 
{

    unsigned int nr_parts = types.size();

    if((positions && positions->size() != nr_parts) || (velocities && velocities->size() != nr_parts) || (clusterIds && clusterIds->size() != nr_parts)) {
        error(MDCERR_bad_el_input);
        return {};
    }

    if(_Engine.s.nr_parts + nr_parts > _Engine.s.size_parts) { 
        int size_incr = (int((_Engine.s.nr_parts - _Engine.s.size_parts + nr_parts) / space_partlist_incr) + 1) * space_partlist_incr;
        if(space_growparts(&_Engine.s, size_incr) != S_OK) { 
            error(MDCERR_space);
            return {};
        }
    }

    // initialize vector of particles to construct
    std::vector<Particle*> parts(nr_parts, 0);
    std::vector<ParticleHandle*> handles(nr_parts, 0);
    std::vector<int> result(nr_parts, -1);

    // construct particles
    for(int i = 0; i < nr_parts; i++) {
        parts[i] = new Particle();
        ParticleType *ptype = types[i];
        ParticleHandle *self = new ParticleHandle();
        FVector3 position = positions ? (*positions)[i] : particle_posdefault();
        FVector3 velocity = velocities ? (*velocities)[i] : particle_veldefault(*ptype);
        int clusterId = clusterIds ? (*clusterIds)[i] : -1;
        particle_ex_construct(self, ptype, position, velocity, clusterId, *parts[i]);
        handles[i] = self;
    }
    
    // load particles in engine
    particles_ex_load(handles, parts);

    // build results
    for(int i = 0; i < nr_parts; i++) {
        result[i] = handles[i]->id;
        delete parts[i];
    }

    return result;
}

std::vector<int> TissueForge::Particles_New(
    ParticleType *type, 
    unsigned int nr_parts, 
    std::vector<FVector3> *positions, 
    std::vector<FVector3> *velocities, 
    std::vector<int> *clusterIds) 
{
    if(nr_parts == 0) {
        if(positions) 
            nr_parts = positions->size();
        else if(velocities) 
            nr_parts = velocities->size();
        else if(clusterIds) 
            nr_parts = clusterIds->size();
        else {
            error(MDCERR_bad_el_input);
            return {};
        }
    }

    return Particles_New(std::vector<ParticleType*>(nr_parts, type), positions, velocities, clusterIds);
}


HRESULT TissueForge::Particle_Become(Particle *part, ParticleType *type) {
    int hr;
    if(!part || !type) {
        return error(MDCERR_null);
    }
    ParticleHandle *pypart = part->handle();
    
    ParticleType *currentType = &_Engine.types[part->typeId];
    
    assert(pypart->getTypeId() == currentType->id);
    
    if(!SUCCEEDED(hr = currentType->del_part(part->id))) {
        return error(MDCERR_particle);
    };
    
    if(!SUCCEEDED(hr = type->addpart(part->id))) {
        return error(MDCERR_particle);
    }
    
    part->typeId = type->id;
    
    part->flags = type->particle_flags;
    
    if(!part->style) {
        bool visible = type->style->flags & STYLE_VISIBLE;
        if(visible != (currentType->style->flags & STYLE_VISIBLE)) {
            if(part->flags & PARTICLE_LARGE) _Engine.s.nr_visible_large_parts += visible ? 1 : -1;
            else _Engine.s.nr_visible_parts += visible ? 1 : -1;
        }
    }
    
    if(part->state_vector) {
        state::StateVector *oldState = part->state_vector;
        
        if(type->species) {
            part->state_vector = new state::StateVector(type->species, part, oldState);
        }
        else {
            part->state_vector = NULL;
        }
        
    }
    
    assert(type == &_Engine.types[part->typeId]);
    
    // TODO: bad things will happen if we convert between cluster and atomic types.
    
    return S_OK;
}

HRESULT TissueForge::ParticleHandle::become(ParticleType *type) {
    TF_PARTICLE_SELFW(this, E_FAIL)
    return Particle_Become(self, type);
}

ParticleList TissueForge::ParticleHandle::neighbors(const FPTYPE &distance, const ParticleTypeList &types) {
    TF_PARTICLE_SELFW(this, ParticleList())

    std::set<short int> typeIds;
    for(int32_t i = 0; i < types.nr_parts; i++) 
        typeIds.insert(types.parts[i]);
    
    // take into account the radius of this particle.
    const FPTYPE radius = distance + self->radius;
    
    uint16_t nr_parts = 0;
    int32_t *parts = NULL;
    
    metrics::particleNeighbors(self, radius, &typeIds, &nr_parts, &parts);
    
    ParticleList result(nr_parts, parts);
    if(parts) std::free(parts);
    return result;
}

ParticleList TissueForge::ParticleHandle::neighbors(const FPTYPE &distance, const std::vector<ParticleType> &types) {
    ParticleTypeList typeList(types.size());
    for(auto &t : types) 
        typeList.insert(&t);
    return neighbors(distance, typeList);
}

ParticleList TissueForge::ParticleHandle::neighbors(const FPTYPE &distance) {
    TF_PARTICLE_SELFW(this, ParticleList())

    ParticleTypeList typeList;
    for(int i = 0; i < _Engine.nr_types; ++i) typeList.insert(_Engine.types[i].id);
    return neighbors(distance, typeList);
}

ParticleList TissueForge::ParticleHandle::neighbors(const ParticleTypeList &types) {
    return neighbors(_Engine.s.cutoff, types);
}

ParticleList TissueForge::ParticleHandle::neighbors(const std::vector<ParticleType> &types) {
    return neighbors(_Engine.s.cutoff, types);
}

std::vector<int32_t> TissueForge::ParticleHandle::neighborIds(const FPTYPE &distance, const ParticleTypeList &types) {
    TF_PARTICLE_SELFW(this, {})
    
    std::set<short int> typeIds;
    for(int32_t i = 0; i < types.nr_parts; i++) 
        typeIds.insert(types.parts[i]);
    
    // take into account the radius of this particle.
    const FPTYPE radius = distance + self->radius;
    
    uint16_t nr_parts = 0;
    int32_t *parts = NULL;
    
    metrics::particleNeighbors(self, radius, &typeIds, &nr_parts, &parts);
    
    std::vector<int32_t> result;
    result.reserve(nr_parts);
    for(int i = 0; i < nr_parts; i++) result.push_back(parts[i]);

    if(parts) std::free(parts);
    return result;
}

std::vector<int32_t> TissueForge::ParticleHandle::neighborIds(const FPTYPE &distance, const std::vector<ParticleType> &types) {
    ParticleTypeList typeList(types.size());
    for(auto &t : types) 
        typeList.insert(&t);
    return neighborIds(distance, typeList);
}

std::vector<int32_t> TissueForge::ParticleHandle::neighborIds(const FPTYPE &distance) {
    TF_PARTICLE_SELFW(this, {})

    const FPTYPE _distance = distance > 0 ? distance : _Engine.s.cutoff;

    ParticleTypeList typeList;
    for(int i = 0; i < _Engine.nr_types; ++i) typeList.insert(_Engine.types[i].id);
    return neighborIds(_distance, typeList);
}

std::vector<int32_t> TissueForge::ParticleHandle::neighborIds(const ParticleTypeList &types) {
    return neighborIds(_Engine.s.cutoff, types);
}

std::vector<int32_t> TissueForge::ParticleHandle::neighborIds(const std::vector<ParticleType> &types) {
    return neighborIds(_Engine.s.cutoff, types);
}

ParticleList TissueForge::ParticleHandle::getBondedNeighbors() {
    TF_PARTICLE_SELFW(this, ParticleList())

    auto id = self->id;
    
    ParticleList list;
    
    for(int i = 0; i < _Engine.nr_bonds; ++i) {
        Bond *b = &_Engine.bonds[i];
        if(b->flags & BOND_ACTIVE) {
            if(b->i == id) {
                list.insert(b->j);
            }
            else if(b->j == id) {
                list.insert(b->i);
            }
        }
    }
    return list;
}

std::vector<int32_t> TissueForge::ParticleHandle::getBondedNeighborIds() {
    TF_PARTICLE_SELFW(this, {})

    auto id = self->id;
    
    std::vector<int32_t> list;
    
    for(int i = 0; i < _Engine.nr_bonds; ++i) {
        Bond *b = &_Engine.bonds[i];
        if(b->flags & BOND_ACTIVE) {
            if(b->i == id) {
                list.push_back(b->j);
            }
            else if(b->j == id) {
                list.push_back(b->i);
            }
        }
    }
    return list;
}

std::vector<BondHandle> TissueForge::ParticleHandle::getBonds() {
    
    std::vector<BondHandle> bonds;

    TF_PARTICLE_SELFW(this, bonds)

    auto id = self->id;
    bonds.reserve(_Engine.nr_active_bonds);
    
    for(int i = 0; i < _Engine.bonds_size; ++i) {
        Bond *b = &_Engine.bonds[i];
        if((b->flags & BOND_ACTIVE) && (b->i == id || b->j == id)) {
            bonds.push_back(BondHandle(i));
        }
    }

    return bonds;
}

std::vector<AngleHandle> TissueForge::ParticleHandle::getAngles() {
    
    std::vector<AngleHandle> angles;

    TF_PARTICLE_SELFW(this, angles)

    auto id = self->id;
    angles.reserve(_Engine.nr_active_angles);
    
    for(int i = 0; i < _Engine.angles_size; ++i) {
        Angle *a = &_Engine.angles[i];
        if((a->flags & ANGLE_ACTIVE) && (a->i == id || a->j == id || a->k == id)) {
            angles.push_back(AngleHandle(i));
        }
    }

    return angles;
}

std::vector<DihedralHandle> TissueForge::ParticleHandle::getDihedrals() {
    
    std::vector<DihedralHandle> dihedrals;

    TF_PARTICLE_SELFW(this, dihedrals)

    auto id = self->id;
    dihedrals.reserve(_Engine.nr_active_dihedrals);
    
    for(int i = 0; i < _Engine.dihedrals_size; ++i) {
        Dihedral *d = &_Engine.dihedrals[i];
        if((d->flags & DIHEDRAL_ACTIVE) && (d->i == id || d->j == id || d->k == id || d->l == id)) {
            dihedrals.push_back(DihedralHandle(i));
        }
    }

    return dihedrals;
}

static ParticleList *particletype_items(ParticleType *self) {
    return &self->parts;
}

ParticleList &TissueForge::ParticleType::items() {
    return parts;
}

uint16_t TissueForge::ParticleType::getNumParts() {
    return this->items().nr_parts;
}

std::vector<int32_t> TissueForge::ParticleType::getPartIds() {
    return this->items().vector();
}

FPTYPE TissueForge::ParticleHandle::distance(ParticleHandle *_other) {
    TF_PARTICLE_SELFW(this, FPTYPE_ZERO)
    auto other = particleSelf(_other);
    
    if(other == NULL) 
        return error(MDCERR_null);
    
    FVector3 pos = self->global_position();
    FVector3 opos = other->global_position();
    return (opos - pos).length();
}

HRESULT particle_init(ParticleHandle *self, ParticleType *ptype, FVector3 *position, FVector3 *velocity, int *cluster) 
{
    
    TF_Log(LOG_TRACE);

    FVector3 _position = position ? FVector3(*position) : particle_posdefault();
    FVector3 _velocity = velocity ? FVector3(*velocity) : particle_veldefault(*ptype);
    
    // particle_init_ex will allocate a new particle, this can re-assign the pointers in
    // the engine particles, so need to pass cluster by id.
    int _clusterId = cluster ? *cluster : -1;
    
    return particle_init_ex(self, ptype, _position, _velocity, _clusterId);
}

HRESULT particle_ex_construct(
    ParticleHandle *self, 
    ParticleType *ptype, 
    const FVector3 &position, 
    const FVector3 &velocity, 
    int clusterId, 
    Particle &part) 
{
    bzero(&part, sizeof(Particle));
    part.radius = ptype->radius;
    part.mass = ptype->mass;
    part.imass = ptype->imass;
    part.typeId = ptype->id;
    part.flags = ptype->particle_flags;
    part.creation_time = _Engine.time;
    part.clusterId = clusterId;
    
    if(ptype->isCluster()) {
        TF_Log(LOG_DEBUG) << "making cluster";
    }
    
    if(ptype->species) {
        part.state_vector = new state::StateVector(ptype->species, self);
    }
    
    part.position = position;
    part.velocity = velocity;
    
    if(part.radius > _Engine.s.cutoff) {
        part.flags |= PARTICLE_LARGE;
    }
    
    return S_OK;
}

HRESULT particle_ex_load(ParticleHandle *self, Particle &part) {
    part.id = engine_next_partid(&_Engine);
    
    Particle *p = NULL;
    FPTYPE pos[] = {part.position[0], part.position[1], part.position[2]};
    
    if(engine_addpart(&_Engine, &part, pos, &p) != S_OK) 
        return error(MDCERR_engine);
    
    self->id = p->id;
    
    if(part.clusterId >= 0) {
        Particle *cluster = _Engine.s.partlist[part.clusterId];
        Cluster_AddParticle((Cluster*)cluster, p);
    } else {
        p->clusterId = -1;
    }
    
    p->_handle = self;

    return S_OK;
}

HRESULT particles_ex_load(std::vector<ParticleHandle*> selfs, std::vector<Particle*> parts) {
    if(selfs.size() != parts.size()) 
        return error(MDCERR_bad_el_input);

    FPTYPE **positions = (FPTYPE**)malloc(sizeof(FPTYPE*) * parts.size());
    std::vector<int> part_ids(parts.size(), 0);
    if(engine_next_partids(&_Engine, part_ids.size(), part_ids.data()) != S_OK) 
        return error(MDCERR_engine);

    for(int i = 0; i < parts.size(); i++) {
        Particle *part = parts[i];
        part->id = part_ids[i];
        positions[i] = (FPTYPE*)malloc(sizeof(FPTYPE) * 3);
        for(int k = 0; k < 3; k++) positions[i][k] = part->position[k];
    }
    
    if(engine_addparts(&_Engine, parts.size(), parts.data(), positions) != S_OK) 
        return error(MDCERR_engine);
    
    for(int i = 0; i < selfs.size(); i++) {
        Particle *part = parts[i];
        ParticleHandle *self = selfs[i];
        
        self->id = part_ids[i];
        Particle *p = self->part();
        
        if(part->clusterId >= 0) {
            Particle *cluster = _Engine.s.partlist[part->clusterId];
            Cluster_AddParticle((Cluster*)cluster, p);
        } else {
            p->clusterId = -1;
        }
        
        p->_handle = self;

        free(positions[i]);
    }

    free(positions);

    return S_OK;
}

HRESULT particle_init_ex(
    ParticleHandle *self, 
    ParticleType *ptype, 
    const FVector3 &position,
    const FVector3 &velocity, 
    int clusterId) 
{
    
    Particle part;
    int result;

    if((result = particle_ex_construct(self, ptype, position, velocity, clusterId, part)) != S_OK) {
        return error(MDCERR_particle);
    }
    if((result = particle_ex_load(self, part)) != S_OK) {
        return error(MDCERR_particle);
    }
    
    return S_OK;
}

ParticleType* TissueForge::ParticleType_FindFromName(const char* name) {
    for(int i = 0; i < _Engine.nr_types; ++i) {
        ParticleType *type = &_Engine.types[i];
        if(std::strncmp(name, type->name, sizeof(ParticleType::name)) == 0) {
            return type;
        }
    }
    return NULL;
}


HRESULT TissueForge::Particle_Verify() {

    bool result = true;

    for (int cid = 0 ; cid < _Engine.s.nr_cells ; cid++) {
        space_cell *cell = &_Engine.s.cells[cid];
        for (int pid = 0 ; pid < cell->count ; pid++) {
            Particle *p  = &cell->parts[pid];
            result = p->verify() && result;
        }
    }

    for (int pid = 0 ; pid < _Engine.s.largeparts.count ; pid++) {
        Particle *p  = &_Engine.s.largeparts.parts[pid];
        result = p->verify() && result;
    }

    return result ? S_OK : error(MDCERR_verify);
}


namespace TissueForge::io { 


    template <>
    HRESULT toFile(const Particle &dataElement, const MetaData &metaData, IOElement &fileElement) { 

        TF_IOTOEASY(fileElement, metaData, "force", dataElement.force);
        TF_IOTOEASY(fileElement, metaData, "force_i", dataElement.force_i);
        TF_IOTOEASY(fileElement, metaData, "number_density", dataElement.number_density);
        TF_IOTOEASY(fileElement, metaData, "velocity", dataElement.velocity);
        TF_IOTOEASY(fileElement, metaData, "position", ParticleHandle(dataElement.id).getPosition());
        TF_IOTOEASY(fileElement, metaData, "creation_time", dataElement.creation_time);
        TF_IOTOEASY(fileElement, metaData, "persistent_force", dataElement.persistent_force);
        TF_IOTOEASY(fileElement, metaData, "radius", dataElement.radius);
        TF_IOTOEASY(fileElement, metaData, "mass", dataElement.mass);
        TF_IOTOEASY(fileElement, metaData, "q", dataElement.q);
        TF_IOTOEASY(fileElement, metaData, "p0", dataElement.p0);
        TF_IOTOEASY(fileElement, metaData, "v0", dataElement.v0);
        TF_IOTOEASY(fileElement, metaData, "xk0", dataElement.xk[0]);
        TF_IOTOEASY(fileElement, metaData, "xk1", dataElement.xk[1]);
        TF_IOTOEASY(fileElement, metaData, "xk2", dataElement.xk[2]);
        TF_IOTOEASY(fileElement, metaData, "xk3", dataElement.xk[3]);
        TF_IOTOEASY(fileElement, metaData, "vk0", dataElement.vk[0]);
        TF_IOTOEASY(fileElement, metaData, "vk1", dataElement.vk[1]);
        TF_IOTOEASY(fileElement, metaData, "vk2", dataElement.vk[2]);
        TF_IOTOEASY(fileElement, metaData, "vk3", dataElement.vk[3]);
        TF_IOTOEASY(fileElement, metaData, "id", dataElement.id);
        TF_IOTOEASY(fileElement, metaData, "vid", dataElement.vid);
        TF_IOTOEASY(fileElement, metaData, "typeId", dataElement.typeId);
        TF_IOTOEASY(fileElement, metaData, "clusterId", dataElement.clusterId);
        TF_IOTOEASY(fileElement, metaData, "flags", dataElement.flags);
        
        if(dataElement.nr_parts > 0) {
            std::vector<int32_t> parts;
            for(unsigned int i = 0; i < dataElement.nr_parts; i++) 
                parts.push_back(dataElement.parts[i]);
            TF_IOTOEASY(fileElement, metaData, "parts", parts);
        }
        
        if(dataElement.style != NULL) {
            TF_IOTOEASY(fileElement, metaData, "style", *dataElement.style);
        }
        if(dataElement.state_vector != NULL) {
            TF_IOTOEASY(fileElement, metaData, "state_vector", *dataElement.state_vector);
        }

        fileElement.get()->type = "Particle";

        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, Particle *dataElement) { 

        TF_IOFROMEASY(fileElement, metaData, "force", &dataElement->force);
        if(metaData.versionMajor > 0 || metaData.versionMinor > 31 || (metaData.versionMinor == 31 && metaData.versionPatch > 0)) 
            TF_IOFROMEASY(fileElement, metaData, "force_i", &dataElement->force_i);
        TF_IOFROMEASY(fileElement, metaData, "velocity", &dataElement->velocity);
        TF_IOFROMEASY(fileElement, metaData, "number_density", &dataElement->number_density);
        dataElement->inv_number_density = 1.f / dataElement->number_density;
        TF_IOFROMEASY(fileElement, metaData, "position", &dataElement->position);
        TF_IOFROMEASY(fileElement, metaData, "creation_time", &dataElement->creation_time);
        TF_IOFROMEASY(fileElement, metaData, "persistent_force", &dataElement->persistent_force);
        TF_IOFROMEASY(fileElement, metaData, "radius", &dataElement->radius);
        TF_IOFROMEASY(fileElement, metaData, "mass", &dataElement->mass);
        dataElement->imass = dataElement->mass > 0 ? 1.f / dataElement->mass : 0;
        TF_IOFROMEASY(fileElement, metaData, "q", &dataElement->q);
        TF_IOFROMEASY(fileElement, metaData, "p0", &dataElement->p0);
        TF_IOFROMEASY(fileElement, metaData, "v0", &dataElement->v0);
        TF_IOFROMEASY(fileElement, metaData, "xk0", &dataElement->xk[0]);
        TF_IOFROMEASY(fileElement, metaData, "xk1", &dataElement->xk[1]);
        TF_IOFROMEASY(fileElement, metaData, "xk2", &dataElement->xk[2]);
        TF_IOFROMEASY(fileElement, metaData, "xk3", &dataElement->xk[3]);
        TF_IOFROMEASY(fileElement, metaData, "vk0", &dataElement->vk[0]);
        TF_IOFROMEASY(fileElement, metaData, "vk1", &dataElement->vk[1]);
        TF_IOFROMEASY(fileElement, metaData, "vk2", &dataElement->vk[2]);
        TF_IOFROMEASY(fileElement, metaData, "vk3", &dataElement->vk[3]);
        TF_IOFROMEASY(fileElement, metaData, "id", &dataElement->id);
        TF_IOFROMEASY(fileElement, metaData, "vid", &dataElement->vid);
        TF_IOFROMEASY(fileElement, metaData, "typeId", &dataElement->typeId);
        TF_IOFROMEASY(fileElement, metaData, "clusterId", &dataElement->clusterId);
        TF_IOFROMEASY(fileElement, metaData, "flags", &dataElement->flags);
        
        // Skipping importing constituent particles; deduced from clusterId during import

        IOChildMap fec = IOElement::children(fileElement);
        
        IOChildMap::const_iterator feItr = fec.find("style");
        if(feItr != fec.end()) {
            dataElement->style = new rendering::Style();
            if(fromFile(feItr->second, metaData, dataElement->style) != S_OK) 
                return error(MDCERR_io);
        } 
        else dataElement->style = NULL;
        
        feItr = fec.find("state_vector");
        if(feItr != fec.end()) {
            dataElement->state_vector = NULL;
            if(fromFile(feItr->second, metaData, &dataElement->state_vector) != S_OK) 
                return error(MDCERR_io);
            dataElement->state_vector->owner = dataElement;
        }
        else dataElement->state_vector = NULL;

        return S_OK;
    }

    template <>
    HRESULT toFile(const ParticleType &dataElement, const MetaData &metaData, IOElement &fileElement) { 

        TF_IOTOEASY(fileElement, metaData, "id", dataElement.id);
        TF_IOTOEASY(fileElement, metaData, "type_flags", dataElement.type_flags);
        TF_IOTOEASY(fileElement, metaData, "particle_flags", dataElement.particle_flags);
        TF_IOTOEASY(fileElement, metaData, "mass", dataElement.mass);
        TF_IOTOEASY(fileElement, metaData, "charge", dataElement.charge);
        TF_IOTOEASY(fileElement, metaData, "radius", dataElement.radius);
        TF_IOTOEASY(fileElement, metaData, "kinetic_energy", dataElement.kinetic_energy);
        TF_IOTOEASY(fileElement, metaData, "potential_energy", dataElement.potential_energy);
        TF_IOTOEASY(fileElement, metaData, "target_energy", dataElement.target_energy);
        TF_IOTOEASY(fileElement, metaData, "minimum_radius", dataElement.minimum_radius);
        TF_IOTOEASY(fileElement, metaData, "eps", dataElement.eps);
        TF_IOTOEASY(fileElement, metaData, "rmin", dataElement.rmin);
        TF_IOTOEASY(fileElement, metaData, "dynamics", (unsigned int)dataElement.dynamics);
        TF_IOTOEASY(fileElement, metaData, "name", std::string(dataElement.name));
        TF_IOTOEASY(fileElement, metaData, "name2", std::string(dataElement.name2));
        if(dataElement.parts.nr_parts > 0) 
            TF_IOTOEASY(fileElement, metaData, "parts", dataElement.parts);
        if(dataElement.types.nr_parts > 0) 
            TF_IOTOEASY(fileElement, metaData, "types", dataElement.types);
        if(dataElement.style != NULL) {
            TF_IOTOEASY(fileElement, metaData, "style", *dataElement.style);
        }
        if(dataElement.species != NULL) {
            TF_IOTOEASY(fileElement, metaData, "species", *dataElement.species);
        }

        fileElement.get()->type = "ParticleType";

        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, ParticleType *dataElement) { 

        IOChildMap::const_iterator feItr;

        // Id set during registration: type ids are not preserved during import
        // TF_IOFROMEASY(fileElement, metaData, "id", &dataElement->id);

        TF_IOFROMEASY(fileElement, metaData, "type_flags", &dataElement->type_flags);
        TF_IOFROMEASY(fileElement, metaData, "particle_flags", &dataElement->particle_flags);
        TF_IOFROMEASY(fileElement, metaData, "mass", &dataElement->mass);
        dataElement->imass = 1.f / dataElement->mass;
        TF_IOFROMEASY(fileElement, metaData, "charge", &dataElement->charge);
        TF_IOFROMEASY(fileElement, metaData, "radius", &dataElement->radius);
        TF_IOFROMEASY(fileElement, metaData, "kinetic_energy", &dataElement->kinetic_energy);
        TF_IOFROMEASY(fileElement, metaData, "potential_energy", &dataElement->potential_energy);
        TF_IOFROMEASY(fileElement, metaData, "target_energy", &dataElement->target_energy);
        TF_IOFROMEASY(fileElement, metaData, "minimum_radius", &dataElement->minimum_radius);
        TF_IOFROMEASY(fileElement, metaData, "eps", &dataElement->eps);
        TF_IOFROMEASY(fileElement, metaData, "rmin", &dataElement->rmin);
        
        unsigned int dynamics;
        TF_IOFROMEASY(fileElement, metaData, "dynamics", &dynamics);
        dataElement->dynamics = dynamics;
        
        std::string name;
        TF_IOFROMEASY(fileElement, metaData, "name", &name);
        std::strncpy(dataElement->name, std::string(name).c_str(), ParticleType::MAX_NAME);
        
        std::string name2;
        TF_IOFROMEASY(fileElement, metaData, "name2", &name2);
        std::strncpy(dataElement->name2, std::string(name2).c_str(), ParticleType::MAX_NAME);
        
        // Parts must be manually added, since part ids are not preserved during import
        // TF_IOFROMEASY(fileElement, metaData, "parts", &dataElement->parts);

        IOChildMap fec = IOElement::children(fileElement);
        
        if(fec.find("types") != fec.end()) 
            TF_IOFROMEASY(fileElement, metaData, "types", &dataElement->types);
        
        feItr = fec.find("style");
        if(feItr != fec.end()) { 
            dataElement->style = new rendering::Style();
            if(fromFile(feItr->second, metaData, dataElement->style) != S_OK) 
                return error(MDCERR_io);
        } 
        else {
            fVector3 c = Magnum::Color3::fromSrgb(colors[(dataElement->id - 1) % (sizeof(colors)/sizeof(unsigned))]);
            dataElement->style = new rendering::Style(&c);
        }
        
        feItr = fec.find("species");
        if(feItr != fec.end()) {
            dataElement->species = new state::SpeciesList();
            if(fromFile(feItr->second, metaData, dataElement->species) != S_OK) 
                return error(MDCERR_io);
        } 
        else dataElement->species = NULL;

        return S_OK;
    }

};

std::string TissueForge::Particle::toString() {
    return io::toString(*this);
}

Particle *TissueForge::Particle::fromString(const std::string &str) {
    return new Particle(io::fromString<Particle>(str));
}

std::string TissueForge::ParticleType::toString() {
    return io::toString(*this);
}

ParticleType *TissueForge::ParticleType::fromString(const std::string &str) {
    return new ParticleType(io::fromString<ParticleType>(str));
}
