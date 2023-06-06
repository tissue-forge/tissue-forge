/*******************************************************************************
 * This file is part of Tissue Forge.
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

#include "tfColorMapper.h"
#include <tfParticle.h>
#include <tfAngle.h>
#include <tfBond.h>
#include <tfDihedral.h>
#include <tfLogger.h>
#include <state/tfSpeciesList.h>
#include <tfError.h>
#include <tfSimulator.h>

#include "tfColorMaps.h"


using namespace TissueForge;


#define REGULARIZE_SCALAR(s, smin, smax) (s - smin) / (smax - smin)

enum MappedParticleData {
    MappedParticleData_NONE = 0,
    ParticlePositionX,
    ParticlePositionY,
    ParticlePositionZ,
    ParticleVelocityX,
    ParticleVelocityY,
    ParticleVelocityZ,
    ParticleSpeed,
    ParticleForceX,
    ParticleForceY,
    ParticleForceZ,
    ParticleSpecies
};

enum MappedAngleData {
    MappedAngleData_NONE = 0,
    AngleAngle,
    AngleAngleEq
};

enum MappedBondData {
    MappedBondData_NONE = 0,
    BondLength,
    BondLengthEq
};

enum MappedDihedralData {
    MappedDihedralData_NONE = 0,
    DihedralAngle,
    DihedralAngleEq
};

static float map_particle_pos_x(Particle* p, struct rendering::ColorMapper* mapper) {
    return REGULARIZE_SCALAR(p->global_position()[0], mapper->min_val, mapper->max_val);
}

static float map_particle_pos_y(Particle* p, struct rendering::ColorMapper* mapper) {
    return REGULARIZE_SCALAR(p->global_position()[1], mapper->min_val, mapper->max_val);
}

static float map_particle_pos_z(Particle* p, struct rendering::ColorMapper* mapper) {
    return REGULARIZE_SCALAR(p->global_position()[2], mapper->min_val, mapper->max_val);
}

static float map_particle_vel_x(Particle* p, struct rendering::ColorMapper* mapper) {
    return REGULARIZE_SCALAR(p->v[0], mapper->min_val, mapper->max_val);
}

static float map_particle_vel_y(Particle* p, struct rendering::ColorMapper* mapper) {
    return REGULARIZE_SCALAR(p->v[1], mapper->min_val, mapper->max_val);
}

static float map_particle_vel_z(Particle* p, struct rendering::ColorMapper* mapper) {
    return REGULARIZE_SCALAR(p->v[2], mapper->min_val, mapper->max_val);
}

static float map_particle_speed(Particle* p, struct rendering::ColorMapper* mapper) {
    return REGULARIZE_SCALAR(ParticleHandle(p->id).getVelocity().length(), mapper->min_val, mapper->max_val);
}

static float map_particle_force_x(Particle* p, struct rendering::ColorMapper* mapper) {
    return REGULARIZE_SCALAR(p->f[0], mapper->min_val, mapper->max_val);
}

static float map_particle_force_y(Particle* p, struct rendering::ColorMapper* mapper) {
    return REGULARIZE_SCALAR(p->f[1], mapper->min_val, mapper->max_val);
}

static float map_particle_force_z(Particle* p, struct rendering::ColorMapper* mapper) {
    return REGULARIZE_SCALAR(p->f[2], mapper->min_val, mapper->max_val);
}

static float map_particle_species(Particle* p, struct rendering::ColorMapper* mapper) {
    if(!p->state_vector) return FPTYPE_ZERO;
    return REGULARIZE_SCALAR(p->state_vector->fvec[mapper->species_index], mapper->min_val, mapper->max_val);
}

static float map_angle_angle(Angle* a, struct rendering::ColorMapper* mapper) {
    return REGULARIZE_SCALAR(AngleHandle(a->id).getAngle(), mapper->min_val, mapper->max_val);
}

static float map_angle_angleeq(Angle* a, struct rendering::ColorMapper* mapper) {
    return REGULARIZE_SCALAR(std::abs(AngleHandle(a->id).getAngle() - a->potential->r0_plusone + 1), mapper->min_val, mapper->max_val);
}

static float map_bond_length(Bond* b, struct rendering::ColorMapper* mapper) {
    return REGULARIZE_SCALAR(BondHandle(b->id).getLength(), mapper->min_val, mapper->max_val);
}

static float map_bond_lengtheq(Bond* b, struct rendering::ColorMapper* mapper) {
    return REGULARIZE_SCALAR(std::abs(BondHandle(b->id).getLength() - b->potential->r0_plusone + 1), mapper->min_val, mapper->max_val);
}

static float map_dihedral_angle(Dihedral* d, struct rendering::ColorMapper* mapper) {
    return REGULARIZE_SCALAR(DihedralHandle(d->id).getAngle(), mapper->min_val, mapper->max_val);
}

static float map_dihedral_angleeq(Dihedral* d, struct rendering::ColorMapper* mapper) {
    return REGULARIZE_SCALAR(std::abs(DihedralHandle(d->id).getAngle() - d->potential->r0_plusone + 1), mapper->min_val, mapper->max_val);
}


bool rendering::ColorMapper::set_colormap(const std::string& s) {
    auto func = getColorMapperFunc(s);
    
    if(func) {
        this->map = func;
        
        Simulator::get()->redraw();
        
        return true;
    }
    return false;
}

rendering::ColorMapper::ColorMapper(const std::string &name, const float &vmin, const float &vmax) : 
    species_index{-1}, 
    min_val{vmin}, 
    max_val{vmax}, 
    map{NULL}, 
    mapper_angle{NULL}, 
    mapper_bond{NULL}, 
    mapper_dihedral{NULL}, 
    mapper_particle{NULL}, 
    map_enum_angle{MappedAngleData::MappedAngleData_NONE}, 
    map_enum_bond{MappedBondData::MappedBondData_NONE}, 
    map_enum_dihedral{MappedDihedralData::MappedDihedralData_NONE}, 
    map_enum_particle{MappedParticleData::MappedParticleData_NONE}
{
    map = getColorMapperFunc(name);
}

rendering::ColorMapper::ColorMapper() : 
    ColorMapper("Rainbow")
{}

fVector4 rendering::ColorMapper::mapScalar(const float& val) const {
    return (*this->map)(const_cast<ColorMapper*>(this), val);
}

fVector4 rendering::ColorMapper::mapObj(Particle* o) {
    return mapScalar(hasMapParticle() ? mapper_particle(o, this) : FPTYPE_ZERO);
}

fVector4 rendering::ColorMapper::mapObj(Angle* o) {
    return mapScalar(hasMapAngle() ? mapper_angle(o, this) : FPTYPE_ZERO);
}

fVector4 rendering::ColorMapper::mapObj(Bond* o) {
    return mapScalar(hasMapBond() ? mapper_bond(o, this) : FPTYPE_ZERO);
}

fVector4 rendering::ColorMapper::mapObj(Dihedral* o) {
    return mapScalar(hasMapDihedral() ? mapper_dihedral(o, this) : FPTYPE_ZERO);
}

const bool rendering::ColorMapper::hasMapParticle() const {
    return this->mapper_particle;
}

const bool rendering::ColorMapper::hasMapAngle() const {
    return this->mapper_angle;
}

const bool rendering::ColorMapper::hasMapBond() const {
    return this->mapper_bond;
}

const bool rendering::ColorMapper::hasMapDihedral() const {
    return this->mapper_dihedral;
}

void rendering::ColorMapper::clearMapParticle() {
    this->species_index = -1;
    this->mapper_particle = NULL;
    map_enum_particle = MappedParticleData::MappedParticleData_NONE;
}

void rendering::ColorMapper::clearMapAngle() {
    this->mapper_angle = NULL;
    map_enum_angle = MappedAngleData::MappedAngleData_NONE;
}

void rendering::ColorMapper::clearMapBond() {
    this->mapper_bond = NULL;
    map_enum_bond = MappedBondData::MappedBondData_NONE;
}

void rendering::ColorMapper::clearMapDihedral() {
    this->mapper_dihedral = NULL;
    map_enum_dihedral = MappedDihedralData::MappedDihedralData_NONE;
}

void rendering::ColorMapper::setMapParticle(const unsigned int& label) {
    this->clearMapParticle();
    switch (label) {
        case MappedParticleData::ParticlePositionX:
            this->mapper_particle = map_particle_pos_x;
            map_enum_particle = label;
            break;

        case MappedParticleData::ParticlePositionY:
            this->mapper_particle = map_particle_pos_y;
            map_enum_particle = label;
            break;

        case MappedParticleData::ParticlePositionZ:
            this->mapper_particle = map_particle_pos_z;
            map_enum_particle = label;
            break;

        case MappedParticleData::ParticleVelocityX:
            this->mapper_particle = map_particle_vel_x;
            map_enum_particle = label;
            break;

        case MappedParticleData::ParticleVelocityY:
            this->mapper_particle = map_particle_vel_y;
            map_enum_particle = label;
            break;

        case MappedParticleData::ParticleVelocityZ:
            this->mapper_particle = map_particle_vel_z;
            map_enum_particle = label;
            break;

        case MappedParticleData::ParticleSpeed:
            this->mapper_particle = map_particle_speed;
            map_enum_particle = label;
            break;

        case MappedParticleData::ParticleForceX:
            this->mapper_particle = map_particle_force_x;
            map_enum_particle = label;
            break;

        case MappedParticleData::ParticleForceY:
            this->mapper_particle = map_particle_force_y;
            map_enum_particle = label;
            break;

        case MappedParticleData::ParticleForceZ:
            this->mapper_particle = map_particle_force_z;
            map_enum_particle = label;
            break;

        case MappedParticleData::ParticleSpecies:
            this->mapper_particle = map_particle_species;
            map_enum_particle = label;

        default:
            break;
    }
}

void rendering::ColorMapper::setMapAngle(const unsigned int& label) {
    clearMapAngle();
    switch (label) {
        case MappedAngleData::AngleAngle:
            this->mapper_angle = map_angle_angle;
            map_enum_angle = label;
            break;
        
        case MappedAngleData::AngleAngleEq:
            this->mapper_angle = map_angle_angleeq;
            map_enum_angle = label;
            break;
        
        default:
            break;
    }
}

void rendering::ColorMapper::setMapBond(const unsigned int& label) {
    clearMapBond();
    switch (label) {
        case MappedBondData::BondLength:
            this->mapper_bond = map_bond_length;
            map_enum_bond = label;
            break;

        case MappedBondData::BondLengthEq:
            this->mapper_bond = map_bond_lengtheq;
            map_enum_bond = label;
            break;

        default:
            break;
    }
}

void rendering::ColorMapper::setMapDihedral(const unsigned int& label) {
    clearMapDihedral();
    switch (label) {
        case MappedDihedralData::DihedralAngle:
            this->mapper_dihedral = map_dihedral_angle;
            map_enum_dihedral = label;
            break;

        case MappedDihedralData::DihedralAngleEq:
            this->mapper_dihedral = map_dihedral_angleeq;
            map_enum_dihedral = label;
            break;

        default:
            break;
    }
}

void rendering::ColorMapper::setMapParticlePositionX()  { setMapParticle(MappedParticleData::ParticlePositionX); }
void rendering::ColorMapper::setMapParticlePositionY()  { setMapParticle(MappedParticleData::ParticlePositionY); }
void rendering::ColorMapper::setMapParticlePositionZ()  { setMapParticle(MappedParticleData::ParticlePositionZ); }
void rendering::ColorMapper::setMapParticleVelocityX()  { setMapParticle(MappedParticleData::ParticleVelocityX); }
void rendering::ColorMapper::setMapParticleVelocityY()  { setMapParticle(MappedParticleData::ParticleVelocityY); }
void rendering::ColorMapper::setMapParticleVelocityZ()  { setMapParticle(MappedParticleData::ParticleVelocityZ); }
void rendering::ColorMapper::setMapParticleSpeed()      { setMapParticle(MappedParticleData::ParticleSpeed); }
void rendering::ColorMapper::setMapParticleForceX()     { setMapParticle(MappedParticleData::ParticleForceX); }
void rendering::ColorMapper::setMapParticleForceY()     { setMapParticle(MappedParticleData::ParticleForceY); }
void rendering::ColorMapper::setMapParticleForceZ()     { setMapParticle(MappedParticleData::ParticleForceZ); }

void rendering::ColorMapper::setMapParticleSpecies(ParticleType* pType, const std::string& name) {

    if(pType->species == NULL) {
        std::string msg = "can not create color map for particle type \"";
        msg += pType->name;
        msg += "\" without any species defined";
        tf_exp(std::invalid_argument(msg));
        return;
    }
    
    int index = pType->species->index_of(name);
    TF_Log(LOG_DEBUG) << "Got species index: " << index;
    
    if(index < 0) {
        std::string msg = "can not create color map for particle type \"";
        msg += pType->name;
        msg += "\", does not contain species \"";
        msg += name;
        msg += "\"";
        tf_exp(std::invalid_argument(msg));
        return;
    }

    setMapParticle(MappedParticleData::ParticleSpecies);
    this->species_index = index;
}

void rendering::ColorMapper::setMapAngleAngle()         { setMapAngle(MappedAngleData::AngleAngle); }
void rendering::ColorMapper::setMapAngleAngleEq()       { setMapAngle(MappedAngleData::AngleAngleEq); }

void rendering::ColorMapper::setMapBondLength()         { setMapBond(MappedBondData::BondLength); }
void rendering::ColorMapper::setMapBondLengthEq()       { setMapBond(MappedBondData::BondLengthEq); }

void rendering::ColorMapper::setMapDihedralAngle()      { setMapDihedral(MappedDihedralData::DihedralAngle); }
void rendering::ColorMapper::setMapDihedralAngleEq()    { setMapDihedral(MappedDihedralData::DihedralAngleEq); }

std::vector<std::string> rendering::ColorMapper::getNames() {
    return getColorMapperFuncNames();
}

std::string rendering::ColorMapper::getColorMapName() const {
    for(auto& name : getColorMapperFuncNames()) 
        if(getColorMapperFunc(name) == this->map) 
            return name;
    return "";
}


namespace TissueForge::io {


    template <>
    HRESULT toFile(const rendering::ColorMapper &dataElement, const MetaData &metaData, IOElement &fileElement) {

        TF_IOTOEASY(fileElement, metaData, "species_index", dataElement.species_index);
        TF_IOTOEASY(fileElement, metaData, "min_val", dataElement.min_val);
        TF_IOTOEASY(fileElement, metaData, "max_val", dataElement.max_val);
        
        std::string cMapName = dataElement.getColorMapName();
        if(cMapName.size() > 0) {
            TF_IOTOEASY(fileElement, metaData, "colorMap", cMapName);
        }

        if(dataElement.hasMapAngle()) {
            TF_IOTOEASY(fileElement, metaData, "mapAngle", dataElement.getMapAngle());
        }
        if(dataElement.hasMapBond()) {
            TF_IOTOEASY(fileElement, metaData, "mapBond", dataElement.getMapBond());
        }
        if(dataElement.hasMapDihedral()) {
            TF_IOTOEASY(fileElement, metaData, "mapDihedral", dataElement.getMapDihedral());
        }
        if(dataElement.hasMapParticle()) {
            TF_IOTOEASY(fileElement, metaData, "mapParticle", dataElement.getMapParticle());
        }

        fileElement.get()->type = "ColorMapper";

        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, rendering::ColorMapper *dataElement) {

        TF_IOFROMEASY(fileElement, metaData, "species_index", &dataElement->species_index);
        TF_IOFROMEASY(fileElement, metaData, "min_val", &dataElement->min_val);
        TF_IOFROMEASY(fileElement, metaData, "max_val", &dataElement->max_val);

        IOChildMap fec = IOElement::children(fileElement);
        auto feItr = fec.find("colorMap");
        if(feItr != fec.end()) {
            std::string cMapName;
            TF_IOFROMEASY(fileElement, metaData, "colorMap", &cMapName);

            auto func = rendering::getColorMapperFunc(cMapName);
            if(func)
                dataElement->map = func;
        }

        feItr = fec.find("mapAngle");
        if(feItr != fec.end()) {
            unsigned int mapAngle;
            TF_IOFROMEASY(fileElement, metaData, "mapAngle", &mapAngle);
            dataElement->setMapAngle(mapAngle);
        }
        feItr = fec.find("mapBond");
        if(feItr != fec.end()) {
            unsigned int mapBond;
            TF_IOFROMEASY(fileElement, metaData, "mapBond", &mapBond);
            dataElement->setMapBond(mapBond);
        }
        feItr = fec.find("mapDihedral");
        if(feItr != fec.end()) {
            unsigned int mapDihedral;
            TF_IOFROMEASY(fileElement, metaData, "mapDihedral", &mapDihedral);
            dataElement->setMapDihedral(mapDihedral);
        }
        feItr = fec.find("mapParticle");
        if(feItr != fec.end()) {
            unsigned int mapParticle;
            TF_IOFROMEASY(fileElement, metaData, "mapParticle", &mapParticle);
            dataElement->setMapParticle(mapParticle);
        }

        return S_OK;
    }

};
