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

#ifdef _WIN32
#define _USE_MATH_DEFINES
#endif

#include "tfUniverse.h"
#include <tfForce.h>
#include "tfSimulator.h"
#include "tf_util.h"
#include "tf_metrics.h"
#include <cmath>
#include <iostream>
#include <limits>
#include "tfThreadPool.h"
#include "tf_bind.h"
#include "state/tfStateVector.h"
#include "state/tfSpeciesList.h"
#include "tf_system.h"
#include "tfError.h"
#include "rendering/tfStyle.h"
#include "io/tfFIO.h"
#include <tf_mdcore_io.h>


using namespace TissueForge;


Universe TissueForge::_Universe = {
    .isRunning = false
};

Universe *TissueForge::getUniverse() {
    return &_Universe;
}

// the single static engine instance per process

// make the global engine to show up here
engine TissueForge::_Engine = {
        .flags = 0
};

// default to paused universe
static uint32_t universe_flags = 0;


struct engine* TissueForge::engine_get()
{
    return &_Engine;
}


// TODO: fix error handling values
#define TF_UNIVERSE_CHECKERROR() { \
    if (_Engine.flags == 0 ) { \
        std::string err = "Error in "; \
        err += TF_FUNCTION; \
        err += ", Universe not initialized"; \
        return tf_error(E_FAIL, err.c_str()); \
    } \
    }

#define TF_UNIVERSE_TRY() \
    try {\
        if(_Engine.flags == 0) { \
            std::string err = TF_FUNCTION; \
            err += "universe not initialized"; \
            tf_exp(std::domain_error(err.c_str())); \
        }

#define TF_UNIVERSE_CHECK(hr) \
    if(SUCCEEDED(hr)) { Py_RETURN_NONE; } \
    else {return NULL;}

#define TF_UNIVERSE_FINALLY(retval) \
    } \
    catch(const std::exception &e) { \
        tf_exp(e); return retval; \
    }

UniverseConfig::UniverseConfig() :
    dim {10, 10, 10},
    spaceGridSize {4, 4, 4},
    cutoff{1},
    flags{0},
    maxTypes{64},
    dt{0.01}, 
    start_step{0},
    temp{1},
    nParticles{100},
    threads{ThreadPool::hardwareThreadSize()},
    integrator{EngineIntegrator::FORWARD_EULER},
    boundaryConditionsPtr{new BoundaryConditionsArgsContainer()},
    max_distance{-1},
    timers_mask {0},
    timer_output_period {-1}
{
}

std::string Universe::getName() {
    TF_UNIVERSE_TRY();
    return _Universe.name;
    TF_UNIVERSE_FINALLY("");
}

FMatrix3 *Universe::virial(FVector3 *origin, FloatP_t *radius, std::vector<ParticleType*> *types) {
    try {
        FVector3 _origin = origin ? *origin : Universe::getCenter();
        FloatP_t _radius = radius ? *radius : 2 * _origin.max();

        std::set<short int> typeIds;

        if (types) {
            for (auto type : *types) 
                if (type) 
                    typeIds.insert(type->id);
        }
        else {
            for(int i = 0; i < _Engine.nr_types; ++i) 
                typeIds.insert(i);
        }

        FMatrix3 *m;
        if(SUCCEEDED(metrics::calculateVirial(_origin.data(), _radius, typeIds, m->data()))) {
            return m;
        }
    }
    catch(const std::exception &e) {
        TF_RETURN_EXP(e);
    }
    return NULL;
}

HRESULT Universe::step(const FloatP_t &until, const FloatP_t &dt) {
    TF_UNIVERSE_TRY();
    return Universe_Step(until, dt);
    TF_UNIVERSE_FINALLY(1);
}

HRESULT Universe::stop() {
    TF_UNIVERSE_TRY();
    return Universe_SetFlag(Universe::Flags::RUNNING, false);
    TF_UNIVERSE_FINALLY(1);
}

HRESULT Universe::start() {
    TF_UNIVERSE_TRY();
    return Universe_SetFlag(Universe::Flags::RUNNING, true);
    TF_UNIVERSE_FINALLY(1);
}

HRESULT Universe::reset() {
    TF_UNIVERSE_TRY();
    return engine_reset(&_Engine);
    TF_UNIVERSE_FINALLY(1);
}

Universe* Universe::get() {
    return &_Universe;
}

ParticleList Universe::particles() {
    TF_UNIVERSE_TRY();
    return ParticleList::all();
    TF_UNIVERSE_FINALLY(ParticleList());
}

std::vector<int32_t> Universe::particleIds() {
    TF_UNIVERSE_TRY();
    return ParticleList::all().vector();
    TF_UNIVERSE_FINALLY({});
}

void Universe::resetSpecies() {
    TF_UNIVERSE_TRY();
    
    for(int i = 0; i < _Engine.s.size_parts; ++i) {
        Particle *part = _Engine.s.partlist[i];
        if(part && part->state_vector) {
            part->state_vector->reset();
        }
    }
    
    for(int i = 0; i < _Engine.s.largeparts.count; ++i) {
        Particle *part = &_Engine.s.largeparts.parts[i];
        if(part && part->state_vector) {
            part->state_vector->reset();
        }
    }
    
    // redraw, state changed. 
    Simulator::get()->redraw();
    
    TF_UNIVERSE_FINALLY();
}

std::vector<std::vector<std::vector<ParticleList> > > Universe::grid(iVector3 shape) {
    TF_UNIVERSE_TRY();
    return metrics::particleGrid(shape);
    TF_UNIVERSE_FINALLY(std::vector<std::vector<std::vector<ParticleList> > >());
}

std::vector<BondHandle> Universe::bonds() {
    std::vector<BondHandle> bonds;
    TF_UNIVERSE_TRY();
    bonds.reserve(_Engine.nr_bonds);

    for(int i = 0; i < _Engine.nr_bonds; ++i) {
        Bond *b = &_Engine.bonds[i];
        if (b->flags & BOND_ACTIVE)
            bonds.push_back(BondHandle(i));
    }
    return bonds;
    TF_UNIVERSE_FINALLY(bonds);
}

std::vector<AngleHandle> Universe::angles() {
    std::vector<AngleHandle> angles;
    TF_UNIVERSE_TRY();
    angles.reserve(_Engine.nr_angles);

    for(int i = 0; i < _Engine.nr_angles; ++i) {
        Angle *a = &_Engine.angles[i];
        if (a->flags & BOND_ACTIVE)
            angles.push_back(AngleHandle(i));
    }
    return angles;
    TF_UNIVERSE_FINALLY(angles);
}

std::vector<DihedralHandle> Universe::dihedrals() {
    std::vector<DihedralHandle> dihedrals;
    TF_UNIVERSE_TRY();
    dihedrals.reserve(_Engine.nr_dihedrals);

    for(int i = 0; i < _Engine.nr_dihedrals; ++i) {
        Dihedral *d = &_Engine.dihedrals[i];
        dihedrals.push_back(DihedralHandle(i));
    }
    return dihedrals;
    TF_UNIVERSE_FINALLY(dihedrals);
}

FloatP_t Universe::getTemperature() {
    TF_UNIVERSE_TRY();
    return engine_temperature(&_Engine);
    TF_UNIVERSE_FINALLY(0);
}

FloatP_t Universe::getTime() {
    TF_UNIVERSE_TRY();
    return _Engine.time * _Engine.dt;
    TF_UNIVERSE_FINALLY(0);
}

FloatP_t Universe::getDt() {
    TF_UNIVERSE_TRY();
    return _Engine.dt;
    TF_UNIVERSE_FINALLY(0);
}

event::EventList *Universe::getEventList() {
    TF_UNIVERSE_TRY();
    return (event::EventList *)_Universe.events;
    TF_UNIVERSE_FINALLY(NULL);
}

BoundaryConditions *Universe::getBoundaryConditions() {
    TF_UNIVERSE_TRY();
    return &_Engine.boundary_conditions;
    TF_UNIVERSE_FINALLY(NULL);
}

FloatP_t Universe::getKineticEnergy() {
    TF_UNIVERSE_TRY();
    return engine_kinetic_energy(&_Engine);
    TF_UNIVERSE_FINALLY(0);
}

int Universe::getNumTypes() {
    TF_UNIVERSE_TRY();
    return _Engine.nr_types;
    TF_UNIVERSE_FINALLY(0);
}

FloatP_t Universe::getCutoff() {
    TF_UNIVERSE_TRY();
    return _Engine.s.cutoff;
    TF_UNIVERSE_FINALLY(0);
}

FVector3 Universe::dim()
{
    TF_UNIVERSE_TRY();
    return FVector3::from(_Engine.s.dim);
    TF_UNIVERSE_FINALLY(FVector3());
}

FloatP_t Universe::volume() {
    TF_UNIVERSE_TRY();
    return FVector3::from(_Engine.s.dim).product();
    TF_UNIVERSE_FINALLY(0);
}

HRESULT TissueForge::Universe_Step(FloatP_t until, FloatP_t dt) {

    // Ok to call here, since nothing happens if root element is already released. 
    io::FIO::releaseIORootElement();

    // TODO: add support for adaptive time stepping
    // if (dt <= 0.0) dt = _Engine.dt;
    dt = _Engine.dt;

    FloatP_t dtStore = _Engine.dt;
    _Engine.dt = dt;

    if (until <= 0.0) until = _Engine.dt;

    FloatP_t tf = _Engine.time + until / dtStore;

    while (_Engine.time < tf) {
        if(engine_step(&_Engine) != S_OK) 
            return tf_error(E_FAIL, errs_err_msg[MDCERR_engine]);

        // notify time listeners
        if(_Universe.events->eval(_Engine.time * _Engine.dt) != S_OK) 
            return tf_error(E_FAIL, errs_err_msg[MDCERR_engine]);

        if(_Engine.timer_output_period > 0 && _Engine.time % _Engine.timer_output_period == 0 ) {
            system::printPerformanceCounters();
        }

    }

    _Engine.dt = dtStore;

    return S_OK;
}

// TODO: does it make sense to return an hresult???
int TissueForge::Universe_Flag(Universe::Flags flag)
{
    TF_UNIVERSE_CHECKERROR();
    return universe_flags & flag;
}

CAPI_FUNC(HRESULT) TissueForge::Universe_SetFlag(Universe::Flags flag, int value)
{
    TF_UNIVERSE_CHECKERROR();

    if(value) {
        universe_flags |= flag;
    }
    else {
        universe_flags &= ~(flag);
    }

    return Simulator::get()->redraw();
}

FVector3 Universe::getCenter() {
    return engine_center();
}


namespace TissueForge::io {


    template <>
    HRESULT toFile(const Universe &dataElement, const MetaData &metaData, IOElement &fileElement) {

        Universe *u = const_cast<Universe*>(&dataElement);

        TF_IOTOEASY(fileElement, metaData, "name", u->name);
        
        ParticleList pl = u->particles();
        if(pl.nr_parts > 0) {
            std::vector<Particle> particles;
            particles.reserve(pl.nr_parts);
            for(unsigned int i = 0; i < pl.nr_parts; i++) {
                auto ph = pl.item(i);
                if(ph != NULL) {
                    auto p = ph->part();
                    if(p != NULL && !(p->flags & PARTICLE_NONE))
                        particles.push_back(*p);
                }
            }
            TF_IOTOEASY(fileElement, metaData, "particles", particles);
        }

        // Store bonds; potentials and styles are stored separately to reduce storage
        
        std::vector<BondHandle> bhl = u->bonds();
        std::vector<Potential*> bondPotentials;
        std::vector<std::vector<unsigned int> > bondPotentialIdx;
        std::vector<rendering::Style> bondStyles;
        std::vector<rendering::Style*> bondStylesP;
        std::vector<std::vector<unsigned int> > bondStyleIdx;
        if(bhl.size() > 0) {
            std::vector<Bond> bl;
            bl.reserve(bhl.size());
            for(auto bh : bhl) {
                auto b = bh.get();
                if(b->flags & BOND_ACTIVE) {
                    if(b->potential != NULL) {
                        int idx = -1;
                        for(unsigned int i = 0; i < bondPotentials.size(); i++) {
                            if(bondPotentials[i] == b->potential) {
                                idx = i;
                                break;
                            }
                        }

                        if(idx < 0) {
                            idx = bondPotentials.size();
                            bondPotentials.push_back(b->potential);
                            bondPotentialIdx.emplace_back();
                        }
                        bondPotentialIdx[idx].push_back(bl.size());
                    }
                    if(b->style != NULL) {
                        int idx = -1;
                        for(unsigned int i = 0; i < bondStylesP.size(); i++) {
                            if(bondStylesP[i] == b->style) {
                                idx = i;
                                break;
                            }
                        }

                        if(idx < 0) {
                            idx = bondStylesP.size();
                            bondStyles.push_back(*b->style);
                            bondStylesP.push_back(b->style);
                            bondStyleIdx.emplace_back();
                        }
                        bondStyleIdx[idx].push_back(bl.size());
                    }

                    bl.push_back(*b);
                }
            }
            TF_IOTOEASY(fileElement, metaData, "bonds", bl);
            TF_IOTOEASY(fileElement, metaData, "bondPotentials", bondPotentials);
            TF_IOTOEASY(fileElement, metaData, "bondPotentialIdx", bondPotentialIdx);
            TF_IOTOEASY(fileElement, metaData, "bondStyles", bondStyles);
            TF_IOTOEASY(fileElement, metaData, "bondStyleIdx", bondStyleIdx);
        }

        // Store angles; potentials and styles are stored separately to reduce storage
        
        std::vector<AngleHandle> ahl = u->angles();
        std::vector<Potential*> anglePotentials;
        std::vector<std::vector<unsigned int> > anglePotentialIdx;
        std::vector<rendering::Style> angleStyles;
        std::vector<rendering::Style*> angleStylesP;
        std::vector<std::vector<unsigned int> > angleStyleIdx;
        if(ahl.size() > 0) {
            std::vector<Angle> al;
            al.reserve(ahl.size());
            for(auto ah : ahl) {
                auto a = ah.get();
                if(a->flags & ANGLE_ACTIVE) {
                    if(a->potential != NULL) {
                        int idx = -1;
                        for(unsigned int i = 0; i < anglePotentials.size(); i++) {
                            if(anglePotentials[i] == a->potential) {
                                idx = i;
                                break;
                            }
                        }

                        if(idx < 0) {
                            idx = anglePotentials.size();
                            anglePotentials.push_back(a->potential);
                            anglePotentialIdx.emplace_back();
                        }
                        anglePotentialIdx[idx].push_back(al.size());
                    }
                    if(a->style != NULL) {
                        int idx = -1;
                        for(unsigned int i = 0; i < angleStylesP.size(); i++) {
                            if(angleStylesP[i] == a->style) {
                                idx = i;
                                break;
                            }
                        }

                        if(idx < 0) {
                            idx = angleStylesP.size();
                            angleStyles.push_back(*a->style);
                            angleStylesP.push_back(a->style);
                            angleStyleIdx.emplace_back();
                        }
                        angleStyleIdx[idx].push_back(al.size());
                    }

                    al.push_back(*a);
                }
            }
            TF_IOTOEASY(fileElement, metaData, "angles", al);
            TF_IOTOEASY(fileElement, metaData, "anglePotentials", anglePotentials);
            TF_IOTOEASY(fileElement, metaData, "anglePotentialIdx", anglePotentialIdx);
            TF_IOTOEASY(fileElement, metaData, "angleStyles", angleStyles);
            TF_IOTOEASY(fileElement, metaData, "angleStyleIdx", angleStyleIdx);
        }

        // Store dihedrals; potentials and styles are stored separately to reduce storage
        
        std::vector<DihedralHandle> dhl = u->dihedrals();
        std::vector<Potential*> dihedralPotentials;
        std::vector<std::vector<unsigned int> > dihedralPotentialIdx;
        std::vector<rendering::Style> dihedralStyles;
        std::vector<rendering::Style*> dihedralStylesP;
        std::vector<std::vector<unsigned int> > dihedralStyleIdx;
        if(dhl.size() > 0) {
            std::vector<Dihedral> dl;
            dl.reserve(dhl.size());
            for(auto dh : dhl) {
                auto d = dh.get();
                if(d != NULL) {
                    if(d->potential != NULL) {
                        int idx = -1;
                        for(unsigned int i = 0; i < dihedralPotentials.size(); i++) {
                            if(dihedralPotentials[i] == d->potential) {
                                idx = i;
                                break;
                            }
                        }

                        if(idx < 0) {
                            idx = dihedralPotentials.size();
                            dihedralPotentials.push_back(d->potential);
                            dihedralPotentialIdx.emplace_back();
                        }
                        dihedralPotentialIdx[idx].push_back(dl.size());
                    }
                    if(d->style != NULL) {
                        int idx = -1;
                        for(unsigned int i = 0; i < dihedralStylesP.size(); i++) {
                            if(dihedralStylesP[i] == d->style) {
                                idx = i;
                                break;
                            }
                        }

                        if(idx < 0) {
                            idx = dihedralStylesP.size();
                            dihedralStyles.push_back(*d->style);
                            dihedralStylesP.push_back(d->style);
                            dihedralStyleIdx.emplace_back();
                        }
                        dihedralStyleIdx[idx].push_back(dl.size());
                    }

                    dl.push_back(*d);
                }
            }
            TF_IOTOEASY(fileElement, metaData, "dihedrals", dl);
            TF_IOTOEASY(fileElement, metaData, "dihedralPotentials", dihedralPotentials);
            TF_IOTOEASY(fileElement, metaData, "dihedralPotentialIdx", dihedralPotentialIdx);
            TF_IOTOEASY(fileElement, metaData, "dihedralStyles", dihedralStyles);
            TF_IOTOEASY(fileElement, metaData, "dihedralStyleIdx", dihedralStyleIdx);
        }
        
        TF_IOTOEASY(fileElement, metaData, "temperature", u->getTemperature());
        TF_IOTOEASY(fileElement, metaData, "kineticEnergy", u->getKineticEnergy());

        ParticleTypeList ptl = ParticleTypeList::all();
        std::vector<ParticleType> partTypes;
        partTypes.reserve(ptl.nr_parts);
        for(unsigned int i = 0; i < ptl.nr_parts; i++) 
            partTypes.push_back(*ptl.item(i));
        TF_IOTOEASY(fileElement, metaData, "particleTypes", partTypes);

        Potential *p, *p_cluster;
        std::vector<Potential*> pV, pV_cluster;
        std::vector<unsigned int> pIdxA, pIdxB, pIdxA_cluster, pIdxB_cluster;
        for(unsigned int i = 0; i < ptl.nr_parts; i++) {
            for(unsigned int j = i; j < ptl.nr_parts; j++) {
                unsigned int k = ptl.parts[i] * _Engine.max_type + ptl.parts[j];
                p = _Engine.p[k];
                p_cluster = _Engine.p_cluster[k];
                if(p != NULL) {
                    pV.push_back(p);
                    pIdxA.push_back(i);
                    pIdxB.push_back(j);
                }
                if(p_cluster != NULL) {
                    pV_cluster.push_back(p_cluster);
                    pIdxA_cluster.push_back(i);
                    pIdxB_cluster.push_back(j);
                }
            }
        }
        if(pV.size() > 0) {
            TF_IOTOEASY(fileElement, metaData, "potentials", pV);
            TF_IOTOEASY(fileElement, metaData, "potentialTypeA", pIdxA);
            TF_IOTOEASY(fileElement, metaData, "potentialTypeB", pIdxB);
        }
        if(pV_cluster.size() > 0) {
            TF_IOTOEASY(fileElement, metaData, "potentialsCluster", pV_cluster);
            TF_IOTOEASY(fileElement, metaData, "potentialClusterTypeA", pIdxA_cluster);
            TF_IOTOEASY(fileElement, metaData, "potentialClusterTypeB", pIdxB_cluster);
        }

        // save forces
        
        std::vector<Force*> forces;
        Force *f;
        std::vector<unsigned int> fIdx;
        for(unsigned int i = 0; i < ptl.nr_parts; i++) { 
            auto pTypeId = ptl.parts[i];
            f = _Engine.forces[pTypeId];
            if(f != NULL) {
                bool storeForce = true;
                if(f->isCustom()) {
                    CustomForce *cf = (CustomForce*)f;
                    storeForce = cf->userFunc == NULL;
                }
                if(storeForce) {
                    forces.push_back(f);
                    fIdx.push_back(i);
                }
            }
        }
        if(forces.size() > 0) {
            TF_IOTOEASY(fileElement, metaData, "forces", forces);
            TF_IOTOEASY(fileElement, metaData, "forceType", fIdx);
        }

        fileElement.get()->type = "Universe";

        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, Universe *dataElement) {

        FIO::importSummary = new FIOImportSummary();

        Universe *u = Universe::get();

        TF_IOFROMEASY(fileElement, metaData, "name", &u->name);
        TF_IOFROMEASY(fileElement, metaData, "temperature", &_Engine.temperature);

        IOChildMap fec = IOElement::children(fileElement);

        // Setup data should have already been intercepted by this stage, so populate universe

        // load types
        //      Special handling here: export includes default types, 
        //      which must be skipped here to avoid duplicating imported defaults with those of installation

        IOChildMap::const_iterator feItr = fec.find("particleTypes");
        IOElement fePartTypes = feItr->second;
        IOChildMap fePartTypes_children = IOElement::children(fePartTypes);
        for(unsigned int i = 0; i < fePartTypes_children.size(); i++) {
            if(i < 2) { 
                FIO::importSummary->particleTypeIdMap[i] = i;
            } 
            else {
                IOElement fePartType = fePartTypes_children[std::to_string(i)];
                ParticleType partType;
                auto typeId = partType.id;
                fromFile(fePartType, metaData, &_Engine.types[typeId]);
                FIO::importSummary->particleTypeIdMap[i] = typeId;
            }
        }

        // load potentials

        if(fec.find("potentials") != fec.end()) {
            std::vector<Potential*> pV;
            std::vector<unsigned int> pIdxA, pIdxB;
            TF_IOFROMEASY(fileElement, metaData, "potentials", &pV);
            TF_IOFROMEASY(fileElement, metaData, "potentialTypeA", &pIdxA);
            TF_IOFROMEASY(fileElement, metaData, "potentialTypeB", &pIdxB);
            for(unsigned int i = 0; i < pV.size(); i++) {
                auto typeIdA = FIO::importSummary->particleTypeIdMap[pIdxA[i]];
                auto typeIdB = FIO::importSummary->particleTypeIdMap[pIdxB[i]];
                bind::types(pV[i], &_Engine.types[typeIdA], &_Engine.types[typeIdB]);
            }
        }
        if(fec.find("potentialsCluster") != fec.end()) {
            std::vector<Potential*> pV_cluster;
            std::vector<unsigned int> pIdxA_cluster, pIdxB_cluster;
            TF_IOFROMEASY(fileElement, metaData, "potentialsCluster", &pV_cluster);
            TF_IOFROMEASY(fileElement, metaData, "potentialClusterTypeA", &pIdxA_cluster);
            TF_IOFROMEASY(fileElement, metaData, "potentialClusterTypeB", &pIdxB_cluster);
            for(unsigned int i = 0; i < pV_cluster.size(); i++) {
                auto typeIdA = FIO::importSummary->particleTypeIdMap[pIdxA_cluster[i]];
                auto typeIdB = FIO::importSummary->particleTypeIdMap[pIdxB_cluster[i]];
                bind::types(pV_cluster[i], &_Engine.types[typeIdA], &_Engine.types[typeIdB], true);
            }
        }

        // load forces

        if(fec.find("forces") != fec.end()) {
            std::vector<Force*> forces;
            std::vector<unsigned int> fIdx;
            int fSVIdx;
            
            TF_IOFROMEASY(fileElement, metaData, "forces", &forces);
            TF_IOFROMEASY(fileElement, metaData, "forceType", &fIdx);
            for(unsigned int i = 0; i < fIdx.size(); i++) { 
                auto pType = &_Engine.types[FIO::importSummary->particleTypeIdMap[fIdx[i]]];
                Force *f = forces[i];
                bind::force(f, pType);
            }
        }
        
        // load particles

        if(fec.find("particles") != fec.end()) {

            std::vector<Particle> particles;
            Particle *part;
            TF_IOFROMEASY(fileElement, metaData, "particles", &particles);
            auto feParticles = feItr->second;
            for(unsigned int i = 0; i < particles.size(); i++) {
                Particle &p = particles[i];
                auto typeId = FIO::importSummary->particleTypeIdMap[p.typeId];
                auto pType = &_Engine.types[typeId];
                int32_t clusterId = p.clusterId > 0 ? FIO::importSummary->particleIdMap[p.clusterId] : p.clusterId;
                auto ph = Particle_New(pType, &p.position, &p.velocity, &clusterId);
                auto part = ph->part();
                auto pId = p.id;
                FIO::importSummary->particleIdMap[pId] = part->id;

                part->radius = p.radius;
                part->mass = p.mass;
                part->imass = p.imass;
                part->flags = p.flags;
                part->creation_time = p.creation_time;
                if(p.state_vector) {
                    for(unsigned int j = 0; j < p.state_vector->size; j++) {
                        state::Species *species = p.state_vector->species->item(j);
                        int k = part->state_vector->species->index_of(species->getId().c_str());
                        if(k >= 0) 
                            part->state_vector->fvec[k] = p.state_vector->fvec[j];
                    }
                }
                if(p.style) 
                    part->style = new rendering::Style(*p.style);
            }

        }

        // load bonds; potentials and styles are stored separately to reduce storage

        if(fec.find("bonds") != fec.end()) {
            std::vector<Bond> bonds;
            std::vector<Potential*> bondPotentials;
            std::vector<std::vector<unsigned int> > bondPotentialIdx;
            std::vector<rendering::Style> bondStyles;
            std::vector<std::vector<unsigned int> > bondStyleIdx;
            TF_IOFROMEASY(fileElement, metaData, "bonds", &bonds);
            TF_IOFROMEASY(fileElement, metaData, "bondPotentials", &bondPotentials);
            TF_IOFROMEASY(fileElement, metaData, "bondPotentialIdx", &bondPotentialIdx);
            TF_IOFROMEASY(fileElement, metaData, "bondStyles", &bondStyles);
            TF_IOFROMEASY(fileElement, metaData, "bondStyleIdx", &bondStyleIdx);
            std::vector<BondHandle> bondsCreated(bonds.size(), BondHandle());

            for(unsigned int i = 0; i < bondPotentialIdx.size(); i++) { 
                auto bIndices = bondPotentialIdx[i];
                Potential *p = bondPotentials[i];
                for(auto bIdx : bIndices) {
                    auto b = bonds[bIdx];
                    BondHandle bh(
                        p, 
                        FIO::importSummary->particleIdMap[b.i], 
                        FIO::importSummary->particleIdMap[b.j], 
                        b.half_life, b.dissociation_energy, b.flags
                    );
                    auto be = bh.get();

                    be->creation_time = b.creation_time;
                    bondsCreated[bIdx] = bh;
                }
            }

            for(unsigned int i = 0; i < bondStyleIdx.size(); i++) {
                auto bIndices = bondStyleIdx[i];
                rendering::Style *s = new rendering::Style(bondStyles[i]);
                for(auto bIdx : bIndices) {
                    auto bh = bondsCreated[bIdx];
                    if(bh.id >= 0) 
                        bh.get()->style = s;
                }
            }
        }

        // load angles; potentials and styles are stored separately to reduce storage
        
        if(fec.find("angles") != fec.end()) {
            std::vector<Angle> angles;
            std::vector<Potential*> anglePotentials;
            std::vector<std::vector<unsigned int> > anglePotentialIdx;
            std::vector<rendering::Style> angleStyles;
            std::vector<std::vector<unsigned int> > angleStyleIdx;
            TF_IOFROMEASY(fileElement, metaData, "angles", &angles);
            TF_IOFROMEASY(fileElement, metaData, "anglePotentials", &anglePotentials);
            TF_IOFROMEASY(fileElement, metaData, "anglePotentialIdx", &anglePotentialIdx);
            TF_IOFROMEASY(fileElement, metaData, "angleStyles", &angleStyles);
            TF_IOFROMEASY(fileElement, metaData, "angleStyleIdx", &angleStyleIdx);
            std::vector<AngleHandle*> anglesCreated(angles.size(), 0);

            for(unsigned int i = 0; i < anglePotentialIdx.size(); i++) { 
                auto aIndices = anglePotentialIdx[i];
                Potential *p = anglePotentials[i];
                for(auto aIdx : aIndices) {
                    auto a = angles[aIdx];
                    Particle *pi, *pj, *pk;
                    pi = _Engine.s.partlist[FIO::importSummary->particleIdMap[a.i]];
                    pj = _Engine.s.partlist[FIO::importSummary->particleIdMap[a.j]];
                    pk = _Engine.s.partlist[FIO::importSummary->particleIdMap[a.k]];
                    auto ah = Angle::create(p, pi->handle(), pj->handle(), pk->handle(), a.flags);
                    auto ae = ah->get();

                    ae->half_life = a.half_life;
                    ae->dissociation_energy = a.dissociation_energy;
                    ae->creation_time = a.creation_time;
                    anglesCreated[aIdx] = ah;
                }
            }

            for(unsigned int i = 0; i < angleStyleIdx.size(); i++) {
                auto aIndices = angleStyleIdx[i];
                rendering::Style *s = new rendering::Style(angleStyles[i]);
                for(auto aIdx : aIndices) {
                    auto a = anglesCreated[aIdx];
                    if(a != NULL) 
                        a->get()->style = s;
                }
            }
        }

        // load dihedrals; potentials and styles are stored separately to reduce storage
        
        if(fec.find("dihedrals") != fec.end()) {
            std::vector<Dihedral> dihedrals;
            std::vector<Potential*> dihedralPotentials;
            std::vector<std::vector<unsigned int> > dihedralPotentialIdx;
            std::vector<rendering::Style> dihedralStyles;
            std::vector<std::vector<unsigned int> > dihedralStyleIdx;
            TF_IOFROMEASY(fileElement, metaData, "dihedrals", &dihedrals);
            TF_IOFROMEASY(fileElement, metaData, "dihedralPotentials", &dihedralPotentials);
            TF_IOFROMEASY(fileElement, metaData, "dihedralPotentialIdx", &dihedralPotentialIdx);
            TF_IOFROMEASY(fileElement, metaData, "dihedralStyles", &dihedralStyles);
            TF_IOFROMEASY(fileElement, metaData, "dihedralStyleIdx", &dihedralStyleIdx);
            std::vector<DihedralHandle*> dihedralsCreated(dihedrals.size(), 0);

            for(unsigned int i = 0; i < dihedralPotentialIdx.size(); i++) { 
                auto dIndices = dihedralPotentialIdx[i];
                Potential *p = dihedralPotentials[i];
                for(auto dIdx : dIndices) {
                    auto d = dihedrals[dIdx];
                    Particle *pi, *pj, *pk, *pl;
                    pi = _Engine.s.partlist[FIO::importSummary->particleIdMap[d.i]];
                    pj = _Engine.s.partlist[FIO::importSummary->particleIdMap[d.j]];
                    pk = _Engine.s.partlist[FIO::importSummary->particleIdMap[d.k]];
                    pl = _Engine.s.partlist[FIO::importSummary->particleIdMap[d.l]];
                    auto dh = Dihedral::create(p, pi->handle(), pj->handle(), pk->handle(), pl->handle());
                    auto de = dh->get();

                    de->half_life = d.half_life;
                    de->dissociation_energy = d.dissociation_energy;
                    de->creation_time = d.creation_time;
                    dihedralsCreated[dIdx] = dh;
                }
            }

            for(unsigned int i = 0; i < dihedralStyleIdx.size(); i++) {
                auto dIndices = dihedralStyleIdx[i];
                rendering::Style *s = new rendering::Style(dihedralStyles[i]);
                for(auto dIdx : dIndices) {
                    auto d = dihedralsCreated[dIdx];
                    if(d != NULL) 
                        d->get()->style = s;
                }
            }
        }

        return S_OK;
    }

};
