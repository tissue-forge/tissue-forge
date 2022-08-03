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

#include "tfCellPolarity.h"

#include <tfUniverse.h>
#include <tf_metrics.h>
#include <tf_util.h>
#include <types/tf_types.h>
#include <tfLogger.h>
#include <tfError.h>
#include <event/tfTimeEvent.h>
#include <tf_bind.h>
#include <io/tfFIO.h>

#include <unordered_set>
#include <utility>


using namespace TissueForge;
namespace CPMod = TissueForge::models::center::CellPolarity;


namespace TissueForge::io { 


    struct CellPolarityFIOModule : io::FIOModule {

        std::string moduleName() { return "CenterCellPolarity"; }

        HRESULT toFile(const io::MetaData &metaData, io::IOElement *fileElement);

        HRESULT fromFile(const io::MetaData &metaData, const io::IOElement &fileElement);
    };

}


namespace TissueForge::models::center::CellPolarity { 


static int polarityVecsIdxOld = 0;
static int polarityVecsIdxCurrent = 1;

static std::string _ABColor = "blue";
static std::string _PCPColor = "green";
static FloatP_t _polarityVectorScale = 0.5;
static FloatP_t _polarityVectorLength = 0.5;
static bool _drawingPolarityVecs = true;

struct PolarityModelParams {
    std::string initMode = "value";
    FVector3 initPolarAB = FVector3(0.0);
    FVector3 initPolarPCP = FVector3(0.0);
};

struct PolarityVecsPack {
    FVector3 v[6];

    FVector3 &operator[](const unsigned int &idx) { return v[idx]; }
};

struct PolarityArrowsPack {
    int i[2];

    int &operator[](const unsigned int &idx) { return i[idx]; }
};

struct ParticlePolarityPack {
    PolarityVecsPack v;
    PolarityArrowsPack i;
    int32_t pId;
    bool showing;
    PolarityArrowData *arrowAB, *arrowPCP;

    ParticlePolarityPack() :
        showing{false}
    {}
    ParticlePolarityPack(PolarityVecsPack _v, const int32_t &_pId, const bool &_showing=true) : 
        v{_v}, pId{_pId}, arrowAB{NULL}, arrowPCP{NULL}, showing{false}
    {
        if(_showing) this->addArrows();
    }

    FVector3 &vectorAB(const bool &current=true) {
        int idx = current ? polarityVecsIdxCurrent : polarityVecsIdxOld;
        
        return this->v[idx];
    }

    FVector3 &vectorPCP(const bool &current=true) {
        int idx = current ? polarityVecsIdxCurrent : polarityVecsIdxOld;
        
        return this->v[idx + 2];
    }

    void cacheVectorIncrements(const FVector3 &vecAB, const FVector3 &vecPCP) {
        this->v[4] += vecAB;
        this->v[5] += vecPCP;
    }

    void applyVectorIncrements() {
        this->v[polarityVecsIdxOld    ] = (this->v[polarityVecsIdxCurrent    ] + this->v[4]).normalized();
        this->v[polarityVecsIdxOld + 2] = (this->v[polarityVecsIdxCurrent + 2] + this->v[5]).normalized();
        this->v[4] = FVector3(0.0);
        this->v[5] = FVector3(0.0);
    }

    void addArrows() {
        if(this->showing) return;

        auto *renderer = rendering::ArrowRenderer::get();
        this->arrowAB = new PolarityArrowData();
        this->arrowPCP = new PolarityArrowData();
        
        this->arrowAB->scale = _polarityVectorScale;
        this->arrowAB->arrowLength = _polarityVectorLength;
        this->arrowAB->style.setColor(_ABColor);
        this->arrowPCP->scale = _polarityVectorScale;
        this->arrowPCP->arrowLength = _polarityVectorLength;
        this->arrowPCP->style.setColor(_PCPColor);
        auto idAB = renderer->addArrow(this->arrowAB);
        auto idPCP = renderer->addArrow(this->arrowPCP);

        this->i = {idAB, idPCP};
        this->showing = true;
    }

    void removeArrows() {
        if(!this->showing) return;

        auto *renderer = rendering::ArrowRenderer::get();
        renderer->removeArrow(i[0]);
        renderer->removeArrow(i[1]);

        delete this->arrowAB;
        delete this->arrowPCP;
        this->arrowAB = NULL;
        this->arrowPCP = NULL;

        this->showing = false;
    }

    void updateArrows(const bool &current=true) {
        ParticleHandle *ph = Particle_FromId(pId)->handle();
        FVector3 position = ph->getPosition();

        // Scaling such that each vector appears oustide the particle with the prescribed scale
        FloatP_t scaleAB = (ph->getRadius() + this->arrowAB->arrowLength) / this->arrowAB->scale;
        FloatP_t scalePCP = (ph->getRadius() + this->arrowPCP->arrowLength) / this->arrowPCP->scale;

        this->arrowAB->position = position;
        this->arrowPCP->position = position;
        this->arrowAB->components = this->vectorAB(current) * scaleAB;
        this->arrowPCP->components = this->vectorPCP(current) * scalePCP;
    }
};

typedef std::unordered_map<int32_t, PolarityModelParams*> PolarityParamsType;
typedef std::vector<ParticlePolarityPack*> PartPolPackType;

int nr_partPolPack, size_partPolPack, inc_partPolPack=100;
static PartPolPackType *_partPolPack = NULL;
static PolarityParamsType *_polarityParams = NULL;

void polarityVecsFlip() {
    if(polarityVecsIdxOld == 0) {
        polarityVecsIdxOld = 1;
        polarityVecsIdxCurrent = 0;
    }
    else {
        polarityVecsIdxOld = 0;
        polarityVecsIdxCurrent = 1;
    }
}

void initPartPolPack() {
    if(_partPolPack) return;
    _partPolPack = new PartPolPackType();
    _partPolPack->resize(inc_partPolPack, NULL);
    size_partPolPack = inc_partPolPack;
    nr_partPolPack = 0;
}

ParticlePolarityPack *insertPartPolPack(const int &pId, const PolarityVecsPack &pvp) {
    while (pId >= size_partPolPack) {
        size_partPolPack += inc_partPolPack;
        _partPolPack->resize(size_partPolPack, NULL);
    }

    if(pId < nr_partPolPack && (*_partPolPack)[pId] != NULL) {
        tf_exp(std::invalid_argument("polarity parameters already set!"));
        return NULL;
    }

    auto ppp = new ParticlePolarityPack(pvp, pId);
    if(ppp->showing) ppp->updateArrows();
    (*_partPolPack)[pId] = ppp;
    nr_partPolPack++;
    return ppp;
}

void removePartPolPack(const int &pId) {
    auto p = (*_partPolPack)[pId];

    if(!p) {
        tf_exp(std::invalid_argument("polarity parameters not set!"));
        return;
    }

    if(p->showing) p->removeArrows();
    delete p;
    (*_partPolPack)[pId] = NULL;
    nr_partPolPack--;
}

std::pair<FVector3, FVector3> initPolarityVec(const int &pId) {
    Particle *p = Particle_FromId(pId);
    if(!p) return std::make_pair(FVector3(0.0), FVector3(0.0));
    
    auto itrPolarityParams = _polarityParams->find(p->typeId);
    if(itrPolarityParams == _polarityParams->end()) {
        TF_Log(LOG_TRACE) << "No known particle type for initializing polar particle: " << pId << ", " << p->typeId;

        return std::make_pair(FVector3(0.0), FVector3(0.0));
    }

    PolarityModelParams *pmp = itrPolarityParams->second;
    FVector3 ivAB, ivPCP;
    if(strcmp(pmp->initMode.c_str(), "value") == 0) {
        ivAB = FVector3(pmp->initPolarAB);
        ivPCP = FVector3(pmp->initPolarPCP);
    }
    else if(strcmp(pmp->initMode.c_str(), "random") == 0) {
        ivAB = randomPoint(PointsType::Sphere);
        ivPCP = randomPoint(PointsType::Sphere);
    }
    TF_Log(LOG_TRACE) << "Initialized particle " << pId << ": " << ivAB << ", " << ivPCP;

    insertPartPolPack(pId, {ivAB, ivAB, ivPCP, ivPCP, FVector3(0.0), FVector3(0.0)});

    return std::make_pair(ivAB, ivPCP);
}

void registerParticle(const int &pId) {
    if(pId >= nr_partPolPack || (*_partPolPack)[pId] == NULL) initPolarityVec(pId);
}

void registerParticle(ParticleHandle *ph) {
    if(!ph) return;

    registerParticle(ph->id);
}

void unregister(ParticleHandle *ph) {
    removePartPolPack(ph->id);
}

void registerType(
    ParticleType *pType, 
    const std::string &initMode, 
    const FVector3 &initPolarAB, 
    const FVector3 &initPolarPCP) 
{
    if(!pType) return;

    auto itr = _polarityParams->find(pType->id);
    if(itr != _polarityParams->end()) {
        _polarityParams->erase(itr);
        delete itr->second;
    }

    PolarityModelParams *pmp = new PolarityModelParams();
    pmp->initMode = initMode;
    pmp->initPolarAB = initPolarAB;
    pmp->initPolarPCP = initPolarPCP;
    (*_polarityParams)[pType->id] = pmp;
}

FVector3 getVectorAB(const int &pId, const bool &current) {
    return (*_partPolPack)[pId]->vectorAB(current);
}

void setVectorAB(const int &pId, const FVector3 &pVec, const bool &current, const bool &init) {
    auto pp = (*_partPolPack)[pId];
    pp->v[current ? polarityVecsIdxCurrent : polarityVecsIdxOld] = pVec;
    pp->updateArrows(current);
    if (init) setVectorAB(pId, pVec, !current, false);
}

FVector3 getVectorPCP(const int &pId, const bool &current) {
    return (*_partPolPack)[pId]->vectorPCP(current);
}

void setVectorPCP(const int &pId, const FVector3 &pVec, const bool &current, const bool &init) {
    auto pp = (*_partPolPack)[pId];
    auto idx = current ? polarityVecsIdxCurrent : polarityVecsIdxOld;
    pp->v[idx + 2] = pVec;
    pp->updateArrows(current);
    if (init) setVectorPCP(pId, pVec, !current, false);
}

void cacheVectorIncrements(const int &pId, const FVector3 &vecAB, const FVector3 &vecPCP) {
    (*_partPolPack)[pId]->cacheVectorIncrements(vecAB, vecPCP);
}

void applyVectorIncrements(const int &pId) {
    (*_partPolPack)[pId]->applyVectorIncrements();
}

const std::string getInitMode(ParticleType *pType) {
    if(!pType) return "";

    auto itr = _polarityParams->find(pType->id);
    if(itr == _polarityParams->end()) return "";

    return itr->second->initMode;
}

void setInitMode(ParticleType *pType, const std::string &value) {
    if(!pType) return;

    auto itr = _polarityParams->find(pType->id);
    if(itr == _polarityParams->end()) return;

    itr->second->initMode = value;
}

const FVector3 getInitPolarAB(ParticleType *pType) {
    if(!pType) return FVector3(0.0);

    auto itr = _polarityParams->find(pType->id);
    if(itr == _polarityParams->end()) return FVector3(0.0);

    return itr->second->initPolarAB;
}

void setInitPolarAB(ParticleType *pType, const FVector3 &value) {
    if(!pType) return;

    auto itr = _polarityParams->find(pType->id);
    if(itr == _polarityParams->end()) return;

    itr->second->initPolarAB = value;
}

const FVector3 getInitPolarPCP(ParticleType *pType) {
    if(!pType) return FVector3(0.0);

    auto itr = _polarityParams->find(pType->id);
    if(itr == _polarityParams->end()) return FVector3(0.0);

    return itr->second->initPolarPCP;
}

void setInitPolarPCP(ParticleType *pType, const FVector3 &value) {
    if(!pType) return;

    auto itr = _polarityParams->find(pType->id);
    if(itr == _polarityParams->end()) return;

    itr->second->initPolarPCP = value;
}

void eval_polarity_force_persistent(struct Force *force, struct Particle *p, int stateVectorId, FPTYPE *f) {
    PersistentForce *pf = (PersistentForce*)force;

    auto ppp = (*_partPolPack)[p->id];
    FVector3 polAB = ppp->vectorAB();
    FVector3 polPCP = ppp->vectorPCP();

    for(int i = 0; i < 3; i++) f[i] += pf->sensAB * polAB[i] + pf->sensPCP * polPCP[i];
}


static std::vector<PersistentForce*> *storedPersistentForces = NULL;

static void storePersistentForce(PersistentForce *f) {
    if(f == NULL) 
        return;

    if(storedPersistentForces == NULL) 
        storedPersistentForces = new std::vector<PersistentForce*>();

    for(auto sf : *storedPersistentForces) 
        if(sf == f) 
            return;

    storedPersistentForces->push_back(f);

}

static void unstorePersistentForce(PersistentForce *f) {
    if(f == NULL || storedPersistentForces == NULL) 
        return;

    auto itr = std::find(storedPersistentForces->begin(), storedPersistentForces->end(), f);
    if(itr != storedPersistentForces->end()) 
        storedPersistentForces->erase(itr);
}

PersistentForce::~PersistentForce() {
    unstorePersistentForce(this);
}

PersistentForce *createPersistentForce(const FloatP_t &sensAB, const FloatP_t &sensPCP) {
    PersistentForce *pf = new PersistentForce();

    storePersistentForce(pf);

    pf->func = (Force_EvalFcn)eval_polarity_force_persistent;
    pf->sensAB = sensAB;
    pf->sensPCP = sensPCP;
    return pf;
}

static inline FMatrix3 tensorProduct(const FVector3 &rowVec, const FVector3 &colVec) {
    FMatrix3 result(1.0);
    for(int j = 0; j < 3; ++j)
        for(int i = 0; i < 3; ++i)
            result[j][i] *= rowVec[i] * colVec[j];
    return result;
}

void setDrawVectors(const bool &_draw) {
    _drawingPolarityVecs = _draw;

    if (_draw) for(auto p : *_partPolPack) p->addArrows();
    else for(auto p : *_partPolPack) p->removeArrows();
}

void setArrowColors(const std::string &colorAB, const std::string &colorPCP) {
    _ABColor = colorAB;
    _PCPColor = colorPCP;

    for(auto p : *_partPolPack) {
        if(!p) continue;

        p->arrowAB->style.setColor(_ABColor);
        p->arrowPCP->style.setColor(_PCPColor);
    }
}

void setArrowScale(const FloatP_t &_scale) {
    _polarityVectorScale = _scale;

    for(auto p : *_partPolPack) {
        if(!p) continue;

        p->arrowAB->scale = _polarityVectorScale;
        p->arrowPCP->scale = _polarityVectorScale;
    }
}

void setArrowLength(const FloatP_t &_length) {
    _polarityVectorLength = _length;

    for(auto p : *_partPolPack) {
        if(!p) continue;

        p->arrowAB->arrowLength = _length;
        p->arrowPCP->arrowLength = _length;
    }
}

PolarityArrowData *getVectorArrowAB(const int32_t &pId) {
    return (*_partPolPack)[pId]->arrowAB;
}

PolarityArrowData *getVectorArrowPCP(const int32_t &pId) {
    return (*_partPolPack)[pId]->arrowPCP;
}

void updatePolariyVectorArrows(const int32_t &pId) {
    (*_partPolPack)[pId]->updateArrows();
}

void removePolarityVectorArrows(const int &pId) {
    TF_Log(LOG_DEBUG) << "";
    
    (*_partPolPack)[pId]->removeArrows();
}

HRESULT run(const event::TimeEvent &event) {
    update();
    return S_OK;
}

static bool _loaded = false;

void load() {
    if(_loaded) return;

    // Instantiate i/o module and register for export
    io::CellPolarityFIOModule *ioModule = new io::CellPolarityFIOModule();
    ioModule->registerIOModule();

    // initialize all module variables
    _polarityParams = new PolarityParamsType();
    initPartPolPack();

    // import data if available
    if(io::FIO::hasImport()) ioModule->load();
    
    // load callback to execute model along with simulation
    event::TimeEventMethod *fcn = new event::TimeEventMethod(run);
    event::onTimeEvent(getUniverse()->getDt(), fcn);

    _loaded = true;
}

void update() {
    int i;
    ParticlePolarityPack *p;
#pragma omp parallel for schedule(static), private(i,p,size_partPolPack,_partPolPack,_drawingPolarityVecs)
    for(i = 0; i < size_partPolPack; i++) {
        p = (*_partPolPack)[i];
        if(!p) continue;

        p->applyVectorIncrements();
        if(_drawingPolarityVecs) p->updateArrows();
    }

    polarityVecsFlip();

    TF_Log(LOG_DEBUG) << "";
}

void eval_potential_cellpolarity(
    struct Potential *p, 
    struct Particle *part_i, 
    struct Particle *part_j, 
    FPTYPE *dx, 
    FPTYPE r2, 
    FPTYPE *e, 
    FPTYPE *f) 
{
    ContactPotential *pot = (ContactPotential*)p;
    auto ppi = (*_partPolPack)[part_i->id];
    auto ppj = (*_partPolPack)[part_j->id];

    if(r2 > pot->b * pot->b) return;

    FVector3 rel_pos = - FVector3::from(dx);
    FloatP_t len_r = std::sqrt(r2);
    FVector3 rh = rel_pos / len_r;

    FVector3 pi = ppi->v[polarityVecsIdxCurrent];
    FVector3 qi = ppi->v[polarityVecsIdxCurrent + 2];
    FVector3 pj = ppj->v[polarityVecsIdxCurrent];
    FVector3 qj = ppj->v[polarityVecsIdxCurrent + 2];

    FloatP_t g = 0.0;
    FVector3 dgdrh(0.0), dgdpi(0.0), dgdqi(0.0), dgdpj(0.0), dgdqj(0.0);
    FVector3 v1, v2;

    if(pot->couplingFlat > 0.0) {
        FloatP_t len_v1, len_v2, u1, u2, u3, u4;
        FVector3 pti(0.0), ptj(0.0);
        FVector3 v3;
        FMatrix3 dptidpi(0.0), dptjdpj(0.0);
        FMatrix3 eye = Magnum::Matrix3x3{Magnum::Math::IdentityInit};
        FMatrix3 ir = eye - tensorProduct(rh, rh);
        FMatrix3 ptiptj;

        switch(pot->cType) {
            case PolarContactType::REGULAR : {
                pti = pi;
                ptj = pj;
                
                u1 = rh.dot(pti);
                u2 = rh.dot(ptj);
                dgdrh += pot->couplingFlat * (2.0 * pti.dot(ptj) * rh - u2 * pti - u1 * ptj);
                dgdpi += pot->couplingFlat * (ptj - u2 * rh);
                dgdpj += pot->couplingFlat * (pti - u1 * rh);
                break;
            }
            case PolarContactType::ISOTROPIC : {
                v3 = - pot->bendingCoeff * rh;
                v1 = pi - v3;
                len_v1 = v1.length();
                
                if(len_v1 > 0.0) {

                    v2 = pj + v3;
                    len_v2 = v2.length();

                    if(len_v2 > 0.0) {
                        pti = v1 / len_v1;
                        ptj = v2 / len_v2;

                        u1 = rh.dot(pti);
                        u2 = rh.dot(ptj);
                        u3 = pti.dot(ptj);
                        u4 = u1 * u2 - u3;
                        dgdrh += pot->couplingFlat * (2.0 * u3 * rh - u2 * pti - u1 * ptj);
                        dgdpi += pot->couplingFlat * (ptj - u2 * rh + u4 * pti) / len_v1;
                        dgdpj += pot->couplingFlat * (pti - u1 * rh + u4 * ptj) / len_v2;
                    }
                }

                break;
            }
            case PolarContactType::ANISOTROPIC : {
                v3 = - 0.5 * (qi + qj);
                v3 *= pot->bendingCoeff * rh.dot(v3);

                v1 = pi + v3;
                len_v1 = v1.length();
                
                if(len_v1 > 0.0) {
                    
                    v2 = pj - v3;
                    len_v2 = v2.length();

                    if(len_v2 > 0.0) {
                        pti = v1 / len_v1;
                        ptj = v2 / len_v2;

                        dptidpi = (eye - tensorProduct(pti, pti)) / len_v1;
                        dptjdpj = (eye - tensorProduct(ptj, ptj)) / len_v2;

                        ptiptj = tensorProduct(pti, ptj);
                        ptiptj = ptiptj + ptiptj.transposed();
                        dgdrh += 2.0 * pot->couplingFlat * pti.dot(ptj) * rh - ptiptj * rh;
                        dgdpi += pot->couplingFlat * (dptidpi * ir * ptj);
                        dgdpj += pot->couplingFlat * (dptjdpj * ir * pti);
                    }
                }
                
                break;
            }
        }

        v1 = Magnum::Math::cross(rh, pti);
        v2 = Magnum::Math::cross(rh, ptj);
        g += pot->couplingFlat * v1.dot(v2);
    }

    if(pot->couplingOrtho > 0.0) {
        v1 = Magnum::Math::cross(pi, qi);
        v2 = Magnum::Math::cross(pj, qj);

        g += v1.dot(v2);

        FloatP_t pipj = pi.dot(pj);
        FloatP_t qiqj = qi.dot(qj);
        FloatP_t piqj = pi.dot(qj);
        FloatP_t pjqi = pj.dot(qi);
        dgdpi += pot->couplingOrtho * (qiqj * pj - pjqi * qj);
        dgdqi += pot->couplingOrtho * (pipj * qj - piqj * pj);
        dgdpj += pot->couplingOrtho * (qiqj * pi - piqj * qi);
        dgdqj += pot->couplingOrtho * (pipj * qi - pjqi * pi);
    }

    if(pot->couplingLateral > 0.0) {
        v1 = Magnum::Math::cross(rh, qi);
        v2 = Magnum::Math::cross(rh, qj);

        g += pot->couplingLateral * (v1.dot(v2));

        FloatP_t rhqi = rh.dot(qi);
        FloatP_t rhqj = rh.dot(qj);
        dgdrh += pot->couplingLateral * (2.0 * qi.dot(qj) * rh - rhqi * qj - rhqj * qi);
        dgdqi += pot->couplingLateral * (qj - rhqj * rh);
        dgdqj += pot->couplingLateral * (qi - rhqi * rh);
    }

    FloatP_t powTerm = std::pow(M_E, -len_r / pot->distanceCoeff);
    FVector3 dgdr = (rh.dot(dgdrh) * rh - dgdrh) / len_r;

    FVector3 incForce = pot->mag * powTerm * (g / pot->distanceCoeff * rh + dgdr);

    // No flipping here. Needs to be done in update function

    FloatP_t polmag = powTerm * pot->rate * getUniverse()->getDt();
    
    ppi->cacheVectorIncrements(polmag * dgdpi, polmag * dgdqi);
    ppj->cacheVectorIncrements(polmag * dgdpj, polmag * dgdqj);

    *e += powTerm * g;
    f[0] += incForce[0];
    f[1] += incForce[1];
    f[2] += incForce[2];

}

ContactPotential::ContactPotential() : 
    Potential()
{
    this->kind = POTENTIAL_KIND_BYPARTICLES;
    this->eval_byparts = PotentialEval_ByParticles(eval_potential_cellpolarity);
}

static std::unordered_map<std::string, PolarContactType> polarContactTypeMap {
    {"regular", PolarContactType::REGULAR}, 
    {"isotropic", PolarContactType::ISOTROPIC}, 
    {"anisotropic", PolarContactType::ANISOTROPIC}
};

static std::vector<ContactPotential*> *storedContactPotentials = NULL;

static void storeContactPotential(ContactPotential *p) {
    if(p == NULL) 
        return;

    if(storedContactPotentials == NULL) 
        storedContactPotentials = new std::vector<ContactPotential*>();

    for(auto sp : *storedContactPotentials) 
        if(sp == p) 
            return;

    storedContactPotentials->push_back(p);

}

static void unstoreContactPotential(ContactPotential *p) {
    if(p == NULL || storedContactPotentials == NULL) 
        return;

    auto itr = std::find(storedContactPotentials->begin(), storedContactPotentials->end(), p);
    if(itr != storedContactPotentials->end()) 
        storedContactPotentials->erase(itr);
}

ContactPotential *_boundContactPotential = NULL;

static void _boundUnstoreContactPotential(Potential *p) {
    ContactPotential *pp = _boundContactPotential;
    unstoreContactPotential(pp);
}

static PotentialClear bindUnstoreContactPotential(ContactPotential *p) {
    _boundContactPotential = p;
    return PotentialClear(_boundUnstoreContactPotential);
}

ContactPotential *createContactPotential(
    const FloatP_t &cutoff, 
    const FloatP_t &mag, 
    const FloatP_t &rate,
    const FloatP_t &distanceCoeff, 
    const FloatP_t &couplingFlat, 
    const FloatP_t &couplingOrtho, 
    const FloatP_t &couplingLateral, 
    std::string contactType, 
    const FloatP_t &bendingCoeff)
{
    ContactPotential *pot = new ContactPotential();

    auto tItr = polarContactTypeMap.find(contactType);
    if(tItr == polarContactTypeMap.end()) tf_exp(std::runtime_error("Invalid type"));

    storeContactPotential(pot);
    pot->clear_func = bindUnstoreContactPotential(pot);

    pot->mag = mag;
    pot->rate = rate;
    pot->distanceCoeff = distanceCoeff;
    pot->couplingFlat = couplingFlat;
    pot->couplingOrtho = couplingOrtho;
    pot->couplingLateral = couplingLateral;
    pot->cType = tItr->second;
    pot->bendingCoeff = bendingCoeff;

    pot->a = std::sqrt(std::numeric_limits<FloatP_t>::epsilon());
    pot->b = cutoff;
    pot->name = "Cell Polarity Contact";

    TF_Log(LOG_TRACE) << "";

    return pot;
}

static bool recursivePotentialCompare(ContactPotential *pp, Potential *p) {
    if(!pp || !p) 
        return false;

    if(p->kind == POTENTIAL_KIND_COMBINATION && p->flags & POTENTIAL_SUM) {
        if(recursivePotentialCompare(pp, p->pca)) return true;

        if(recursivePotentialCompare(pp, p->pcb)) return true;

        return false;
    } 
    else
        return pp == p;

}

static bool recursiveForceCompare(PersistentForce *pf, Force *f) {
    if(!pf || !f) 
        return false;

    if(f->type == FORCE_SUM) {
        ForceSum *sf = (ForceSum*)f;
        if(recursiveForceCompare(pf, sf->f1)) return true;

        if(recursiveForceCompare(pf, sf->f2)) return true;

        return false;
    } 
    else return pf == f;
}


}


#define TF_CENTERCELLPOLARITYIOTOEASY(fe, key, member) \
    fe = new io::IOElement(); \
    if(io::toFile(member, metaData, fe) != S_OK)  \
        return E_FAIL; \
    fe->parent = fileElement; \
    fileElement->children[key] = fe;

#define TF_CENTERCELLPOLARITYIOFROMEASY(feItr, children, metaData, key, member_p) \
    feItr = children.find(key); \
    if(feItr == children.end() || io::fromFile(*feItr->second, metaData, member_p) != S_OK) \
        return E_FAIL;


namespace TissueForge::io {


    template <>
    HRESULT toFile(const CPMod::PolarityModelParams &dataElement, const MetaData &metaData, IOElement *fileElement) {

        IOElement *fe;

        TF_CENTERCELLPOLARITYIOTOEASY(fe, "initMode", dataElement.initMode);
        TF_CENTERCELLPOLARITYIOTOEASY(fe, "initPolarAB", dataElement.initPolarAB);
        TF_CENTERCELLPOLARITYIOTOEASY(fe, "initPolarPCP", dataElement.initPolarPCP);

        fileElement->type = "PolarityParameters";

        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, CPMod::PolarityModelParams *dataElement) {

        IOChildMap::const_iterator feItr;

        TF_CENTERCELLPOLARITYIOFROMEASY(feItr, fileElement.children, metaData, "initMode", &dataElement->initMode);
        TF_CENTERCELLPOLARITYIOFROMEASY(feItr, fileElement.children, metaData, "initPolarAB", &dataElement->initPolarAB);
        TF_CENTERCELLPOLARITYIOFROMEASY(feItr, fileElement.children, metaData, "initPolarPCP", &dataElement->initPolarPCP);

        return S_OK;
    }

    template <>
    HRESULT toFile(const CPMod::PolarityVecsPack &dataElement, const MetaData &metaData, IOElement *fileElement) {

        IOElement *fe;

        TF_CENTERCELLPOLARITYIOTOEASY(fe, "v0", dataElement.v[0]);
        TF_CENTERCELLPOLARITYIOTOEASY(fe, "v1", dataElement.v[1]);
        TF_CENTERCELLPOLARITYIOTOEASY(fe, "v2", dataElement.v[2]);
        TF_CENTERCELLPOLARITYIOTOEASY(fe, "v3", dataElement.v[3]);
        TF_CENTERCELLPOLARITYIOTOEASY(fe, "v4", dataElement.v[4]);
        TF_CENTERCELLPOLARITYIOTOEASY(fe, "v5", dataElement.v[5]);

        fileElement->type = "PolarityVecs";

        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, CPMod::PolarityVecsPack *dataElement) {

        IOChildMap::const_iterator feItr;

        TF_CENTERCELLPOLARITYIOFROMEASY(feItr, fileElement.children, metaData, "v0", &dataElement->v[0]);
        TF_CENTERCELLPOLARITYIOFROMEASY(feItr, fileElement.children, metaData, "v1", &dataElement->v[1]);
        TF_CENTERCELLPOLARITYIOFROMEASY(feItr, fileElement.children, metaData, "v2", &dataElement->v[2]);
        TF_CENTERCELLPOLARITYIOFROMEASY(feItr, fileElement.children, metaData, "v3", &dataElement->v[3]);
        TF_CENTERCELLPOLARITYIOFROMEASY(feItr, fileElement.children, metaData, "v4", &dataElement->v[4]);
        TF_CENTERCELLPOLARITYIOFROMEASY(feItr, fileElement.children, metaData, "v5", &dataElement->v[5]);

        return S_OK;
    }

    template <>
    HRESULT toFile(const CPMod::ParticlePolarityPack &dataElement, const MetaData &metaData, IOElement *fileElement) {

        IOElement *fe;

        TF_CENTERCELLPOLARITYIOTOEASY(fe, "polarityVectors", dataElement.v);
        TF_CENTERCELLPOLARITYIOTOEASY(fe, "particleId", dataElement.pId);
        TF_CENTERCELLPOLARITYIOTOEASY(fe, "showing", dataElement.showing);

        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, CPMod::ParticlePolarityPack *dataElement) {

        IOChildMap::const_iterator feItr;

        TF_CENTERCELLPOLARITYIOFROMEASY(feItr, fileElement.children, metaData, "polarityVectors", &dataElement->v);
        TF_CENTERCELLPOLARITYIOFROMEASY(feItr, fileElement.children, metaData, "particleId", &dataElement->pId);
        TF_CENTERCELLPOLARITYIOFROMEASY(feItr, fileElement.children, metaData, "showing", &dataElement->showing);

        return S_OK;
    }

    template <>
    HRESULT toFile(const CPMod::PersistentForce &dataElement, const MetaData &metaData, IOElement *fileElement) {

        IOElement *fe;

        TF_CENTERCELLPOLARITYIOTOEASY(fileElement, "sensAB", dataElement.sensAB);
        TF_CENTERCELLPOLARITYIOTOEASY(fileElement, "sensPCP", dataElement.sensPCP);

        fileElement->type = "PersistentForce";

        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, CPMod::PersistentForce **dataElement) {

        IOChildMap::const_iterator feItr;

        FloatP_t sensAB, sensPCP;

        TF_CENTERCELLPOLARITYIOFROMEASY(feItr, fileElement.children, metaData, "sensAB", &sensAB);
        TF_CENTERCELLPOLARITYIOFROMEASY(feItr, fileElement.children, metaData, "sensPCP", &sensPCP);

        *dataElement = CPMod::createPersistentForce(sensAB, sensPCP);

        return S_OK;
    }

    template <>
    HRESULT toFile(const CPMod::ContactPotential &dataElement, const MetaData &metaData, IOElement *fileElement) {

        IOElement *fe;

        TF_CENTERCELLPOLARITYIOTOEASY(fe, "cutoff", dataElement.b);
        TF_CENTERCELLPOLARITYIOTOEASY(fe, "couplingFlat", dataElement.couplingFlat);
        TF_CENTERCELLPOLARITYIOTOEASY(fe, "couplingOrtho", dataElement.couplingOrtho);
        TF_CENTERCELLPOLARITYIOTOEASY(fe, "couplingLateral", dataElement.couplingLateral);
        TF_CENTERCELLPOLARITYIOTOEASY(fe, "distanceCoeff", dataElement.distanceCoeff);
        TF_CENTERCELLPOLARITYIOTOEASY(fe, "cType", (unsigned int)dataElement.cType);
        TF_CENTERCELLPOLARITYIOTOEASY(fe, "mag", dataElement.mag);
        TF_CENTERCELLPOLARITYIOTOEASY(fe, "rate", dataElement.rate);
        TF_CENTERCELLPOLARITYIOTOEASY(fe, "bendingCoeff", dataElement.bendingCoeff);

        fileElement->type = "ContactPotential";

        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, CPMod::ContactPotential **dataElement) {

        IOChildMap::const_iterator feItr;

        FloatP_t cutoff, couplingFlat, couplingOrtho, couplingLateral, distanceCoeff, mag, rate, bendingCoeff;
        TF_CENTERCELLPOLARITYIOFROMEASY(feItr, fileElement.children, metaData, "cutoff", &cutoff);
        TF_CENTERCELLPOLARITYIOFROMEASY(feItr, fileElement.children, metaData, "couplingFlat", &couplingFlat);
        TF_CENTERCELLPOLARITYIOFROMEASY(feItr, fileElement.children, metaData, "couplingOrtho", &couplingOrtho);
        TF_CENTERCELLPOLARITYIOFROMEASY(feItr, fileElement.children, metaData, "couplingLateral", &couplingLateral);
        TF_CENTERCELLPOLARITYIOFROMEASY(feItr, fileElement.children, metaData, "distanceCoeff", &distanceCoeff);
        
        unsigned int cTypeUI;
        TF_CENTERCELLPOLARITYIOFROMEASY(feItr, fileElement.children, metaData, "cType", &cTypeUI);
        CPMod::PolarContactType cType = (CPMod::PolarContactType)cTypeUI;
        std::string cTypeName = "regular";
        for(auto &cTypePair : CPMod::polarContactTypeMap) 
            if(cTypePair.second == cType) 
                cTypeName = cTypePair.first;

        TF_CENTERCELLPOLARITYIOFROMEASY(feItr, fileElement.children, metaData, "mag", &mag);
        TF_CENTERCELLPOLARITYIOFROMEASY(feItr, fileElement.children, metaData, "rate", &rate);
        TF_CENTERCELLPOLARITYIOFROMEASY(feItr, fileElement.children, metaData, "bendingCoeff", &bendingCoeff);

        *dataElement = CPMod::createContactPotential(cutoff, mag, rate, distanceCoeff, couplingFlat, couplingOrtho, couplingLateral, cTypeName, bendingCoeff);

        return S_OK;
    }

    HRESULT CellPolarityFIOModule::toFile(const MetaData &metaData, IOElement *fileElement) {

        IOElement *fe;

        // Store registered types

        if(CPMod::_polarityParams != NULL) {
            std::vector<CPMod::PolarityModelParams> polarityParams;
            std::vector<int32_t> polarTypes;
            for(auto &typePair : *CPMod::_polarityParams) {
                if(typePair.second != NULL) {
                    polarTypes.push_back(typePair.first);
                    polarityParams.push_back(*typePair.second);
                }
            }

            if(polarityParams.size() > 0) {
                TF_CENTERCELLPOLARITYIOTOEASY(fe, "polarTypes", polarTypes);
                TF_CENTERCELLPOLARITYIOTOEASY(fe, "polarityParams", polarityParams);
            }
        }

        // Store potentials
        
        if(CPMod::storedContactPotentials != NULL && CPMod::storedContactPotentials->size() > 0) {

            std::vector<CPMod::ContactPotential> potentials;
            auto numPots = CPMod::storedContactPotentials->size();
            std::vector<std::vector<uint16_t> > potentialTypesA(numPots, std::vector<uint16_t>()), potentialTypesB(numPots, std::vector<uint16_t>());
            std::vector<std::vector<uint16_t> > potentialTypesClusterA(numPots, std::vector<uint16_t>()), potentialTypesClusterB(numPots, std::vector<uint16_t>());

            for(unsigned int pi = 0; pi < numPots; pi++) 
                potentials.push_back(*(*CPMod::storedContactPotentials)[pi]);

            for(unsigned int i = 0; i < _Engine.nr_types; i++) {
                for(unsigned int j = i; j < _Engine.nr_types; j++) {
                    unsigned int k = i * _Engine.max_type + j;

                    Potential *p = _Engine.p[k], *pc = _Engine.p_cluster[k];

                    for(unsigned int pi = 0; pi < numPots; pi++) {
                        CPMod::ContactPotential *pp = (*CPMod::storedContactPotentials)[pi];
                        if(CPMod::recursivePotentialCompare(pp, p)) {
                            potentialTypesA[pi].push_back(i);
                            potentialTypesB[pi].push_back(j);
                        }
                        if(CPMod::recursivePotentialCompare(pp, pc)) {
                            potentialTypesClusterA[pi].push_back(i);
                            potentialTypesClusterB[pi].push_back(j);
                        }
                    }

                }
            }

            TF_CENTERCELLPOLARITYIOTOEASY(fe, "potentials", potentials);
            TF_CENTERCELLPOLARITYIOTOEASY(fe, "potentialTypesA", potentialTypesA);
            TF_CENTERCELLPOLARITYIOTOEASY(fe, "potentialTypesB", potentialTypesB);
            TF_CENTERCELLPOLARITYIOTOEASY(fe, "potentialTypesClusterA", potentialTypesClusterA);
            TF_CENTERCELLPOLARITYIOTOEASY(fe, "potentialTypesClusterB", potentialTypesClusterB);

        }

        // Store forces

        if(CPMod::storedPersistentForces != NULL && CPMod::storedPersistentForces->size() > 0) {
            std::vector<CPMod::PersistentForce> forces;
            std::vector<int16_t> forceTypes(CPMod::storedPersistentForces->size(), -1);

            for(unsigned int i = 0; i < CPMod::storedPersistentForces->size(); i++) 
                forces.push_back(*(*CPMod::storedPersistentForces)[i]);

            for(unsigned int i = 0; i < _Engine.nr_types; i++) 
                for(unsigned int fi = 0; fi < CPMod::storedPersistentForces->size(); fi++) 
                    if(CPMod::recursiveForceCompare((*CPMod::storedPersistentForces)[fi], _Engine.forces[i])) 
                        forceTypes[fi] = i;

            TF_CENTERCELLPOLARITYIOTOEASY(fe, "forces", forces);
            TF_CENTERCELLPOLARITYIOTOEASY(fe, "forceTypes", forceTypes);
        }
        
        // Store states of registered particles

        if(CPMod::_partPolPack != NULL && CPMod::_partPolPack->size() > 0) {
            std::vector<CPMod::ParticlePolarityPack> particles;
            for(auto &pp : *CPMod::_partPolPack) 
                if(pp) 
                    particles.push_back(*pp);

            TF_CENTERCELLPOLARITYIOTOEASY(fe, "particles", particles);
        }

        return S_OK;
    }

    HRESULT CellPolarityFIOModule::fromFile(const MetaData &metaData, const IOElement &fileElement) {

        IOChildMap::const_iterator feItr;

        // Load registered types

        if(fileElement.children.find("polarityParams") != fileElement.children.end()) {
        
            std::vector<CPMod::PolarityModelParams> polarityParams;
            std::vector<int32_t> polarTypes;

            TF_CENTERCELLPOLARITYIOFROMEASY(feItr, fileElement.children, metaData, "polarityParams", &polarityParams);
            TF_CENTERCELLPOLARITYIOFROMEASY(feItr, fileElement.children, metaData, "polarTypes", &polarTypes);

            for(unsigned int i = 0; i < polarTypes.size(); i++) {
                unsigned int pTypeId = io::FIO::importSummary->particleTypeIdMap[polarTypes[i]];
                auto pmp = polarityParams[i];

                ParticleType *pType = &_Engine.types[pTypeId];
                if(pType == NULL) {
                    tf_exp(std::runtime_error("Particle type not defined"));
                    return E_FAIL;
                }
                CPMod::registerType(pType, pmp.initMode, pmp.initPolarAB, pmp.initPolarPCP);
            }

        }

        // Load potentials

        if(fileElement.children.find("potentials") != fileElement.children.end()) {

            std::vector<CPMod::ContactPotential*> potentials;
            std::vector<std::vector<unsigned int> > potentialTypesA, potentialTypesB, potentialTypesClusterA, potentialTypesClusterB;

            TF_CENTERCELLPOLARITYIOFROMEASY(feItr, fileElement.children, metaData, "potentials", &potentials);
            TF_CENTERCELLPOLARITYIOFROMEASY(feItr, fileElement.children, metaData, "potentialTypesA", &potentialTypesA);
            TF_CENTERCELLPOLARITYIOFROMEASY(feItr, fileElement.children, metaData, "potentialTypesB", &potentialTypesB);
            TF_CENTERCELLPOLARITYIOFROMEASY(feItr, fileElement.children, metaData, "potentialTypesClusterA", &potentialTypesClusterA);
            TF_CENTERCELLPOLARITYIOFROMEASY(feItr, fileElement.children, metaData, "potentialTypesClusterB", &potentialTypesClusterB);

            for(unsigned int i = 0; i < potentials.size(); i++) {
                auto p = potentials[i];
                std::vector<unsigned int> pTypesIdA = potentialTypesA[i];
                std::vector<unsigned int> pTypesIdB = potentialTypesB[i];
                std::vector<unsigned int> pTypesIdClusterA = potentialTypesClusterA[i];
                std::vector<unsigned int> pTypesIdClusterB = potentialTypesClusterB[i];

                for(unsigned int j = 0; j < pTypesIdA.size(); j++) { 
                    ParticleType *pTypeA = &_Engine.types[io::FIO::importSummary->particleTypeIdMap[pTypesIdA[j]]];
                    ParticleType *pTypeB = &_Engine.types[io::FIO::importSummary->particleTypeIdMap[pTypesIdB[j]]];
                    if(!pTypeA || !pTypeB) {
                        tf_exp(std::runtime_error("Particle type not defined"));
                        return E_FAIL;
                    }
                    bind::types(p, pTypeA, pTypeB);
                }

                for(unsigned int j = 0; j < pTypesIdClusterA.size(); j++) { 
                    ParticleType *pTypeA = &_Engine.types[io::FIO::importSummary->particleTypeIdMap[pTypesIdClusterA[j]]];
                    ParticleType *pTypeB = &_Engine.types[io::FIO::importSummary->particleTypeIdMap[pTypesIdClusterB[j]]];
                    if(!pTypeA || !pTypeB) {
                        tf_exp(std::runtime_error("Particle type not defined"));
                        return E_FAIL;
                    }
                    bind::types(p, pTypeA, pTypeB, true);
                }
            }

        }

        // Load forces

        if(fileElement.children.find("forces") != fileElement.children.end()) {

            std::vector<CPMod::PersistentForce*> forces;
            std::vector<int16_t> forceTypes;

            TF_CENTERCELLPOLARITYIOFROMEASY(feItr, fileElement.children, metaData, "forces", &forces);
            TF_CENTERCELLPOLARITYIOFROMEASY(feItr, fileElement.children, metaData, "forceTypes", &forceTypes);

            for(unsigned int i = 0; i < forces.size(); i++) {
                auto pTypeIdImport = forceTypes[i];
                if(pTypeIdImport < 0) 
                    continue;

                auto pTypeId = io::FIO::importSummary->particleTypeIdMap[pTypeIdImport];

                ParticleType *pType = &_Engine.types[pTypeId];
                if(pType == NULL) {
                    tf_exp(std::runtime_error("Particle type not defined"));
                    return E_FAIL;
                }

                bind::force(forces[i], pType);
            }

        }
        
        // Load states of registered particles

        if(fileElement.children.find("particles") != fileElement.children.end()) {

            std::vector<CPMod::ParticlePolarityPack> particles;

            TF_CENTERCELLPOLARITYIOFROMEASY(feItr, fileElement.children, metaData, "particles", &particles);

            for(unsigned int i = 0; i < particles.size(); i++) {
                auto ppp = particles[i];
                auto pId = io::FIO::importSummary->particleIdMap[ppp.pId];
                CPMod::registerParticle(pId);
                auto pppe = (*CPMod::_partPolPack)[pId];
                if(ppp.showing) 
                    pppe->addArrows();

            }

        }

        return S_OK;
    }

};
