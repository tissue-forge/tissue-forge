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

#include <tfBoundaryConditions.h>
#include <tfSpace.h>
#include <tfEngine.h>
#include <tfLogger.h>
#include <tfError.h>
#include <io/tfFIO.h>
#include <state/tfStateVector.h>

#include <algorithm>
#include <string>
#include <cstring>
#include <unordered_map>
#include <tf_mdcore_io.h>


using namespace TissueForge;


std::unordered_map<unsigned int, std::string> boundaryConditionsEnumToNameMap{
    {BOUNDARY_FREESLIP, "FREESLIP"},
    {BOUNDARY_NO_SLIP, "NOSLIP"},
    {BOUNDARY_PERIODIC, "PERIODIC"},
    {BOUNDARY_POTENTIAL, "POTENTIAL"},
    {BOUNDARY_RESETTING, "RESET"},
    {BOUNDARY_VELOCITY, "VELOCITY"}
};

std::unordered_map<std::string, unsigned int> boundaryConditionsNameToEnumMap{
    {"FREESLIP", BOUNDARY_FREESLIP},
    {"FREE_SLIP", BOUNDARY_FREESLIP},
    {"NOSLIP", BOUNDARY_NO_SLIP},
    {"NO_SLIP", BOUNDARY_NO_SLIP},
    {"PERIODIC", BOUNDARY_PERIODIC},
    {"POTENTIAL", BOUNDARY_POTENTIAL},
    {"RESET", BOUNDARY_RESETTING},
    {"VELOCITY", BOUNDARY_VELOCITY}
};

std::string invalidBoundaryConditionsName = "INVALID";


/**
 * boundary was initialized from flags, set individual values
 */
static void boundaries_from_flags(BoundaryConditions *bc) {
    
    if(bc->periodic & space_periodic_x) {
        bc->left.kind = BOUNDARY_PERIODIC;
        bc->right.kind = BOUNDARY_PERIODIC;
    }
    else if(bc->periodic & SPACE_FREESLIP_X) {
        bc->left.kind = BOUNDARY_FREESLIP;
        bc->right.kind = BOUNDARY_FREESLIP;
    }
    
    if(bc->periodic & space_periodic_y) {
        bc->front.kind = BOUNDARY_PERIODIC;
        bc->back.kind = BOUNDARY_PERIODIC;
    }
    else if(bc->periodic & SPACE_FREESLIP_Y) {
        bc->front.kind = BOUNDARY_FREESLIP;
        bc->back.kind = BOUNDARY_FREESLIP;
    }
    
    if(bc->periodic & space_periodic_z) {
        bc->top.kind = BOUNDARY_PERIODIC;
        bc->bottom.kind = BOUNDARY_PERIODIC;
    }
    else if(bc->periodic & SPACE_FREESLIP_Z) {
        bc->top.kind = BOUNDARY_FREESLIP;
        bc->bottom.kind = BOUNDARY_FREESLIP;
    }
}

// check if valid type, and return, if string and invalid string, throw exception.
static unsigned bc_kind_from_string(const std::string &s) {
    TF_Log(LOG_DEBUG) << s;
    
    int result = 0;
    
    std::string _s = s;
    std::transform(_s.begin(), _s.end(), _s.begin(), ::toupper);

    auto itr = boundaryConditionsNameToEnumMap.find(_s);
    std::vector<std::string> validKindNames{
        "PERIODIC", "FREE_SLIP", "FREESLIP", "NO_SLIP", "NOSLIP", "POTENTIAL", "RESET"
    };
    if(itr!=boundaryConditionsNameToEnumMap.end()) {
        for (auto name : validKindNames) {
            if(_s.compare(name) == 0) {
                TF_Log(LOG_DEBUG) << name;

                if(_s.compare("POTENTIAL") == 0) return itr->second | boundaryConditionsNameToEnumMap["FREESLIP"];
                return itr->second;
            }
        }
    }
    
    std::string msg = "invalid choice of value for boundary condition, \"" + _s + "\"";
    msg += ", only the following are supported for cardinal direction init: ";
    for(auto name : validKindNames) msg += "\"" + name + "\" ";
    tf_exp(std::invalid_argument(msg));
    return 0;
}

static unsigned int bc_kind_from_strings(const std::vector<std::string> &kinds) {
    TF_Log(LOG_TRACE);

    int result = 0;

    for (auto k : kinds) result = result | bc_kind_from_string(k);
    
    return result;
}

static unsigned init_bc_direction(BoundaryCondition *low_bl, BoundaryCondition *high_bl, const unsigned &kind) {
    if(kind == BOUNDARY_NO_SLIP) {
        low_bl->kind = high_bl->kind = BOUNDARY_VELOCITY;
        low_bl->velocity = high_bl->velocity = FVector3{0.f, 0.f, 0.f};
    }
    else {
        low_bl->kind = (BoundaryConditionKind)kind;
        high_bl->kind = (BoundaryConditionKind)kind;
    }

    TF_Log(LOG_DEBUG) << low_bl->name << ": " << low_bl->kindStr();
    TF_Log(LOG_DEBUG) << high_bl->name << ": " << high_bl->kindStr();
    
    return kind;
}

unsigned BoundaryCondition::init(const unsigned &kind) {
    this->kind = (BoundaryConditionKind)kind;
    if(this->kind == BOUNDARY_NO_SLIP) {
        this->kind = BOUNDARY_VELOCITY;
        this->velocity = FVector3{};
    }

    TF_Log(LOG_DEBUG) << this->name << ": " << kindStr();

    return this->kind;
}

unsigned BoundaryCondition::init(const FVector3 &velocity, const FPTYPE *restore) {
    if(restore) this->restore = *restore;
    this->kind = BOUNDARY_VELOCITY;
    this->velocity = velocity;

    TF_Log(LOG_DEBUG) << this->name << ": " << kindStr();

    return this->kind;
}

unsigned BoundaryCondition::init(const std::unordered_map<std::string, unsigned int> vals, 
                                   const std::unordered_map<std::string, FVector3> vels, 
                                   const std::unordered_map<std::string, FPTYPE> restores) 
{
    auto itr = vals.find(this->name);
    if(itr != vals.end()) return init(itr->second);

    auto itrv = vels.find(this->name);
    auto itrr = restores.find(this->name);
    if(itrv != vels.end()) {
        auto a = itrr == restores.end() ? NULL : &itrr->second;
        return init(itrv->second, a);
    }

    TF_Log(LOG_DEBUG) << this->name << ": " << kindStr();

    return this->kind;
}

static void check_periodicy(BoundaryCondition *low_bc, BoundaryCondition *high_bc) {
    if((low_bc->kind & BOUNDARY_PERIODIC) ^ (high_bc->kind & BOUNDARY_PERIODIC)) {
        BoundaryCondition *has;
        BoundaryCondition *notHas;
    
        if(low_bc->kind & BOUNDARY_PERIODIC) {
            has = low_bc;
            notHas = high_bc;
        }
        else {
            has = high_bc;
            notHas = low_bc;
        }
        
        std::string msg = "only ";
        msg += has->name;
        msg += "has periodic boundary conditions set, but not ";
        msg += notHas->name;
        msg += ", setting both to periodic";
        
        low_bc->kind = BOUNDARY_PERIODIC;
        high_bc->kind = BOUNDARY_PERIODIC;
        
        TF_Log(LOG_INFORMATION) << msg.c_str();
    }
}

BoundaryConditions::BoundaryConditions(int *cells) {
    if(_initIni() != S_OK) return;
    
	TF_Log(LOG_INFORMATION) << "Initializing boundary conditions";

    this->periodic = space_periodic_full;
    boundaries_from_flags(this);
    
    _initFin(cells);
}

BoundaryConditions::BoundaryConditions(int *cells, const int &value) {
    if(_initIni() != S_OK) return;
    
    TF_Log(LOG_INFORMATION) << "Initializing boundary conditions by value: " << value;
    
    switch(value) {
        case space_periodic_none :
        case space_periodic_x:
        case space_periodic_y:
        case space_periodic_z:
        case space_periodic_full:
        case space_periodic_ghost_x:
        case space_periodic_ghost_y:
        case space_periodic_ghost_z:
        case space_periodic_ghost_full:
        case SPACE_FREESLIP_X:
        case SPACE_FREESLIP_Y:
        case SPACE_FREESLIP_Z:
        case SPACE_FREESLIP_FULL:
            TF_Log(LOG_INFORMATION) << "Processing as: SPACE_FREESLIP_FULL";

            this->periodic = value;
            break;
        default: {
            std::string msg = "invalid value " + std::to_string(value) + ", for integer boundary condition";
            tf_exp(std::invalid_argument(msg.c_str()));
            return;
        }
    }
    
    boundaries_from_flags(this);
    
    _initFin(cells);
}

void BoundaryConditions::_initDirections(const std::unordered_map<std::string, unsigned int> vals) {
    TF_Log(LOG_INFORMATION) << "Initializing boundary conditions by directions";

    unsigned dir;
    auto itr = vals.find("x");
    if(itr != vals.end()) {
        dir = init_bc_direction(&(this->left), &(this->right), itr->second);
        if(dir & BOUNDARY_PERIODIC) {
            this->periodic |= space_periodic_x;
        }
        if(dir & BOUNDARY_FREESLIP) {
            this->periodic |= SPACE_FREESLIP_X;
        }
    }

    itr = vals.find("y");
    if(itr != vals.end()) {
        dir = init_bc_direction(&(this->front), &(this->back), itr->second);
        if(dir & BOUNDARY_PERIODIC) {
            this->periodic |= space_periodic_y;
        }
        if(dir & BOUNDARY_FREESLIP) {
            this->periodic |= SPACE_FREESLIP_Y;
        }
    }

    itr = vals.find("z");
    if(itr != vals.end()) {
        dir = init_bc_direction(&(this->bottom), &(this->top), itr->second);
        if(dir & BOUNDARY_PERIODIC) {
            this->periodic |= space_periodic_z;
        }
        if(dir & BOUNDARY_FREESLIP) {
            this->periodic |= SPACE_FREESLIP_Z;
        }
    }
}

void BoundaryConditions::_initSides(const std::unordered_map<std::string, unsigned int> vals, 
                                      const std::unordered_map<std::string, FVector3> vels, 
                                      const std::unordered_map<std::string, FPTYPE> restores) 
{
    TF_Log(LOG_INFORMATION) << "Initializing boundary conditions by sides";

    unsigned dir;

    dir = this->left.init(vals, vels, restores);
    if(dir & BOUNDARY_PERIODIC) {
        this->periodic |= space_periodic_x;
    }
    if(dir & BOUNDARY_FREESLIP) {
        this->periodic |= SPACE_FREESLIP_X;
    }

    dir = this->right.init(vals, vels, restores);
    if(dir & BOUNDARY_PERIODIC) {
        this->periodic |= space_periodic_x;
    }
    if(dir & BOUNDARY_FREESLIP) {
        this->periodic |= SPACE_FREESLIP_X;
    }

    dir = this->front.init(vals, vels, restores);
    if(dir & BOUNDARY_PERIODIC) {
        this->periodic |= space_periodic_y;
    }
    if(dir & BOUNDARY_FREESLIP) {
        this->periodic |= SPACE_FREESLIP_Y;
    }

    dir = this->back.init(vals, vels, restores);
    if(dir & BOUNDARY_PERIODIC) {
        this->periodic |= space_periodic_y;
    }
    if(dir & BOUNDARY_FREESLIP) {
        this->periodic |= SPACE_FREESLIP_Y;
    }

    dir = this->top.init(vals, vels, restores);
    if(dir & BOUNDARY_PERIODIC) {
        this->periodic |= space_periodic_z;
    }
    if(dir & BOUNDARY_FREESLIP) {
        this->periodic |= SPACE_FREESLIP_Z;
    }

    dir = this->bottom.init(vals, vels, restores);
    if(dir & BOUNDARY_PERIODIC) {
        this->periodic |= space_periodic_z;
    }
    if(dir & BOUNDARY_FREESLIP) {
        this->periodic |= SPACE_FREESLIP_Z;
    }
}

BoundaryConditions::BoundaryConditions(int *cells, 
                                           const std::unordered_map<std::string, unsigned int> vals, 
                                           const std::unordered_map<std::string, FVector3> vels, 
                                           const std::unordered_map<std::string, FPTYPE> restores) 
{
    if(_initIni() != S_OK) return;

    TF_Log(LOG_INFORMATION) << "Initializing boundary conditions by values";

    _initDirections(vals);
    _initSides(vals, vels, restores);

    check_periodicy(&(this->left), &(this->right));
    check_periodicy(&(this->front), &(this->back));
    check_periodicy(&(this->top), &(this->bottom));
    
    _initFin(cells);
}

// Initializes bc initialization, independently of what all was specified
HRESULT BoundaryConditions::_initIni() {
    TF_Log(LOG_INFORMATION) << "Initializing boundary conditions initialization";

    bzero(this, sizeof(BoundaryConditions));

    this->potenntials = (Potential**)malloc(6 * engine::max_type * sizeof(Potential*));
    bzero(this->potenntials, 6 * engine::max_type * sizeof(Potential*));

    this->left.kind = BOUNDARY_PERIODIC;
    this->right.kind = BOUNDARY_PERIODIC;
    this->front.kind = BOUNDARY_PERIODIC;
    this->back.kind = BOUNDARY_PERIODIC;
    this->bottom.kind = BOUNDARY_PERIODIC;
    this->top.kind = BOUNDARY_PERIODIC;

    this->left.name = "left";     this->left.restore = 1.f;     this->left.potenntials =   &this->potenntials[0 * engine::max_type];
    this->right.name = "right";   this->right.restore = 1.f;    this->right.potenntials =  &this->potenntials[1 * engine::max_type];
    this->front.name = "front";   this->front.restore = 1.f;    this->front.potenntials =  &this->potenntials[2 * engine::max_type];
    this->back.name = "back";     this->back.restore = 1.f;     this->back.potenntials =   &this->potenntials[3 * engine::max_type];
    this->top.name = "top";       this->top.restore = 1.f;      this->top.potenntials =    &this->potenntials[4 * engine::max_type];
    this->bottom.name = "bottom"; this->bottom.restore = 1.f;   this->bottom.potenntials = &this->potenntials[5 * engine::max_type];
    
    this->left.normal =   { 1.f,  0.f,  0.f};
    this->right.normal =  {-1.f,  0.f,  0.f};
    this->front.normal =  { 0.f,  1.f,  0.f};
    this->back.normal =   { 0.f, -1.f,  0.f};
    this->bottom.normal = { 0.f,  0.f,  1.f};
    this->top.normal =    { 0.f,  0.f, -1.f};

    return S_OK;
}

// Finalizes bc initialization, independently of what all was specified
HRESULT BoundaryConditions::_initFin(int *cells) {
    TF_Log(LOG_INFORMATION) << "Finalizing boundary conditions initialization";

    if(cells[0] < 3 && (this->periodic & space_periodic_x)) {
        cells[0] = 3;
        std::string msg = "requested periodic_x and " + std::to_string(cells[0]) +
        " space cells in the x direction, need at least 3 cells for periodic, setting cell count to 3";
        tf_exp(std::invalid_argument(msg.c_str()));
        return E_INVALIDARG;
    }
    if(cells[1] < 3 && (this->periodic & space_periodic_y)) {
        cells[1] = 3;
        std::string msg = "requested periodic_x and " + std::to_string(cells[1]) +
        " space cells in the x direction, need at least 3 cells for periodic, setting cell count to 3";
        tf_exp(std::invalid_argument(msg.c_str()));
        return E_INVALIDARG;
    }
    if(cells[2] < 3 && (this->periodic & space_periodic_z)) {
        cells[2] = 3;
        std::string msg = "requested periodic_x and " + std::to_string(cells[2]) +
        " space cells in the x direction, need at least 3 cells for periodic, setting cell count to 3";
        tf_exp(std::invalid_argument(msg.c_str()));
        return E_INVALIDARG;
    }
    
    TF_Log(LOG_INFORMATION) << "engine periodic x : " << (bool)(this->periodic & space_periodic_x) ;
    TF_Log(LOG_INFORMATION) << "engine periodic y : " << (bool)(this->periodic & space_periodic_y) ;
    TF_Log(LOG_INFORMATION) << "engine periodic z : " << (bool)(this->periodic & space_periodic_z) ;
    TF_Log(LOG_INFORMATION) << "engine freeslip x : " << (bool)(this->periodic & SPACE_FREESLIP_X) ;
    TF_Log(LOG_INFORMATION) << "engine freeslip y : " << (bool)(this->periodic & SPACE_FREESLIP_Y) ;
    TF_Log(LOG_INFORMATION) << "engine freeslip z : " << (bool)(this->periodic & SPACE_FREESLIP_Z) ;
    TF_Log(LOG_INFORMATION) << "engine periodic ghost x : " << (bool)(this->periodic & space_periodic_ghost_x) ;
    TF_Log(LOG_INFORMATION) << "engine periodic ghost y : " << (bool)(this->periodic & space_periodic_ghost_y) ;
    TF_Log(LOG_INFORMATION) << "engine periodic ghost z : " << (bool)(this->periodic & space_periodic_ghost_z) ;
    
    return S_OK;
}

unsigned BoundaryConditions::boundaryKindFromString(const std::string &s) {
    return bc_kind_from_string(s);
}

unsigned BoundaryConditions::boundaryKindFromStrings(const std::vector<std::string> &kinds) {
    return bc_kind_from_strings(kinds);
}

std::string BoundaryConditions::str() {
    std::string s = "BoundaryConditions(\n";
    s += "  " + left.str(false) + ", \n";
    s += "  " + right.str(false) + ", \n";
    s += "  " + front.str(false) + ", \n";
    s += "  " + back.str(false) + ", \n";
    s += "  " + bottom.str(false) + ", \n";
    s += "  " + top.str(false) + ", \n";
    s += ")";
    return s;
}

#if defined(HAVE_CUDA)
static bool boundary_conditions_cuda_defer_update = false;
#endif

void BoundaryCondition::set_potential(struct ParticleType *ptype,
        struct Potential *pot)
{
    potenntials[ptype->id] = pot;

    #if defined(HAVE_CUDA)
    if(!boundary_conditions_cuda_defer_update && _Engine.flags & engine_flag_cuda)
        cuda::engine_cuda_boundary_conditions_refresh(&_Engine);
    #endif
}

std::string BoundaryCondition::kindStr() const {
    std::string s = "";
    bool foundEntries = false;

    for(auto &itr : boundaryConditionsEnumToNameMap) {
        if(this->kind & itr.first) {
            if(foundEntries) s += ", " + itr.second;
            else s = itr.second;

            foundEntries = true;
        }
    }

    if(!foundEntries) s = invalidBoundaryConditionsName;

    return s;
}

std::string BoundaryCondition::str(bool show_type) const
{
    std::string s;
    
    if(show_type) {
        s +=  "BoundaryCondition(";
    }
    
    s += "\'";
    s += this->name;
    s += "\' : {";
    
    s += "\'kind\' : \'";
    s += kindStr();
    s += "\'";
    s += ", \'velocity\' : [" + std::to_string(velocity[0]) + ", " + std::to_string(velocity[1]) + ", " + std::to_string(velocity[2]) + "]";
    s += ", \'restore\' : " + std::to_string(restore);
    s += "}";
    
    if(show_type) {
        s +=  ")";
    }
    
    return s;
}

void BoundaryConditions::set_potential(struct ParticleType *ptype,
        struct Potential *pot)
{
    #if defined(HAVE_CUDA)
    boundary_conditions_cuda_defer_update = _Engine.flags & engine_flag_cuda;
    #endif

    left.set_potential(ptype, pot);
    right.set_potential(ptype, pot);
    front.set_potential(ptype, pot);
    back.set_potential(ptype, pot);
    bottom.set_potential(ptype, pot);
    top.set_potential(ptype, pot);

    #if defined(HAVE_CUDA)
    if(_Engine.flags & engine_flag_cuda) {
        boundary_conditions_cuda_defer_update = false;
        cuda::engine_cuda_boundary_conditions_refresh(&_Engine);
    }
    #endif
}

void BoundaryConditionsArgsContainer::setValueAll(const int &_bcValue) {
    TF_Log(LOG_INFORMATION) << std::to_string(_bcValue);
    
    switchType(true);
    *bcValue = _bcValue;
}

void BoundaryConditionsArgsContainer::setValue(const std::string &name, const unsigned int &value) {
    TF_Log(LOG_INFORMATION) << name << ", " << std::to_string(value);

    switchType(false);
    (*bcVals)[name] = value;
}

void BoundaryConditionsArgsContainer::setVelocity(const std::string &name, const FVector3 &velocity) {
    TF_Log(LOG_INFORMATION) << name << ", " << std::to_string(velocity.x()) << ", " << std::to_string(velocity.y()) << ", " << std::to_string(velocity.z());

    switchType(false);
    (*bcVels)[name] = velocity;
}

void BoundaryConditionsArgsContainer::setRestore(const std::string &name, const FPTYPE restore) {
    TF_Log(LOG_INFORMATION) << name << ", " << std::to_string(restore);

    switchType(false);
    (*bcRestores)[name] = restore;
}

BoundaryConditions *BoundaryConditionsArgsContainer::create(int *cells) {
    BoundaryConditions *result;

    if(bcValue) {
        TF_Log(LOG_INFORMATION) << "Creating boundary conditions by value."; 
        result = new BoundaryConditions(cells, *bcValue);
    }
    else if(bcVals) {
        TF_Log(LOG_INFORMATION) << "Creating boundary conditions by values"; 
        result = new BoundaryConditions(cells, *bcVals, *bcVels, *bcRestores);
    }
    else {
        TF_Log(LOG_INFORMATION) << "Creating boundary conditions by defaults";
        result = new BoundaryConditions(cells);
    }
    return result;
}

BoundaryConditionsArgsContainer::BoundaryConditionsArgsContainer(int *_bcValue, 
                                                                     std::unordered_map<std::string, unsigned int> *_bcVals, 
                                                                     std::unordered_map<std::string, FVector3> *_bcVels, 
                                                                     std::unordered_map<std::string, FPTYPE> *_bcRestores) : 
    bcValue(nullptr), bcVals(nullptr), bcVels(nullptr), bcRestores(nullptr)
{
    if(_bcValue) setValueAll(*_bcValue);
    else {
        if(_bcVals)
            for(auto &itr : *_bcVals)
                setValue(itr.first, itr.second);
        if(_bcVels)
            for(auto &itr : *_bcVels)
                setVelocity(itr.first, itr.second);
        if(_bcRestores)
            for(auto &itr : *_bcRestores)
                setRestore(itr.first, itr.second);
    }
}

void BoundaryConditionsArgsContainer::switchType(const bool &allSides) {
    if(allSides) {
        if(bcVals) {
            delete bcVals;
            bcVals = NULL;
        }
        if(bcVels) {
            delete bcVels;
            bcVels = NULL;
        }
        if(bcRestores) {
            delete bcRestores;
            bcRestores = NULL;
        }

        if(!bcValue) bcValue = new int(BOUNDARY_PERIODIC);
    }
    else {
        if(bcValue) {
            delete bcValue;
            bcValue = NULL;
        }

        if(!bcVals) bcVals = new std::unordered_map<std::string, unsigned int>();
        if(!bcVels) bcVels = new std::unordered_map<std::string, FVector3>();
        if(!bcRestores) bcRestores = new std::unordered_map<std::string, FPTYPE>();
    }
}

void TissueForge::apply_boundary_particle_crossing(
    struct Particle *p, 
    const int *delta,
    const struct space_cell *src_cell, 
    const struct space_cell *dest_cell) 
{
    
    const BoundaryConditions &bc = _Engine.boundary_conditions;

    if(!p->state_vector) 
        return;

    bool didReset = false;
    if(src_cell->loc[0] != dest_cell->loc[0]) {
        if(bc.periodic & space_periodic_x &&
            src_cell->flags & cell_periodic_x && dest_cell->flags & cell_periodic_x) 
        {
            
            if((dest_cell->flags  & cell_periodic_left  && bc.left.kind  & BOUNDARY_RESETTING) || 
                (dest_cell->flags & cell_periodic_right && bc.right.kind & BOUNDARY_RESETTING)) 
            {
                p->state_vector->reset();
                didReset = true;
            }
            
        }
    }

    if(!didReset && src_cell->loc[1] != dest_cell->loc[1]) {
        if(bc.periodic & space_periodic_y &&
            src_cell->flags & cell_periodic_y && dest_cell->flags & cell_periodic_y) 
        {
            
            if((dest_cell->flags  & cell_periodic_front && bc.front.kind & BOUNDARY_RESETTING) || 
                (dest_cell->flags & cell_periodic_back  && bc.back.kind  & BOUNDARY_RESETTING)) 
            {
                p->state_vector->reset();
                didReset = true;
            }

        }
    } 
    
    if(!didReset && src_cell->loc[2] != dest_cell->loc[2]) {
        if(bc.periodic & space_periodic_z &&
            src_cell->flags & cell_periodic_z && dest_cell->flags & cell_periodic_z) 
        {
            
            if((dest_cell->flags  & cell_periodic_top    && bc.top.kind    & BOUNDARY_RESETTING) ||
                (dest_cell->flags & cell_periodic_bottom && bc.bottom.kind & BOUNDARY_RESETTING)) 
            {
                p->state_vector->reset();
                
            }

        }
    }
}

std::string BoundaryConditions::toString() {
    return io::toString(*this);
}

BoundaryConditions *BoundaryConditions::fromString(const std::string &str) {
    return new BoundaryConditions(io::fromString<BoundaryConditions>(str));
}


namespace TissueForge::io {


    template <>
    HRESULT toFile(const BoundaryCondition &dataElement, const MetaData &metaData, IOElement &fileElement) {

        TF_IOTOEASY(fileElement, metaData, "kind", (int)dataElement.kind);
        TF_IOTOEASY(fileElement, metaData, "id", dataElement.id);
        if(dataElement.kind & BOUNDARY_VELOCITY) {
            TF_IOTOEASY(fileElement, metaData, "velocity", dataElement.velocity);
        }
        TF_IOTOEASY(fileElement, metaData, "restore", dataElement.restore);
        TF_IOTOEASY(fileElement, metaData, "name", std::string(dataElement.name));
        TF_IOTOEASY(fileElement, metaData, "normal", dataElement.normal);

        std::vector<unsigned int> potentialIndices;
        std::vector<Potential*> potentials;
        Potential *pot;
        for(unsigned int i = 0; i < engine::max_type; i++) {
            pot = dataElement.potenntials[i];
            if(pot != NULL) {
                potentialIndices.push_back(i);
                potentials.push_back(pot);
            }
        }
        if(potentialIndices.size() > 0) {
            TF_IOTOEASY(fileElement, metaData, "potentialIndices", potentialIndices);
            TF_IOTOEASY(fileElement, metaData, "potentials", potentials);
        }

        TF_IOTOEASY(fileElement, metaData, "radius", dataElement.radius);

        fileElement.get()->type = "boundaryCondition";

        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, BoundaryCondition *dataElement) {

        unsigned int kind;
        TF_IOFROMEASY(fileElement, metaData, "kind", &kind);
        dataElement->kind = (BoundaryConditionKind)kind;

        TF_IOFROMEASY(fileElement, metaData, "id", &dataElement->id);

        IOChildMap fec = IOElement::children(fileElement);

        if(fec.find("velocity") != fec.end()) {
            FVector3 velocity;
            TF_IOFROMEASY(fileElement, metaData, "velocity", &velocity);
            dataElement->velocity = velocity;
        }

        TF_IOFROMEASY(fileElement, metaData, "restore", &dataElement->restore);

        std::string name;
        TF_IOFROMEASY(fileElement, metaData, "name", &name);
        char *cname = new char[name.size() + 1];
        std::strcpy(cname, name.c_str());
        dataElement->name = cname;

        TF_IOFROMEASY(fileElement, metaData, "normal", &dataElement->normal);

        if(fec.find("potentials") != fec.end()) {

            std::vector<unsigned int> potentialIndices;
            std::vector<Potential*> potentials;
            TF_IOFROMEASY(fileElement, metaData, "potentialIndices", &potentialIndices);
            TF_IOFROMEASY(fileElement, metaData, "potentials", &potentials);

            if(potentials.size() > 0) 
                for(unsigned int i = 0; i < potentials.size(); i++) 
                    dataElement->potenntials[potentialIndices[i]] = potentials[i];

        }

        TF_IOFROMEASY(fileElement, metaData, "radius", &dataElement->radius);

        return S_OK;
    }

    template <>
    HRESULT toFile(const BoundaryConditions &dataElement, const MetaData &metaData, IOElement &fileElement) {

        TF_IOTOEASY(fileElement, metaData, "top", dataElement.top);
        TF_IOTOEASY(fileElement, metaData, "bottom", dataElement.bottom);
        TF_IOTOEASY(fileElement, metaData, "left", dataElement.left);
        TF_IOTOEASY(fileElement, metaData, "right", dataElement.right);
        TF_IOTOEASY(fileElement, metaData, "front", dataElement.front);
        TF_IOTOEASY(fileElement, metaData, "back", dataElement.back);

        fileElement.get()->type = "boundaryConditions";
        
        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, BoundaryConditions *dataElement) {

        // Initialize potential arrays
        // todo: implement automatic initialization of potential arrays in boundary conditions under all circumstances

        dataElement->potenntials = (Potential**)malloc(6 * engine::max_type * sizeof(Potential*));
        bzero(dataElement->potenntials, 6 * engine::max_type * sizeof(Potential*));

        dataElement->left.potenntials =   &dataElement->potenntials[0 * engine::max_type];
        dataElement->right.potenntials =  &dataElement->potenntials[1 * engine::max_type];
        dataElement->front.potenntials =  &dataElement->potenntials[2 * engine::max_type];
        dataElement->back.potenntials =   &dataElement->potenntials[3 * engine::max_type];
        dataElement->top.potenntials =    &dataElement->potenntials[4 * engine::max_type];
        dataElement->bottom.potenntials = &dataElement->potenntials[5 * engine::max_type];

        TF_IOFROMEASY(fileElement, metaData, "top",    &dataElement->top);
        TF_IOFROMEASY(fileElement, metaData, "bottom", &dataElement->bottom);
        TF_IOFROMEASY(fileElement, metaData, "left",   &dataElement->left);
        TF_IOFROMEASY(fileElement, metaData, "right",  &dataElement->right);
        TF_IOFROMEASY(fileElement, metaData, "front",  &dataElement->front);
        TF_IOFROMEASY(fileElement, metaData, "back",   &dataElement->back);

        return S_OK;
    }

    #define TF_MDCARGSIOFROMEASY(side) \
        feItr = fec.find(side); \
        if(feItr == fec.end())  \
            return E_FAIL; \
        fe = feItr->second; \
        TF_IOFROMEASY(fe, metaData, "kind", &kind); \
        bcVals[side] = kind; \
        TF_IOFROMEASY(fe, metaData, "restore", &restore); \
        bcRestores[side] = restore; \
        if((BoundaryConditionKind)kind & BOUNDARY_VELOCITY) { \
            TF_IOFROMEASY(fe, metaData, "velocity", &velocity); \
            bcVels["velocity"] = velocity; \
        }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, BoundaryConditionsArgsContainer *dataElement) {

        IOChildMap::const_iterator feItr;
        IOElement fe;
        unsigned int kind;
        std::string side;
        FVector3 velocity;
        FPTYPE restore;
        IOChildMap fec = IOElement::children(fileElement);

        std::unordered_map<std::string, unsigned int> bcVals; 
        std::unordered_map<std::string, FVector3> bcVels; 
        std::unordered_map<std::string, FPTYPE> bcRestores;

        TF_MDCARGSIOFROMEASY("top");
        TF_MDCARGSIOFROMEASY("bottom");
        TF_MDCARGSIOFROMEASY("left");
        TF_MDCARGSIOFROMEASY("right");
        TF_MDCARGSIOFROMEASY("front");
        TF_MDCARGSIOFROMEASY("back");

        dataElement = new BoundaryConditionsArgsContainer(0, &bcVals, &bcVels, &bcRestores);

        return S_OK;
    }

};
