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

// todo: parallelize execution

#include "tfMeshSolver.h"

#include "tfMeshObj.h"
#include "tfMeshRenderer.h"
#include "tfVertexSolverFIO.h"

#include <tfEngine.h>
#include <tf_util.h>
#include <tfLogger.h>
#include <tfTaskScheduler.h>
#include <io/tfFIO.h>

#include <atomic>
#include <future>
#include <typeinfo>


#define TF_MESHSOLVER_CHECKINIT_RET(retval) { if(!_solver) return retval; }

#define TF_MESHSOLVER_CHECKINIT { TF_MESHSOLVER_CHECKINIT_RET(E_FAIL) }


static std::mutex _meshEngineLock;
static std::future<std::vector<unsigned int> > fut_surfaceVertexIndices;


using namespace TissueForge;
using namespace TissueForge::models::vertex;

static MeshSolver *_solver = NULL;

static TissueForge::models::vertex::io::VertexSolverFIOModule *_ioModule = NULL;


HRESULT TissueForge::models::vertex::VertexForce(const Vertex *v, FloatP_t *f) {
    // Surfaces
    for(auto &s : v->getSurfaces()) {
        for(auto &a : s->type()->actors) 
            a->force(s, v, f);
        
        for(auto &a : s->actors) 
            a->force(s, v, f);
    }

    // Bodies
    for(auto &b : v->getBodies()) {
        for(auto &a : b->type()->actors) 
            a->force(b, v, f);

        for(auto &a : b->actors) 
            a->force(b, v, f);
    }

    // Structures
    for(auto &st : v->getStructures()) {
        for(auto &a : st->type()->actors) 
            a->force(st, v, f);

        for(auto &a : st->actors) 
            a->force(st, v, f);
    }

    return S_OK;
}

double MeshSolverTimers::ms(const Section &section, const bool &avg) const {
    double val = timers[section];
    return avg ? val / (_Engine.time * CLOCKS_PER_SEC) : val;
}

std::string MeshSolverTimers::str() const {
    std::stringstream ss;
    ss << "Force: " << ms(Section::FORCE) << ", " << std::endl;
    ss << "Advance: " << ms(Section::ADVANCE) << ", " << std::endl;
    ss << "Update: " << ms(Section::UPDATE) << ", " << std::endl;
    ss << "Quality: " << ms(Section::QUALITY) << std::endl;
    ss << "Rendering: " << ms(Section::RENDERING) << std::endl;
    return ss.str();
}

MeshSolverTimerInstance::MeshSolverTimerInstance(const MeshSolverTimers::Section &_section) : 
    section{_section}, 
    tic{getticks()}
{}

MeshSolverTimerInstance::~MeshSolverTimerInstance() {
    if(_solver) 
        _solver->timers.append(section, getticks() - tic);
}

HRESULT MeshSolver::init() {
    if(_solver != NULL) 
        return S_OK;

    if(!_ioModule) {
        _ioModule = new TissueForge::models::vertex::io::VertexSolverFIOModule();
        _ioModule->registerIOModule();
    }

    _solver = new MeshSolver();
    _solver->mesh = new Mesh();

    if(TissueForge::io::FIO::currentRootElement) 
        _ioModule->load();

    _solver->_bufferSize = 1;
    _solver->_forces = (FloatP_t*)malloc(3 * sizeof(FloatP_t));
    _solver->timers.reset();
    _solver->registerEngine();

    // Launches and registers renderer
    MeshRenderer::get();

    return S_OK;
}

MeshSolver *MeshSolver::get() { 
    if(_solver == NULL) 
        if(init() != S_OK) 
            return NULL;
    return _solver;
}

HRESULT MeshSolver::_compactInst() { 
    TF_MESHSOLVER_CHECKINIT

    if(_solver->_bufferSize > 1) {
        free(_forces);
        _bufferSize = 1;
        _forces = (FloatP_t*)malloc(3 * sizeof(FloatP_t));
    }

    return S_OK;
}

HRESULT MeshSolver::compact() { 
    TF_MESHSOLVER_CHECKINIT

    return _solver->_compactInst();
}

HRESULT MeshSolver::engineLock() {
    TF_MESHSOLVER_CHECKINIT

    _solver->_engineLock.lock();
    return S_OK;
}

HRESULT MeshSolver::engineUnlock() {
    TF_MESHSOLVER_CHECKINIT

    _solver->_engineLock.unlock();
    return S_OK;
}

Mesh *MeshSolver::getMesh() {
    TF_MESHSOLVER_CHECKINIT_RET(0)

    return _solver->mesh;
}

template <typename T> 
static bool MeshSolver_assignUniqueNameAsNecessary(T *inst, std::vector<T*> registeredInsts) {
    bool uniqueName = true;
    for(auto &ri : registeredInsts) 
        if(strcmp(inst->name.c_str(), ri->name.c_str()) == 0) 
            uniqueName = false;

    if(uniqueName) 
        return false;

    inst->name = typeid(*inst).name();
    return true;
}

HRESULT MeshSolver::_registerTypeInst(StructureType *_type) {
    if(!_type || _type->id >= 0) 
        return E_FAIL;

    _type->id = _structureTypes.size();
    if(MeshSolver_assignUniqueNameAsNecessary(_type, _structureTypes)) 
        TF_Log(LOG_INFORMATION) << "Type name not unique. Generating name: " << _type->name.c_str();
    _structureTypes.push_back(_type);

    return S_OK;
}

HRESULT MeshSolver::_registerTypeInst(BodyType *_type) {
    if(!_type || _type->id >= 0) 
        return E_FAIL;
    
    _type->id = _bodyTypes.size();
    if(MeshSolver_assignUniqueNameAsNecessary(_type, _bodyTypes)) 
        TF_Log(LOG_INFORMATION) << "Type name not unique. Generating name: " << _type->name.c_str();
    _bodyTypes.push_back(_type);

    return S_OK;
}

HRESULT MeshSolver::registerType(StructureType *_type) {
    TF_MESHSOLVER_CHECKINIT

    return _solver->_registerTypeInst(_type);
}

HRESULT MeshSolver::registerType(BodyType *_type) {
    TF_MESHSOLVER_CHECKINIT

    return _solver->_registerTypeInst(_type);
}

HRESULT MeshSolver::_registerTypeInst(SurfaceType *_type) {
    if(!_type || _type->id >= 0) 
        return E_FAIL;

    _type->id = _surfaceTypes.size();
    if(!_type->style) {
        auto colors = color3Names();
        auto c = colors[(_surfaceTypes.size() - 1) % colors.size()];
        _type->style = new rendering::Style(c);
    }
    if(MeshSolver_assignUniqueNameAsNecessary(_type, _surfaceTypes)) 
        TF_Log(LOG_INFORMATION) << "Type name not unique. Generating name: " << _type->name.c_str();
    _surfaceTypes.push_back(_type);

    return S_OK;
}

template <typename T> 
T *MeshSolver_findTypeFromName(const std::string name, std::vector<T*> types) {
    for(auto &t : types) 
        if(strcmp(name.c_str(), t->name.c_str()) == 0) 
            return t;
    return NULL;
}

SurfaceType *MeshSolver::_findSurfaceFromNameInst(const std::string &_name) {
    return MeshSolver_findTypeFromName(_name, _surfaceTypes);
}

BodyType *MeshSolver::_findBodyFromNameInst(const std::string &_name) {
    return MeshSolver_findTypeFromName(_name, _bodyTypes);
}

StructureType *MeshSolver::_findStructureFromNameInst(const std::string &_name) {
    return MeshSolver_findTypeFromName(_name, _structureTypes);
}

HRESULT MeshSolver::registerType(SurfaceType *_type) {
    TF_MESHSOLVER_CHECKINIT

    return _solver->_registerTypeInst(_type);
}

SurfaceType *MeshSolver::findSurfaceFromName(const std::string &_name) {
    TF_MESHSOLVER_CHECKINIT_RET(NULL)

    return _solver->_findSurfaceFromNameInst(_name);
}

BodyType *MeshSolver::findBodyFromName(const std::string &_name) {
    TF_MESHSOLVER_CHECKINIT_RET(NULL)

    return _solver->_findBodyFromNameInst(_name);
}

StructureType *MeshSolver::findStructureFromName(const std::string &_name) {
    TF_MESHSOLVER_CHECKINIT_RET(NULL)

    return _solver->_findStructureFromNameInst(_name);
}

StructureType *MeshSolver::_getStructureTypeInst(const unsigned int &typeId) const {
    if(typeId >= _structureTypes.size()) 
        return NULL;
    return _structureTypes[typeId];
}

StructureType *MeshSolver::getStructureType(const unsigned int &typeId) {
    TF_MESHSOLVER_CHECKINIT_RET(NULL);

    return _solver->_getStructureTypeInst(typeId);
}

BodyType *MeshSolver::_getBodyTypeInst(const unsigned int &typeId) const {
    if(typeId >= _bodyTypes.size()) 
        return NULL;
    return _bodyTypes[typeId];
}

BodyType *MeshSolver::getBodyType(const unsigned int &typeId) {
    TF_MESHSOLVER_CHECKINIT_RET(NULL);

    return _solver->_getBodyTypeInst(typeId);
}

SurfaceType *MeshSolver::_getSurfaceTypeInst(const unsigned int &typeId) const {
    if(typeId >= _surfaceTypes.size()) 
        return NULL;
    return _surfaceTypes[typeId];
}

SurfaceType *MeshSolver::getSurfaceType(const unsigned int &typeId) {
    TF_MESHSOLVER_CHECKINIT_RET(NULL);

    return _solver->_getSurfaceTypeInst(typeId);
}

const int MeshSolver::numStructureTypes() {
    TF_MESHSOLVER_CHECKINIT_RET(-1)

    return _solver->_structureTypes.size();
}

const int MeshSolver::numBodyTypes() {
    TF_MESHSOLVER_CHECKINIT_RET(-1)

    return _solver->_bodyTypes.size();
}

const int MeshSolver::numSurfaceTypes() {
    TF_MESHSOLVER_CHECKINIT_RET(-1)

    return _solver->_surfaceTypes.size();
}

unsigned int MeshSolver::numVertices() {
    TF_MESHSOLVER_CHECKINIT_RET(0)

    return _solver->mesh->numVertices();
}

unsigned int MeshSolver::numSurfaces() {
    TF_MESHSOLVER_CHECKINIT_RET(0)

    return _solver->mesh->numSurfaces();
}

unsigned int MeshSolver::numBodies() {
    TF_MESHSOLVER_CHECKINIT_RET(0)

    return _solver->mesh->numBodies();
}

unsigned int MeshSolver::numStructures() {
    TF_MESHSOLVER_CHECKINIT_RET(0)

    return _solver->mesh->numStructures();
}

unsigned int MeshSolver::sizeVertices() {
    TF_MESHSOLVER_CHECKINIT_RET(0)

    return _solver->mesh->sizeVertices();
}

unsigned int MeshSolver::sizeSurfaces() {
    TF_MESHSOLVER_CHECKINIT_RET(0)

    return _solver->mesh->sizeSurfaces();
}

unsigned int MeshSolver::sizeBodies() {
    TF_MESHSOLVER_CHECKINIT_RET(0)

    return _solver->mesh->sizeBodies();
}

unsigned int MeshSolver::sizeStructures() {
    TF_MESHSOLVER_CHECKINIT_RET(0)

    return _solver->mesh->sizeStructures();
}

template <typename T> 
void Mesh_actRecursive(MeshObj *vertex, T *source, FloatP_t *f) {
    for(auto &a : source->type()->actors) 
        a->force(source, vertex, f);
    for(auto &c : source->children()) 
        Mesh_actRecursive(vertex, (T*)c, f);
}

HRESULT MeshSolver::_positionChangedInst() {

    unsigned int i;
    _surfaceVertices = 0;
    _totalVertices = 0;

    static int stride = ThreadPool::size();

    // Update vertices

    std::vector<Vertex*> &m_vertices = mesh->vertices;
    auto func_vertices = [&m_vertices](int i) -> void {
        Vertex *v = m_vertices[i];
        if(v) 
            v->positionChanged();
    };
    parallel_for(m_vertices.size(), func_vertices);
    _totalVertices += mesh->numVertices();

    // Update surfaces
    
    std::vector<Surface*> &m_surfaces = mesh->surfaces;
    std::atomic<unsigned int> _surfaceVertices_m = 0;
    auto func_surfaces = [&m_surfaces, &_surfaceVertices_m](int tid) -> void {
        unsigned int _surfaceVertices_local = 0;
        for(int i = tid; i < m_surfaces.size();) { 
            Surface *s = m_surfaces[i];
            if(s) {
                s->positionChanged();
                _surfaceVertices_local += s->parents().size();
            }
            i += stride;
        }
        _surfaceVertices_m.fetch_add(_surfaceVertices_local);
    };
    parallel_for(stride, func_surfaces);
    _surfaceVertices += _surfaceVertices_m;

    // Update bodies
    
    std::vector<Body*> &m_bodies = mesh->bodies;
    auto func_bodies = [&m_bodies](int i) -> void {
        Body *b = m_bodies[i];
        if(b) 
            b->positionChanged();
    };
    parallel_for(m_bodies.size(), func_bodies);

    mesh->isDirty = false;

    _isDirty = false;

    return S_OK;
}

HRESULT MeshSolver::positionChanged() {
    TF_MESHSOLVER_CHECKINIT

    return _solver->_positionChangedInst();
}

HRESULT MeshSolver::update(const bool &_force) {
    if(!isDirty() || _force) 
        return S_OK;
    
    positionChanged();
    return S_OK;
}

HRESULT MeshSolver::preStepStart() { 
    TF_MESHSOLVER_CHECKINIT

    MeshLogger::clear();

    _totalVertices = mesh->sizeVertices();

    MeshSolverTimerInstance t(MeshSolverTimers::Section::FORCE);

    if(_totalVertices > _bufferSize) {
        free(_solver->_forces);
        _bufferSize = _totalVertices;
        _solver->_forces = (FloatP_t*)malloc(3 * sizeof(FloatP_t) * _bufferSize);
    }
    memset(_solver->_forces, 0.f, 3 * sizeof(FloatP_t) * _bufferSize);

    std::vector<Vertex*> &m_vertices = mesh->vertices;
    FloatP_t *v_forces = &_forces[0];
    auto func = [&m_vertices, &v_forces](int i) -> void {
        Vertex *v = m_vertices[i];
        if(v) 
            VertexForce(v, &v_forces[i * 3]);
    };
    parallel_for(m_vertices.size(), func);

    return S_OK;
}

HRESULT MeshSolver::preStepJoin() {

    MeshSolverTimerInstance t(MeshSolverTimers::Section::ADVANCE);

    FloatP_t *v_forces = &_forces[0];
    std::vector<Vertex*> &m_vertices = mesh->vertices;
    auto func = [&m_vertices, &v_forces](int k) -> void {
        Vertex *v = m_vertices[k];
        if(!v) {
            return;
        }

        Particle *p = v->particle()->part();
        FloatP_t *buff = &v_forces[k * 3];
        p->f[0] += buff[0];
        p->f[1] += buff[1];
        p->f[2] += buff[2];
    };
    parallel_for(m_vertices.size(), func);

    return S_OK;
}

static std::vector<unsigned int> MeshSolver_surfaceVertexIndices(Mesh *mesh) { 
    unsigned int idx = 0;

    std::vector<unsigned int> indices;
    indices.reserve(mesh->sizeSurfaces());
    
    for(unsigned int i = 0; i < mesh->sizeSurfaces(); i++) { 
        indices.push_back(idx);
        Surface *s = mesh->getSurface(i);
        if(s) 
            idx += s->parents().size();
    }

    return indices;
}

HRESULT MeshSolver::postStepStart() {
    setDirty(true);

    {
        MeshSolverTimerInstance t(MeshSolverTimers::Section::UPDATE);

        if(positionChanged() != S_OK) 
            return E_FAIL;
    }
    
    {
        MeshSolverTimerInstance t(MeshSolverTimers::Section::QUALITY);
        
        if(mesh->hasQuality()) 
            mesh->getQuality().doQuality();

        if(positionChanged() != S_OK) 
            return E_FAIL;
    }

    return S_OK;
}

HRESULT MeshSolver::postStepJoin() {
    return S_OK;
}

std::vector<unsigned int> MeshSolver::_getSurfaceVertexIndicesInst() const {
    TF_MESHSOLVER_CHECKINIT_RET({});

    return MeshSolver_surfaceVertexIndices(_solver->mesh);
}

std::vector<unsigned int> MeshSolver::getSurfaceVertexIndices() {
    TF_MESHSOLVER_CHECKINIT_RET({});

    return _solver->_getSurfaceVertexIndicesInst();
}

HRESULT MeshSolver::_getSurfaceVertexIndicesAsyncStartInst() {
    fut_surfaceVertexIndices = std::async(MeshSolver_surfaceVertexIndices, mesh);
    return S_OK;
}

HRESULT MeshSolver::getSurfaceVertexIndicesAsyncStart() {
    TF_MESHSOLVER_CHECKINIT

    return _solver->_getSurfaceVertexIndicesAsyncStartInst();
}

std::vector<unsigned int> MeshSolver::_getSurfaceVertexIndicesAsyncJoinInst() {
    if(fut_surfaceVertexIndices.valid()) 
        _surfaceVertexIndices = fut_surfaceVertexIndices.get();
    return _surfaceVertexIndices;
}

std::vector<unsigned int> MeshSolver::getSurfaceVertexIndicesAsyncJoin() {
    TF_MESHSOLVER_CHECKINIT_RET({});

    return _solver->_getSurfaceVertexIndicesAsyncJoinInst();
}

HRESULT MeshSolver::_logInst(const MeshLogEventType &type, const std::vector<int> &objIDs, const std::vector<MeshObj::Type> &objTypes, const std::string &name) {
    MeshLogEvent event;
    event.name = name;
    event.type = type;
    event.objIDs = objIDs;
    event.objTypes = objTypes;
    return MeshLogger::log(event);
}

HRESULT MeshSolver::log(const MeshLogEventType &type, const std::vector<int> &objIDs, const std::vector<MeshObj::Type> &objTypes, const std::string &name) {
    TF_MESHSOLVER_CHECKINIT

    return _solver->_logInst(type, objIDs, objTypes, name);
}

bool MeshSolver::_isDirtyInst() const {
    if(_isDirty) 
        return true;
    return mesh->isDirty;
}

bool MeshSolver::isDirty() {
    TF_MESHSOLVER_CHECKINIT

    return _solver->_isDirtyInst();
}

HRESULT MeshSolver::_setDirtyInst(const bool &_dirty) {
    _isDirty = _dirty;
    mesh->isDirty = _dirty;
    return S_OK;
}

HRESULT MeshSolver::setDirty(const bool &_dirty) {
    TF_MESHSOLVER_CHECKINIT

    return _solver->_setDirtyInst(_dirty);
}
