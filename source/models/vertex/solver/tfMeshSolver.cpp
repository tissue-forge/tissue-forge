/*******************************************************************************
 * This file is part of Tissue Forge.
 * Copyright (c) 2022, 2023 T.J. Sego and Tien Comlekoglu
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
    FVector3 force;

    // Surfaces
    int tid = -1;
    SurfaceType *stype;
    for(auto &s : v->getSurfaces()) {
        if(s->typeId != tid) {
            tid = s->typeId;
            stype = s->type();
        }
        for(auto &a : stype->actors) 
            force += a->force(s, v);
        
        for(auto &a : s->actors) 
            force += a->force(s, v);
    }

    // Bodies
    tid = -1;
    BodyType *btype;
    for(auto &b : v->getBodies()) {
        if(b->typeId != tid) {
            tid = b->typeId;
            btype = b->type();
        }
        for(auto &a : btype->actors) 
            force += a->force(b, v);

        for(auto &a : b->actors) 
            force += a->force(b, v);
    }

    f[0] += force[0];
    f[1] += force[1];
    f[2] += force[2];

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

    if(TissueForge::io::FIO::hasImport()) 
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

bool MeshSolver::is3D() {
    TF_MESHSOLVER_CHECKINIT_RET(false)

    return _solver->mesh->is3D();
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

HRESULT MeshSolver::_registerTypeInst(BodyType *_type) {
    if(!_type || _type->id >= 0) 
        return E_FAIL;
    
    _type->id = _bodyTypes.size();
    if(MeshSolver_assignUniqueNameAsNecessary(_type, _bodyTypes)) {
        TF_Log(LOG_INFORMATION) << "Type name not unique. Generating name: " << _type->name.c_str();
    }
    _bodyTypes.push_back(_type);

    return S_OK;
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
    if(MeshSolver_assignUniqueNameAsNecessary(_type, _surfaceTypes)) {
        TF_Log(LOG_INFORMATION) << "Type name not unique. Generating name: " << _type->name.c_str();
    }
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

HRESULT MeshSolver::_positionChangedInst() {

    _surfaceVertices = 0;
    _totalVertices = 0;

    // Update vertices

    if(mesh->vertices->size() > 0) {

        Vertex *m_vertices = &(*mesh->vertices)[0];
        const size_t m_size_vertices = mesh->vertices->size();
        const int blockSize = std::ceil(float(m_size_vertices) / ThreadPool::size());
        auto func_vertices = [&m_vertices, m_size_vertices, blockSize](int tid) -> void {
            int i0 = tid * blockSize;
            int i1 = std::min<int>(i0 + blockSize, m_size_vertices);
            for(int i = i0; i < i1; i++) {
                Vertex &v = m_vertices[i];
                if(v.objectId() >= 0) 
                    v.positionChanged();
            }
        };
        parallel_for(ThreadPool::size(), func_vertices);
        _totalVertices += mesh->numVertices();

    }

    // Update surfaces

    if(mesh->surfaces->size() > 0) {
    
        Surface *m_surfaces = &(*mesh->surfaces)[0];
        const size_t m_size_surfaces = mesh->surfaces->size();
        const int blockSize = std::ceil(float(m_size_surfaces) / ThreadPool::size());
        std::atomic<unsigned int> _surfaceVertices_m = 0;
        auto func_surfaces = [&m_surfaces, &m_size_surfaces, blockSize, &_surfaceVertices_m](int tid) -> void {
            int i0 = tid * blockSize;
            int i1 = std::min<int>(i0 + blockSize, m_size_surfaces);
            unsigned int _surfaceVertices_local = 0;
            for(int i = i0; i < i1; i++) { 
                Surface &s = m_surfaces[i];
                if(s.objectId() >= 0) {
                    s.positionChanged();
                    _surfaceVertices_local += s.getVertices().size();
                }
            }
            _surfaceVertices_m.fetch_add(_surfaceVertices_local);
        };
        parallel_for(ThreadPool::size(), func_surfaces);
        _surfaceVertices = _surfaceVertices_m;

    }

    // Update bodies

    if(mesh->bodies->size() > 0) {
    
        Body *m_bodies = &(*mesh->bodies)[0];
        const size_t m_size_bodies = mesh->bodies->size();
        const int blockSize = std::ceil(float(m_size_bodies) / ThreadPool::size());
        auto func_bodies = [&m_bodies, &m_size_bodies, blockSize](int tid) -> void {
            int i0 = tid * blockSize;
            int i1 = std::min<int>(i0 + blockSize, m_size_bodies);
            for(int i = i0; i < i1; i++) {
                Body &b = m_bodies[i];
                if(b.objectId() >= 0) 
                    b.positionChanged();
            }
        };
        parallel_for(ThreadPool::size(), func_bodies);

    }

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

    if(_totalVertices == 0) 
        return S_OK;

    Vertex *m_vertices = &(*mesh->vertices)[0];
    FloatP_t *v_forces = &_forces[0];
    const size_t m_size_vertices = mesh->vertices->size();
    const int blockSize = std::ceil(float(m_size_vertices) / ThreadPool::size());
    auto func = [&m_vertices, &v_forces, m_size_vertices, blockSize](int tid) -> void {
        int i0 = tid * blockSize;
        int i1 = std::min<int>(i0 + blockSize, m_size_vertices);
        for(int i = i0; i < i1; i++) { 
            Vertex &v = m_vertices[i];
            if(v.objectId() >= 0) {
                v.updateProperties();
                VertexForce(&v, &v_forces[i * 3]);
            }
        }
    };
    parallel_for(ThreadPool::size(), func);

    return S_OK;
}

HRESULT MeshSolver::preStepJoin() {

    MeshSolverTimerInstance t(MeshSolverTimers::Section::ADVANCE);

    if(_totalVertices == 0) 
        return S_OK;

    FloatP_t *v_forces = &_forces[0];
    Vertex *m_vertices = &(*mesh->vertices)[0];
    auto func = [&m_vertices, &v_forces](int k) -> void {
        Vertex &v = m_vertices[k];
        if(v.objectId() < 0) {
            return;
        }

        Particle *p = v.particle()->part();
        FloatP_t *buff = &v_forces[k * 3];
        p->f[0] += buff[0];
        p->f[1] += buff[1];
        p->f[2] += buff[2];
    };
    parallel_for(mesh->vertices->size(), func);

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
            idx += s->getVertices().size();
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

HRESULT MeshSolver::_logInst(const MeshLogEventType &type, const std::vector<int> &objIDs, const std::vector<MeshObjTypeLabel> &objTypes, const std::string &name) {
    MeshLogEvent event;
    event.name = name;
    event.type = type;
    event.objIDs = objIDs;
    event.objTypes = objTypes;
    return MeshLogger::log(event);
}

HRESULT MeshSolver::log(const MeshLogEventType &type, const std::vector<int> &objIDs, const std::vector<MeshObjTypeLabel> &objTypes, const std::string &name) {
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
