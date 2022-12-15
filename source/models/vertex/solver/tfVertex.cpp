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

#include "tfVertex.h"

#include "tfSurface.h"
#include "tfBody.h"
#include "tfStructure.h"
#include "tfMeshSolver.h"
#include "tf_mesh_io.h"
#include "tfVertexSolverFIO.h"

#include <tfError.h>
#include <tfLogger.h>
#include <tfEngine.h>
#include <io/tfIO.h>
#include <io/tfFIO.h>

#include <unordered_set>


using namespace TissueForge;
using namespace TissueForge::models::vertex;

#define Vertex_GETMESH(name, retval)                \
    Mesh *name = Mesh::get();                       \
    if(!name) {                                     \
        TF_Log(LOG_ERROR) << "Could not get mesh";  \
        return retval;                              \
    }


MeshParticleType *TissueForge::models::vertex::MeshParticleType_get() {
    TF_Log(LOG_TRACE);

    MeshParticleType tmp;
    ParticleType *result = ParticleType_FindFromName(tmp.name);
    if(result) 
        return (MeshParticleType*)result;
    
    TF_Log(LOG_DEBUG) << "Registering vertex particle type with name " << tmp.name;
    tmp.registerType();
    TF_Log(LOG_DEBUG) << "Particle types: " << _Engine.nr_types;
    
    result = ParticleType_FindFromName(tmp.name);
    if(!result) {
        TF_Log(LOG_ERROR);
        return NULL;
    }
    return (MeshParticleType*)result;
}

std::vector<Vertex*> Vertex::neighborVertices() const {
    std::unordered_set<Vertex*> result;
    Vertex *vp, *vn;

    for(auto &s : surfaces) {
        std::tie(vp, vn) = s->neighborVertices(this);
        result.insert(vp);
        result.insert(vn);
    }
    return std::vector<Vertex*>(result.begin(), result.end());
}

std::vector<Surface*> Vertex::sharedSurfaces(const Vertex *other) const {
    std::vector<Surface*> result;
    for(auto &s : surfaces) 
        if(other->in(s) && std::find(result.begin(), result.end(), s) == result.end()) 
            result.push_back(s);
    return result;
}

FloatP_t Vertex::getVolume() const {
    FloatP_t result = 0.f;
    for(auto &b : getBodies()) 
        result += b->getVertexVolume(this);
    return result;
}

FloatP_t Vertex::getMass() const {
    FloatP_t result = 0.f;
    for(auto &b : getBodies()) 
        result += b->getVertexMass(this);
    return result;
}

HRESULT Vertex::positionChanged() {
    return S_OK;
}

HRESULT Vertex::updateProperties() {
    ParticleHandle *p = particle();
    const FloatP_t vMass = getMass();
    if(p && vMass > 0.f) {
        p->setMass(vMass);
    }
    return S_OK;
}

ParticleHandle *Vertex::particle() const {
    if(this->pid < 0) {
        TF_Log(LOG_DEBUG);
        return NULL;
    }

    Particle *p = Particle_FromId(this->pid);
    if(!p) {
        TF_Log(LOG_ERROR);
        return NULL;
    }

    return p->handle();
}

FVector3 Vertex::getPosition() const {
    auto p = particle();
    if(!p) { 
        TF_Log(LOG_ERROR) << "No assigned particle.";
        FVector3(-1.f, -1.f, -1.f);
    }
    return p->getPosition();
}

HRESULT Vertex::setPosition(const FVector3 &pos) {
    auto p = particle();
    if(!p) {
        TF_Log(LOG_ERROR) << "No assigned particle.";
        return E_FAIL;
    }
    p->setPosition(pos);

    for(auto &s : surfaces) 
        s->positionChanged();

    return S_OK;
}

Vertex::Vertex() : 
    MeshObj(), 
    pid{-1}
{}

Vertex::Vertex(const unsigned int &_pid) : 
    Vertex() 
{
    Mesh *mesh = Mesh::get();
    pid = (int)_pid;
    if(!mesh || mesh->add(this) != S_OK)
        pid = -1;
};

Vertex::Vertex(const FVector3 &position) : 
    Vertex()
{
    MeshParticleType *ptype = MeshParticleType_get();
    Mesh *mesh = Mesh::get();
    this->pid = -1;
    if(!ptype) {
        TF_Log(LOG_ERROR) << "Could not instantiate particle type";
    } 
    else if(!mesh) {
        TF_Log(LOG_ERROR) << "Could not get mesh";
    }
    else {
        FVector3 _position = position;
        ParticleHandle *ph = (*ptype)(&_position);
        this->pid = ph->id;
        if(mesh->add(this) != S_OK) {
            TF_Log(LOG_ERROR) << "Could not add vertex";
            this->pid = -1;
            ph->destroy();
        }
    }
}

Vertex::Vertex(TissueForge::io::ThreeDFVertexData *vdata) :
    Vertex(vdata->position)
{}

std::vector<MeshObj*> Vertex::children() const {
    return TissueForge::models::vertex::vectorToBase(surfaces);
}

HRESULT Vertex::addChild(MeshObj *obj) {
    if(!TissueForge::models::vertex::check(obj, MeshObj::Type::SURFACE)) {
        TF_Log(LOG_ERROR);
        return E_FAIL;
    }

    Surface *s = (Surface*)obj;
    if(std::find(surfaces.begin(), surfaces.end(), s) != surfaces.end()) {
        TF_Log(LOG_ERROR);
        return E_FAIL;
    }

    surfaces.push_back(s);
    return S_OK;
}

HRESULT Vertex::removeChild(MeshObj *obj) {
    if(!TissueForge::models::vertex::check(obj, MeshObj::Type::SURFACE)) {
        TF_Log(LOG_ERROR);
        return E_FAIL;
    }

    Surface *s = (Surface*)obj;
    auto itr = std::find(surfaces.begin(), surfaces.end(), s);
    if(itr == surfaces.end()) {
        TF_Log(LOG_ERROR);
        return E_FAIL;
    }
    surfaces.erase(itr);
    return S_OK;
}

std::string Vertex::str() const {
    std::stringstream ss;

    ss << "Vertex(";
    if(this->objId >= 0) 
        ss << "id=" << this->objId;
    ss << ")";

    return ss.str();
}

#define VERTEX_RND_IDX(vec_size, idx) {     \
while(idx < 0) idx += vec_size;             \
while(idx >= vec_size) idx -= vec_size;     \
}

HRESULT Vertex::add(Surface *s) {
    return addChild(s);
}

HRESULT Vertex::insert(Surface *s, const int &idx) { 
    int _idx = idx;
    VERTEX_RND_IDX(this->surfaces.size(), _idx);
    this->surfaces.insert(this->surfaces.begin() + _idx, s);
    return S_OK;
}

HRESULT Vertex::insert(Surface *s, Surface *before) {
    auto itr = std::find(this->surfaces.begin(), this->surfaces.end(), before);
    if(itr == this->surfaces.end()) 
        return E_FAIL;
    this->surfaces.insert(itr, s);
    return S_OK;
}

HRESULT Vertex::remove(Surface *s) {
    auto itr = std::find(this->surfaces.begin(), this->surfaces.end(), s);
    if(itr == this->surfaces.end()) 
        return E_FAIL;
    this->surfaces.erase(itr);
    return S_OK;
}

HRESULT Vertex::replace(Surface *toInsert, const int &idx) {
    int _idx = idx;
    VERTEX_RND_IDX(this->surfaces.size(), _idx);
    std::replace(this->surfaces.begin(), this->surfaces.end(), this->surfaces[idx], toInsert);
    return S_OK;
}

HRESULT Vertex::replace(Surface *toInsert, Surface *toRemove) {
    std::replace(this->surfaces.begin(), this->surfaces.end(), toRemove, toInsert);
    return this->in(toInsert) ? S_OK : E_FAIL;
}

HRESULT Vertex::destroy() {
    TF_Log(LOG_TRACE) << this->objId << "; " << this->pid;

    for(auto &s : surfaces) 
        if(s->destroy() != S_OK) {
            TF_Log(LOG_DEBUG) << s->objId;
            return E_FAIL;
        }
    if(this->objId >= 0) 
        if(Mesh::get()->remove(this) != S_OK) {
            TF_Log(LOG_DEBUG);
        }
    if(this->pid >= 0 && this->particle()->destroy() != S_OK) {
        TF_Log(LOG_DEBUG);
        return E_FAIL;
    }
    this->pid = -1;
    return S_OK;
}

std::vector<Structure*> Vertex::getStructures() const {
    std::unordered_set<Structure*> result;
    for(auto &s : surfaces) 
        for(auto &ss : s->getStructures()) 
            result.insert(ss);
    return std::vector<Structure*>(result.begin(), result.end());
}

std::vector<Body*> Vertex::getBodies() const {
    std::unordered_set<Body*> result;
    for(auto &s : surfaces) 
        for(auto &b : s->getBodies()) 
            result.insert(b);
    return std::vector<Body*>(result.begin(), result.end());
}

Surface *Vertex::findSurface(const FVector3 &dir) const {
    Surface *result = 0;

    FloatP_t bestCrit = 0;
    const FVector3 position = getPosition();

    for(auto &s : getSurfaces()) {
        const FVector3 rel_pt = s->getCentroid() - position;
        if(rel_pt.isZero()) 
            continue;
        const FloatP_t crit = rel_pt.dot(dir) / rel_pt.dot();
        if(!result || crit > bestCrit) { 
            result = s;
            bestCrit = crit;
        }
    }

    return result;
}

Body *Vertex::findBody(const FVector3 &dir) const {
    Body *result = 0;

    FloatP_t bestCrit = 0;
    const FVector3 position = getPosition();

    for(auto &b : getBodies()) {
        const FVector3 rel_pt = b->getCentroid() - position;
        if(rel_pt.isZero()) 
            continue;
        FloatP_t crit = rel_pt.dot(dir) / rel_pt.dot();
        if(!result || crit > bestCrit) { 
            result = b;
            bestCrit = crit;
        }
    }

    return result;
}

HRESULT Vertex::transferBondsTo(Vertex *other) {
    ParticleHandle *ph = this->particle();

    for(auto &ah : ph->getAngles()) {
        Angle *a = ah.get();
        if(a->i == this->pid) {
            if(a->j == other->pid || a->k == other->pid) 
                ah.destroy();
            else 
                a->i = other->pid;
        } 
        else if(a->j == this->pid) {
            if(a->i == other->pid || a->k == other->pid) 
                ah.destroy();
            else 
                a->j = other->pid;
        } 
        else if(a->k == this->pid) {
            if(a->i == other->pid || a->j == other->pid) 
                ah.destroy();
            else 
                a->k = other->pid;
        }
    }
    
    std::unordered_set<uint32_t> bonded_ids;
    bonded_ids.insert(other->pid);
    for(auto &bh : ph->getBonds()) {
        Bond *b = bh.get();
        if(b->i == this->pid) {
            if(std::find(bonded_ids.begin(), bonded_ids.end(), b->j) != bonded_ids.end()) 
                bh.destroy();
            else {
                b->i = other->pid;
                bonded_ids.insert(b->j);
            }
        } 
        else if(b->j == this->pid) {
            if(std::find(bonded_ids.begin(), bonded_ids.end(), b->i) != bonded_ids.end()) 
                bh.destroy();
            else {
                b->j = other->pid;
                bonded_ids.insert(b->i);
            }
        } 
    }
    
    for(auto &dh : ph->getDihedrals()) {
        Dihedral *d = dh.get();
        if(d->i == this->pid) {
            if(d->j == other->pid || d->k == other->pid || d->l == other->pid) 
                dh.destroy();
            else 
                d->i = other->pid;
        } 
        else if(d->j == this->pid) {
            if(d->i == other->pid || d->k == other->pid || d->l == other->pid) 
                dh.destroy();
            else 
                d->j = other->pid;
        } 
        else if(d->k == this->pid) {
            if(d->i == other->pid || d->j == other->pid || d->l == other->pid) 
                dh.destroy();
            else 
                d->k = other->pid;
        }
        else if(d->l == this->pid) {
            if(d->i == other->pid || d->j == other->pid || d->k == other->pid) 
                dh.destroy();
            else 
                d->l = other->pid;
        }
    }

    return S_OK;
}

HRESULT Vertex_SurfaceDisconnectReplace(
    Vertex *toInsert, 
    Surface *toReplace, 
    Surface *targetSurf, 
    std::vector<Vertex*> &targetSurf_vertices, 
    std::set<Vertex*> &totalToRemove) 
{
    std::vector<unsigned int> edgeLabels = targetSurf->contiguousEdgeLabels(toReplace);
    std::vector<Vertex*> toRemove;
    for(unsigned int i = 0; i < edgeLabels.size(); i++) {
        unsigned int lab = edgeLabels[i];
        if(lab > 0) {
            if(lab > 1) {
                TF_Log(LOG_ERROR) << "Replacement cannot occur over non-contiguous contacts";
                return E_FAIL;
            }
            toRemove.push_back(targetSurf_vertices[i]);
        }
    }
    
    if(toRemove.empty()) 
        return S_OK;
    
    targetSurf->insert(toInsert, toRemove[0]);
    toInsert->add(targetSurf);
    for(auto &v : toRemove) {
        targetSurf->remove(v);
        v->remove(targetSurf);
        totalToRemove.insert(v);
    }
    return S_OK;
}

HRESULT Vertex::replace(Surface *toReplace) {
    // For every surface connected to the replaced surface
    //      Gather every vertex connected to the replaced surface
    //      Replace all vertices with the inserted vertex
    // Remove the replaced surface from the mesh
    // Add the inserted vertex to the mesh

    // Prevent nonsensical resultant bodies
    if(toReplace->b1 && toReplace->b1->surfaces.size() < 5) { 
        TF_Log(LOG_DEBUG) << "Insufficient surfaces (" << toReplace->b1->surfaces.size() << ") in first body (" << toReplace->b1->objId << ") for replace";
        return E_FAIL;
    }
    else if(toReplace->b2 && toReplace->b2->surfaces.size() < 5) {
        TF_Log(LOG_DEBUG) << "Insufficient surfaces (" << toReplace->b2->surfaces.size() << ") in first body (" << toReplace->b2->objId << ") for replace";
        return E_FAIL;
    }

    // Gather every contacting surface
    std::vector<Surface*> connectedSurfaces = toReplace->connectedSurfaces();

    // Disconnect every vertex connected to the replaced surface
    std::set<Vertex*> totalToRemove;
    for(auto &s : connectedSurfaces) 
        if(Vertex_SurfaceDisconnectReplace(this, toReplace, s, s->vertices, totalToRemove) != S_OK) 
            return E_FAIL;

    MeshSolver::log(MeshLogEventType::Create, {objId, toReplace->objId}, {objType(), toReplace->objType()}, "replace");

    // Remove the replaced surface and its vertices
    while(!toReplace->vertices.empty()) {
        Vertex *v = toReplace->vertices.front();
        v->remove(toReplace);
        toReplace->remove(v);
        totalToRemove.insert(v);
    }
    if(toReplace->b1) { 
        Body *b1 = toReplace->b1;
        b1->remove(toReplace);
        toReplace->remove(b1);
        b1->positionChanged();
    }
    if(toReplace->b2) { 
        Body *b2 = toReplace->b2;
        b2->remove(toReplace);
        toReplace->remove(b2);
        b2->positionChanged();
    }
    if(toReplace->destroy() != S_OK) 
        return E_FAIL;
    for(auto &v : totalToRemove) 
        if(v->destroy() != S_OK) 
            return E_FAIL;

    if(!Mesh::get()->qualityWorking() && MeshSolver::positionChanged() != S_OK)
        return E_FAIL;

    return S_OK;
}

Vertex *Vertex::replace(const FVector3 &position, Surface *toReplace) {
    Vertex *result = new Vertex(position);
    if(result->replace(toReplace) != S_OK) {
        result->destroy();
        delete result;
        return NULL;
    }
    return result;
}

HRESULT Vertex::replace(Body *toReplace) {

    // Detach surfaces and bodies
    std::set<Vertex*> totalToRemove;
    std::vector<Surface*> b_surfaces(toReplace->surfaces);
    for(auto &s : b_surfaces) { 
        for(auto &ns : s->neighborSurfaces()) { 
            if(ns->in(toReplace)) 
                continue;
            if(Vertex_SurfaceDisconnectReplace(this, s, ns, ns->vertices, totalToRemove) != S_OK) 
                return E_FAIL;
        }

        if(s->b1 && s->b1 != toReplace) {
            if(s->b1->surfaces.size() < 5) {
                TF_Log(LOG_DEBUG) << "Insufficient surfaces (" << s->b1->surfaces.size() << ") in first body (" << s->b1->objId << ") for replace";
                return E_FAIL;
            }
            s->b1->remove(s);
            s->remove(s->b1);
        }
        if(s->b2 && s->b2 != toReplace) {
            if(s->b2->surfaces.size() < 5) {
                TF_Log(LOG_DEBUG) << "Insufficient surfaces (" << s->b2->surfaces.size() << ") in first body (" << s->b2->objId << ") for replace";
                return E_FAIL;
            }
            s->b2->remove(s);
            s->remove(s->b2);
        }
    }

    MeshSolver::log(MeshLogEventType::Create, {objId, toReplace->objId}, {objType(), toReplace->objType()}, "replace");

    while(!toReplace->surfaces.empty()) {
        Surface *s = toReplace->surfaces.front();
        while(!s->vertices.empty()) {
            Vertex *v = s->vertices.front();
            s->remove(v);
            v->remove(s);
            totalToRemove.insert(v);
        }
        toReplace->remove(s);
        s->remove(toReplace);
        s->destroy();
    }
    if(toReplace->destroy() != S_OK) 
        return E_FAIL;
    for(auto &v : totalToRemove) 
        if(v->destroy() != S_OK) 
            return E_FAIL;

    if(!Mesh::get()->qualityWorking() && MeshSolver::positionChanged() != S_OK)
        return E_FAIL;

    return S_OK;
}

Vertex *Vertex::replace(const FVector3 &position, Body *toReplace) {
    Vertex *result = new Vertex(position);
    if(result->replace(toReplace) != S_OK) {
        result->destroy();
        delete result;
        return NULL;
    }
    return result;
}

HRESULT Vertex::merge(Vertex *toRemove, const FloatP_t &lenCf) {

    // In common surfaces, just remove; in different surfaces, replace
    std::vector<Surface*> common_s, different_s;
    common_s.reserve(toRemove->surfaces.size());
    different_s.reserve(toRemove->surfaces.size());
    for(auto &s : toRemove->surfaces) {
        if(!in(s)) 
            different_s.push_back(s);
        else 
            common_s.push_back(s);
    }
    for(auto &s : common_s) {
        s->remove(toRemove);
        toRemove->remove(s);
    }
    for(auto &s : different_s) {
        toRemove->remove(s);
        add(s);
        s->replace(this, toRemove);
    }
    
    // Set new position
    const FVector3 posToKeep = getPosition();
    const FVector3 newPos = posToKeep + (toRemove->getPosition() - posToKeep) * lenCf;
    if(setPosition(newPos) != S_OK) 
        return E_FAIL;

    MeshSolver::log(MeshLogEventType::Create, {objId, toRemove->objId}, {objType(), toRemove->objType()}, "merge");
    
    if(toRemove->transferBondsTo(this) != S_OK || toRemove->destroy() != S_OK) 
        return E_FAIL;

    if(!Mesh::get()->qualityWorking() && MeshSolver::positionChanged() != S_OK)
        return E_FAIL;

    return S_OK;
}

HRESULT Vertex::insert(Vertex *v1, Vertex *v2) {

    std::vector<Vertex*>::iterator vitr;

    // Find the common surface(s)
    for(auto &s1 : v1->surfaces) {
        if(in(s1)) 
            continue;
        for(vitr = s1->vertices.begin(); vitr != s1->vertices.end(); vitr++) {
            std::vector<Vertex*>::iterator vnitr = vitr + 1 == s1->vertices.end() ? s1->vertices.begin() : vitr + 1;
            
            if((*vitr == v1 && *vnitr == v2) || (*vitr == v2 && *vnitr == v1)) {
                s1->vertices.insert(vitr + 1 == s1->vertices.end() ? s1->vertices.begin() : vitr + 1, this);
                add(s1);
                break;
            }
        }
    }

    if(!Mesh::get()->qualityWorking() && MeshSolver::positionChanged() != S_OK)
        return E_FAIL;

    MeshSolver::log(MeshLogEventType::Create, {objId, v2->objId}, {objType(), v2->objType()}, "insert");

    return S_OK;
}

Vertex *Vertex::insert(const FVector3 &position, Vertex *v1, Vertex *v2) {
    Vertex *result = new Vertex(position);
    if(result->insert(v1, v2) != S_OK) {
        result->destroy();
        delete result;
        return NULL;
    }
    return result;
}

HRESULT Vertex::insert(Vertex *vf, std::vector<Vertex*> nbs) {
    for(auto &v : nbs) 
        if(insert(vf, v) != S_OK) 
            return E_FAIL;
    return S_OK;
}

Vertex *Vertex::insert(const FVector3 &position, Vertex *vf, std::vector<Vertex*> nbs) {
    Vertex *result = new Vertex(position);
    if(result->insert(vf, nbs) != S_OK) {
        result->destroy();
        delete result;
        return NULL;
    }
    return result;
}

HRESULT Vertex::splitPlan(const FVector3 &sep, std::vector<Vertex*> &verts_v, std::vector<Vertex*> &verts_new_v) {
    // Verify inputs
    if(sep.isZero()) 
        return tf_error(E_FAIL, "Zero separation");

    verts_v.clear();
    verts_new_v.clear();

    std::vector<Vertex*> nbs = neighborVertices();

    // Verify that 
    //  1. the vertex is in the mesh
    //  2. the vertex defines at least one surface
    if(nbs.size() == 0) 
        return tf_error(E_FAIL, "Vertex must define a surface");
    
    // Define a cut plane at the midpoint of and orthogonal to the new edge
    FVector4 planeEq = FVector4::planeEquation(sep.normalized(), getPosition());

    // Determine which neighbors will be connected to each vertex
    verts_new_v.reserve(nbs.size());
    verts_v.reserve(nbs.size());
    for(auto nv : nbs) {
        if(planeEq.distance(nv->getPosition()) >= 0) 
            verts_new_v.push_back(nv);
        else 
            verts_v.push_back(nv);
    }

    // Reject if either side of the plane has no vertices
    if(verts_new_v.empty() || verts_v.empty()) {
        verts_v.clear();
        verts_new_v.clear();
        TF_Log(LOG_DEBUG) << "No vertices on both sides of cut plane; ignoring";
        return S_OK;
    }

    return S_OK;

}

Vertex *Vertex::splitExecute(const FVector3 &sep, const std::vector<Vertex*> &verts_v, const std::vector<Vertex*> &verts_new_v) {
    FVector3 v_pos0 = getPosition();
    FVector3 hsep = sep * 0.5;
    FVector3 v_pos1 = v_pos0 - hsep;
    FVector3 u_pos = v_pos0 + hsep;

    // Determine which surfaces the target vertex will no longer partially define
    // A surface remains partially defined by the target vertex if the target vertex has 
    // a neighbor on its own side of the cut plane that also partially defines the surface
    std::set<Surface*> u_surfs, vn_surfs;
    for(auto &nv : verts_v) 
        for(auto &s : nv->sharedSurfaces(this)) 
            vn_surfs.insert(s);
    for(auto &nv : verts_new_v) 
        for(auto &s : nv->sharedSurfaces(this)) 
            u_surfs.insert(s);
    std::set<Surface*> surfs_keep_v, surfs_remove_v;
    for(auto &s : u_surfs) {
        if(std::find(vn_surfs.begin(), vn_surfs.end(), s) == vn_surfs.end()) 
            surfs_remove_v.insert(s);
        else 
            surfs_keep_v.insert(s);
    }

    // Create and insert the new vertex
    Vertex *u = new Vertex(u_pos);
    if(u->objId < 0) {
        tf_error(E_FAIL, "Could not add vertex");
        u->destroy();
        delete u;
        return 0;
    }
    setPosition(v_pos1);

    //  Replace v with u where removing
    for(auto &s : surfs_remove_v) {
        remove(s);
        u->add(s);
        s->replace(u, this);
    }

    //  Insert u between v and neighbor where not removing
    for(auto &s : surfs_keep_v) {
        u->add(s);
        for(auto &nv : verts_new_v) {
            std::vector<Vertex*>::iterator verts_new_v_itr = std::find(s->vertices.begin(), s->vertices.end(), nv);
            if(verts_new_v_itr != s->vertices.end()) { 
                s->insert(u, this, *verts_new_v_itr);
                break;
            }
        }
    }

    if(!Mesh::get()->qualityWorking()) 
        MeshSolver::positionChanged();

    MeshSolver::log(MeshLogEventType::Create, {objId, u->objId}, {objType(), u->objType()}, "split");

    return u;
}

Vertex *Vertex::split(const FVector3 &sep) {
    
    std::vector<Vertex*> verts_v, new_verts_v;
    Vertex *u = NULL;
    if(splitPlan(sep, verts_v, new_verts_v))
        u = splitExecute(sep, verts_v, new_verts_v);
    if(!u) {
        tf_error(E_FAIL, "Failed to split");
        return NULL;
    }

    if(!Mesh::get()->qualityWorking()) 
        MeshSolver::positionChanged();

    MeshSolver::log(MeshLogEventType::Create, {objId, u->objId}, {objType(), u->objType()}, "split");

    return u;
}

namespace TissueForge::io {


    #define TF_MESH_VERTEXIOTOEASY(fe, key, member) \
        fe = new IOElement(); \
        if(toFile(member, metaData, fe) != S_OK)  \
            return E_FAIL; \
        fe->parent = fileElement; \
        fileElement->children[key] = fe;

    #define TF_MESH_VERTEXIOFROMEASY(feItr, children, metaData, key, member_p) \
        feItr = children.find(key); \
        if(feItr == children.end() || fromFile(*feItr->second, metaData, member_p) != S_OK) \
            return E_FAIL;

    template <>
    HRESULT toFile(const TissueForge::models::vertex::Vertex &dataElement, const MetaData &metaData, IOElement *fileElement) {

        IOElement *fe;

        TF_MESH_VERTEXIOTOEASY(fe, "objId", dataElement.objId);

        ParticleHandle *ph = dataElement.particle();
        if(ph == NULL) {
            TF_MESH_VERTEXIOTOEASY(fe, "pid", -1);
        } 
        else {
            TF_MESH_VERTEXIOTOEASY(fe, "pid", ph->getId());
        }

        std::vector<TissueForge::models::vertex::Surface*> surfaces = dataElement.getSurfaces();
        std::vector<int> surfaceIds;
        surfaceIds.reserve(surfaces.size());
        for(auto &s : surfaces) 
            surfaceIds.push_back(s->objId);
        TF_MESH_VERTEXIOTOEASY(fe, "surfaces", surfaceIds);

        fileElement->type = "Vertex";

        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, TissueForge::models::vertex::Vertex **dataElement) {
        
        if(!FIO::hasImport()) 
            return tf_error(E_FAIL, "No import data available");
        else if(!TissueForge::models::vertex::io::VertexSolverFIOModule::hasImport()) 
            return tf_error(E_FAIL, "No vertex import data available");

        IOChildMap::const_iterator feItr;

        int pidOld;
        TF_MESH_VERTEXIOFROMEASY(feItr, fileElement.children, metaData, "pid", &pidOld);
        auto idItr = FIO::importSummary->particleIdMap.find(pidOld);
        if(idItr == FIO::importSummary->particleIdMap.end() || idItr->second < 0) 
            return tf_error(E_FAIL, "Could not locate particle to import");

        *dataElement = new TissueForge::models::vertex::Vertex(idItr->second);
        if((*dataElement)->objId < 0) {
            delete *dataElement;
            return tf_error(E_FAIL, "Failed to add vertex");
        }

        int objIdOld;
        TF_MESH_VERTEXIOFROMEASY(feItr, fileElement.children, metaData, "objId", &objIdOld);
        TissueForge::models::vertex::io::VertexSolverFIOModule::importSummary->vertexIdMap.insert({objIdOld, (*dataElement)->objId});

        return S_OK;
    }
}

std::string TissueForge::models::vertex::Vertex::toString() {
    return TissueForge::io::toString(*this);
}
