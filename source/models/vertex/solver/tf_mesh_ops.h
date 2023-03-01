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

/**
 * @file tf_mesh_ops.h
 * 
 */

#ifndef _MODELS_VERTEX_SOLVER_TF_MESH_OPS_H_
#define _MODELS_VERTEX_SOLVER_TF_MESH_OPS_H_


#include "tfVertex.h"
#include "tfSurface.h"
#include "tfBody.h"

#include <unordered_set>
#include <unordered_map>
#include <vector>


namespace TissueForge::models::vertex {


/**
 * @brief Get the surfaces defined by a set of vertices
 * 
 * @param verts a set of vertices
 * @param surfs surfaces
 */
static HRESULT definedBy(const std::unordered_set<Vertex*> &verts, std::unordered_set<Surface*> &surfs) {
    for(auto &v : verts) 
        for(auto &s : v->getSurfaces()) 
            surfs.insert(s);
    return S_OK;
}

/**
 * @brief Get the bodies defined by a set of vertices
 * 
 * @param verts a set of vertices
 * @param bodys bodies
 */
static HRESULT definedBy(const std::unordered_set<Vertex*> &verts, std::unordered_set<Body*> &bodys) {
    for(auto &v : verts) 
        for(auto &b : v->getBodies()) 
            bodys.insert(b);
    return S_OK;
}

/**
 * @brief Get the surfaces and bodies defined by a set of vertices
 * 
 * @param verts a set of vertices
 * @param surfs surfaces
 * @param bodys bodies
 */
static HRESULT definedBy(const std::unordered_set<Vertex*> &verts, std::unordered_set<Surface*> &surfs, std::unordered_set<Body*> &bodys) {
    for(auto &v : verts) {
        for(auto &s : v->getSurfaces()) 
            surfs.insert(s);
        for(auto &b : v->getBodies()) 
            bodys.insert(b);
    }
    return S_OK;
}

/**
 * @brief Get the vertices defining a set of surfaces
 * 
 * @param surfs a set of surfaces
 * @param verts vertices
 */
static HRESULT definedBy(const std::unordered_set<Surface*> &surfs, std::unordered_set<Vertex*> &verts) {
    for(auto &s : surfs) 
        for(auto &v : s->getVertices()) 
            verts.insert(v);
    return S_OK;
}

/**
 * @brief Get the bodies defined by a set of surfaces
 * 
 * @param surfs a set of surfaces
 * @param bodys bodies
 */
static HRESULT definedBy(const std::unordered_set<Surface*> &surfs, std::unordered_set<Body*> &bodys) {
    for(auto &s : surfs) 
        for(auto &b : s->getBodies()) 
            bodys.insert(b);
    return S_OK;
}

/**
 * @brief Get the vertices that define, and bodies defined by, a set of surfaces
 * 
 * @param surfs a set of surfaces
 * @param verts vertices
 * @param bodys bodies
 */
static HRESULT definedBy(const std::unordered_set<Surface*> &surfs, std::unordered_set<Vertex*> &verts, std::unordered_set<Body*> &bodys) {
    for(auto &s : surfs) {
        for(auto &v : s->getVertices()) 
            verts.insert(v);
        for(auto &b : s->getBodies()) 
            bodys.insert(b);
    }
    return S_OK;
}

/**
 * @brief Get the vertices that define a set of bodies
 * 
 * @param bodys a set of bodies
 * @param verts vertices
 */
static HRESULT definedBy(const std::unordered_set<Body*> &bodys, std::unordered_set<Vertex*> &verts) {
    for(auto &b : bodys) 
        for(auto &v : b->getVertices()) 
            verts.insert(v);
    return S_OK;
}

/**
 * @brief Get the surfaces that define a set of bodies
 * 
 * @param bodys a set of bodies
 * @param surfs surfaces
 */
static HRESULT definedBy(const std::unordered_set<Body*> &bodys, std::unordered_set<Surface*> &surfs) {
    for(auto &b : bodys) 
        for(auto &s : b->getSurfaces()) 
            surfs.insert(s);
    return S_OK;
}

/**
 * @brief Get the vertices and surfaces that define a set of bodies
 * 
 * @param bodys a set of bodies
 * @param verts vertices
 * @param surfs surfaces
 */
static HRESULT definedBy(const std::unordered_set<Body*> &bodys, std::unordered_set<Vertex*> &verts, std::unordered_set<Surface*> &surfs) {
    for(auto &b : bodys) {
        for(auto &v : b->getVertices()) 
            verts.insert(v);
        for(auto &s : b->getSurfaces()) 
            surfs.insert(s);
    }
    return S_OK;
}

/**
 * @brief Get the vertices connected to, but not in, a set of vertices
 * 
 * @param verts a set of vertices
 */
static std::unordered_set<Vertex*> connectedTo(const std::unordered_set<Vertex*> &verts) {
    std::unordered_set<Vertex*> result;
    for(auto &v : verts) 
        for(auto &nv : v->connectedVertices()) 
            result.insert(nv);
    for(auto &v : verts) 
        result.erase(v);
    return result;
}

/**
 * @brief Get the vertices connected to, but not in, a set of vertices
 * 
 * @param verts a set of vertices
 * @param connectedVerts connected vertices
 * @param vsmapConnected map of surfaces to connected vertices
 * @param vsmapGiven map of surfaces to the given set of vertices
 */
static HRESULT connectedTo(
    const std::unordered_set<Vertex*> &verts, 
    std::unordered_set<Vertex*> &connectedVerts, 
    std::unordered_map<Surface*, std::unordered_set<Vertex*> > &vsmapConnected, 
    std::unordered_map<Surface*, std::unordered_set<Vertex*> > &vsmapGiven) 
{
    for(auto &v : verts) {
        for(auto &nv : v->connectedVertices()) 
            connectedVerts.insert(nv);
        for(auto &s : v->getSurfaces()) {
            auto itr = vsmapGiven.find(s);
            if(itr == vsmapGiven.end()) vsmapGiven.insert({s, {v}});
            else itr->second.insert(v);
        }
    }
    for(auto &v : verts) 
        connectedVerts.erase(v);
    for(auto &v : connectedVerts) 
        for(auto &s : v->getSurfaces()) {
            auto itr = vsmapConnected.find(s);
            if(itr == vsmapConnected.end()) vsmapConnected.insert({s, {v}});
            else itr->second.insert(v);
        }
    return S_OK;
}

/**
 * @brief Get the vertices connected to, but not in, a set of vertices
 * 
 * @param verts a set of vertices
 * @param connectedVerts connected vertices
 * @param vsmap map of surfaces to connected vertices
 */
static HRESULT connectedToMapConnected(
    const std::unordered_set<Vertex*> &verts, 
    std::unordered_set<Vertex*> &connectedVerts, 
    std::unordered_map<Surface*, std::unordered_set<Vertex*> > &vsmap) 
{
    for(auto &v : verts) 
        for(auto &nv : v->connectedVertices()) 
            connectedVerts.insert(nv);
    for(auto &v : verts) 
        connectedVerts.erase(v);
    for(auto &v : connectedVerts) 
        for(auto &s : v->getSurfaces()) {
            auto itr = vsmap.find(s);
            if(itr == vsmap.end()) vsmap.insert({s, {v}});
            else itr->second.insert(v);
        }
    return S_OK;
}

/**
 * @brief Get the vertices connected to, but not in, a set of vertices
 * 
 * @param verts a set of vertices
 * @param connectedVerts connected vertices
 * @param vsmap map of surfaces to the given set of vertices
 */
static HRESULT connectedToMapGiven(
    const std::unordered_set<Vertex*> &verts, 
    std::unordered_set<Vertex*> &connectedVerts, 
    std::unordered_map<Surface*, std::unordered_set<Vertex*> > &vsmap) 
{
    for(auto &v : verts) {
        for(auto &nv : v->connectedVertices()) 
            connectedVerts.insert(nv);
        for(auto &s : v->getSurfaces()) {
            auto itr = vsmap.find(s);
            if(itr == vsmap.end()) vsmap.insert({s, {v}});
            else itr->second.insert(v);
        }
    }
    for(auto &v : verts) 
        connectedVerts.erase(v);
    return S_OK;
}

/**
 * @brief Get the surfaces connected to, but not in, a set of surfaces
 * 
 * @param surfs a set of surfaces
 */
static std::unordered_set<Surface*> connectedTo(const std::unordered_set<Surface*> &surfs) {
    std::unordered_set<Surface*> result;
    for(auto &s : surfs) 
        for(auto ns : s->connectedSurfaces()) 
            result.insert(ns);
    for(auto &s : surfs) 
        result.erase(s);
    return result;
}

/**
 * @brief Get the surfaces connected to, but not in, a set of surfaces
 * 
 * @param surfs a set of surfaces
 */
static HRESULT connectedTo(
    const std::unordered_set<Surface*> &surfs, 
    std::unordered_set<Surface*> &connectedSurfs, 
    std::unordered_map<Body*, std::unordered_set<Surface*> > &sbmapConnected, 
    std::unordered_map<Body*, std::unordered_set<Surface*> > &sbmapGiven) 
{
    for(auto &s : surfs) {
        for(auto ns : s->connectedSurfaces()) 
            connectedSurfs.insert(ns);
        for(auto b : s->getBodies()) {
            auto itr = sbmapGiven.find(b);
            if(itr == sbmapGiven.end()) sbmapGiven.insert({b, {s}});
            else itr->second.insert(s);
        }
    }
    for(auto &s : surfs) 
        connectedSurfs.erase(s);
    for(auto &s : connectedSurfs) 
        for(auto &b : s->getBodies()) {
            auto itr = sbmapConnected.find(b);
            if(itr == sbmapConnected.end()) sbmapConnected.insert({b, {s}});
            else itr->second.insert(s);
        }
    return S_OK;
}

/**
 * @brief Get the surfaces connected to, but not in, a set of surfaces
 * 
 * @param surfs a set of surfaces
 * @param connectedSurfs connected surfaces
 * @param sbmap map of bodies to connected surfaces
 */
static HRESULT connectedToMapConnected(
    const std::unordered_set<Surface*> &surfs, 
    std::unordered_set<Surface*> &connectedSurfs, 
    std::unordered_map<Body*, std::unordered_set<Surface*> > &sbmap) 
{
    for(auto &s : surfs) 
        for(auto ns : s->connectedSurfaces()) 
            connectedSurfs.insert(ns);
    for(auto &s : surfs) 
        connectedSurfs.erase(s);
    for(auto &s : connectedSurfs) 
        for(auto &b : s->getBodies()) {
            auto itr = sbmap.find(b);
            if(itr == sbmap.end()) sbmap.insert({b, {s}});
            else itr->second.insert(s);
        }
    return S_OK;
}

/**
 * @brief Get the surfaces connected to, but not in, a set of surfaces
 * 
 * @param surfs a set of surfaces
 * @param connectedSurfs connected surfaces
 * @param sbmap map of bodies to the given set of surfaces
 */
static HRESULT connectedToMapGiven(
    const std::unordered_set<Surface*> &surfs, 
    std::unordered_set<Surface*> &connectedSurfs, 
    std::unordered_map<Body*, std::unordered_set<Surface*> > &sbmap) 
{
    for(auto &s : surfs) {
        for(auto ns : s->connectedSurfaces()) 
            connectedSurfs.insert(ns);
        for(auto &b : s->getBodies()) {
            auto itr = sbmap.find(b);
            if(itr == sbmap.end()) sbmap.insert({b, {s}});
            else itr->second.insert(s);
        }
    }
    for(auto &s : surfs) 
        connectedSurfs.erase(s);
    return S_OK;
}

/**
 * @brief Get the bodies connected to, but not in, a set of bodies
 * 
 * @param bodys a set of bodies
 */
static std::unordered_set<Body*> connectedTo(const std::unordered_set<Body*> &bodys) {
    std::unordered_set<Body*> result;
    for(auto &b : bodys) 
        for(auto nb : b->connectedBodies()) 
            result.insert(nb);
    for(auto &b : bodys) 
        result.erase(b);
    return result;
}

/**
 * @brief Get the vertices that define the interface between a set of surfaces and elsewhere
 * 
 * @param surfs a set of surfaces
 */
static std::unordered_set<Vertex*> connectingThrough(const std::unordered_set<Surface*> surfs) {
    std::unordered_multimap<Surface*, Vertex*> cpmap;
    std::unordered_set<Vertex*> result;
    for(auto &s : surfs) 
        for(auto &ns : s->connectedSurfaces()) 
            for(auto &v : ns->connectingVertices(s)) 
                cpmap.insert({ns, v});
    for(auto &s : surfs) 
        cpmap.erase(s);
    for(auto &p : cpmap) 
        result.insert(p.second);
    return result;
}

/**
 * @brief Get the surfaces that define the interface between a set of bodies and elsewhere
 * 
 * @param bodys a set of bodies
 */
static std::unordered_set<Surface*> connectingThrough(const std::unordered_set<Body*> bodys) {
    std::unordered_multimap<Body*, Surface*> cpmap;
    std::unordered_set<Surface*> result;
    for(auto &b : bodys) 
        for(auto &nb : b->connectedBodies()) 
            for(auto &s : nb->findInterface(b)) 
                cpmap.insert({nb, s});
    for(auto &b : bodys) 
        cpmap.erase(b);
    for(auto &p : cpmap) 
        result.insert(p.second);
    return result;
}

/**
 * @brief Get the vertices that define the interface between a set of bodies and elsewhere
 * 
 * @param bodys a set of bodies
 * @param verts vertices that define the interface
 */
static HRESULT connectingThrough(const std::unordered_set<Body*> bodys, std::unordered_set<Vertex*> &result) {
    std::unordered_multimap<Body*, Vertex*> cpmap;
    for(auto &b : bodys) 
        for(auto &v : b->getVertices()) 
            for(auto &nb : v->getBodies()) 
                if(b != nb) 
                    cpmap.insert({nb, v});
    for(auto &b : bodys) 
        cpmap.erase(b);
    for(auto &p : cpmap) 
        result.insert(p.second);
    return S_OK;
}

/**
 * @brief Get the surfaces and vertices that define the interface between a set of bodies and elsewhere
 * 
 * @param bodys a set of bodies
 * @param interfaceSurfs surfaces that define the interface
 * @param interfaceVerts vertices that define the interface and do not define an interface surface
 */
static HRESULT connectingThrough(
    const std::unordered_set<Body*> bodys, 
    std::unordered_set<Surface*> &interfaceSurfs, 
    std::unordered_set<Vertex*> &interfaceVerts) 
{
    std::unordered_multimap<Body*, Vertex*> bvmap;
    std::unordered_multimap<Body*, Surface*> bsmap;
    for(auto &b : bodys) 
        for(auto &nb : b->adjacentBodies()) {
            bool hasInterface = false;
            for(auto &s : nb->findInterface(b)) {
                hasInterface = true;
                bsmap.insert({nb, s});
            }
            if(!hasInterface) 
                for(auto v : nb->sharedVertices(b)) 
                    bvmap.insert({nb, v});
        }
    for(auto &b : bodys) {
        bsmap.erase(b);
        bvmap.erase(b);
    }
    for(auto &p : bsmap) 
        interfaceSurfs.insert(p.second);
    for(auto &p : bvmap) 
        interfaceVerts.insert(p.second);
    return S_OK;
}

/**
 * @brief Get the set of bodies adjacent to, but not in, a set of bodies
 * 
 * @param bodys a set of bodies
 * @return std::unordered_set<Body*> 
 */
static std::unordered_set<Body*> adjacentTo(const std::unordered_set<Body*> &bodys) {
    std::unordered_set<Body*> result;
    for(auto &b : bodys) 
        for(auto &nb : b->adjacentBodies()) 
            result.insert(nb);
    for(auto &b : bodys) 
        result.erase(b);
    return result;
}

/**
 * @brief Get the vertices that would be orphaned by removing a set of surfaces
 * 
 * @param toRemove a set of surfaces to be removed
 */
static std::unordered_set<Vertex*> orphanedVertices(const std::unordered_set<Surface*> &toRemove) {
    std::unordered_set<Vertex*> result, affected;
    for(auto &r : toRemove) 
        for(auto &a : r->getVertices()) 
            affected.insert(a);
    for(auto &a : affected) {
        int numChildren = a->getSurfaces().size();
        for(auto &r : toRemove) 
            if(a->defines(r)) 
                numChildren--;
        if(numChildren < 1) 
            result.insert(a);
    }
    return result;
}

/**
 * @brief Get the surfaces that would be orphaned by removing a set of bodies
 * 
 * @param toRemove a set of bodies to be removed
 */
static std::unordered_set<Surface*> orphanedSurfaces(const std::unordered_set<Body*> &toRemove) {
    std::unordered_set<Surface*> result, affected;
    for(auto &r : toRemove) 
        for(auto &a : r->getSurfaces()) 
            affected.insert(a);
    for(auto &a : affected) {
        int numChildren = a->getBodies().size();
        for(auto &r : toRemove) 
            if(a->defines(r)) 
                numChildren--;
        if(numChildren < 1) 
            result.insert(a);
    }
    return result;
}

/**
 * @brief Determines all surfaces necessarily invalidated by removing a set of vertices. 
 * 
 * Does not account for removing bodies (e.g., removing orphaned surfaces). 
 * 
 * @param toRemove a set of vertices to remove
 */
static std::unordered_set<Surface*> removedChildrenByRemovedParents(const std::unordered_set<Vertex*> &toRemove) {
    if(toRemove.empty()) 
        return {};

    // Get the affected surfaces by all removed vertices
    std::vector<Surface*> affectedSurfacesTotal;
    std::unordered_set<Surface*> affectedSurfacesSet;
    for(auto &v : toRemove) 
        for(auto &s : v->getSurfaces()) {
            affectedSurfacesSet.insert(s);
            affectedSurfacesTotal.push_back(s);
        }
    if(affectedSurfacesSet.empty()) 
        return {};
    
    std::vector<Surface*> affectedSurfaces(affectedSurfacesSet.begin(), affectedSurfacesSet.end());
    std::vector<unsigned int> removedSurfCount(affectedSurfaces.size(), 0);

    // Count number of removed vertices per surface
    for(auto &s : affectedSurfacesTotal) 
        for(int i = 0; i < affectedSurfaces.size(); i++) 
            if(affectedSurfaces[i] == s) 
                removedSurfCount[i]++;

    // Collect bodies that would become invalidated and return the collection
    std::unordered_set<Surface*> result;
    for(int i = 0; i < affectedSurfaces.size(); i++) {
        Surface *s = affectedSurfaces[i];
        if(s->getVertices().size() - removedSurfCount[i] < 3) 
            result.insert(s);
    }
    return result;
}

/**
 * @brief Determines all surfaces necessarily invalidated by removing a set of vertices. 
 * 
 * Does not account for removing bodies (e.g., removing orphaned surfaces). 
 * 
 * @param toRemove a set of vertices to remove
 */
static std::vector<Surface*> removedChildrenByRemovedParents(const std::vector<Vertex*> &toRemove) {
    std::unordered_set<Surface*> result = removedChildrenByRemovedParents(std::unordered_set<Vertex*>(toRemove.begin(), toRemove.end()));
    return std::vector<Surface*>(result.begin(), result.end());
}

/**
 * @brief Determines all bodies necessarily invalidated by removing a set of surfaces
 * 
 * @param toRemove a set of surfaces to remove
 */
static std::unordered_set<Body*> removedChildrenByRemovedParents(const std::unordered_set<Surface*> &toRemove) {
    if(toRemove.empty()) 
        return {};

    // Get the affected bodies by all removed surfaces
    std::vector<Body*> affectedBodiesTotal;
    std::unordered_set<Body*> affectedBodiesSet;
    for(auto &s : toRemove) 
        for(auto &b : s->getBodies()) {
            affectedBodiesSet.insert(b);
            affectedBodiesTotal.push_back(b);
        }
    if(affectedBodiesSet.empty()) 
        return {};
    
    std::vector<Body*> affectedBodies(affectedBodiesSet.begin(), affectedBodiesSet.end());
    std::vector<unsigned int> removedSurfCount(affectedBodies.size(), 0);

    // Count number of removed surfaces per body
    for(auto &b : affectedBodiesTotal) 
        for(int i = 0; i < affectedBodies.size(); i++) 
            if(affectedBodies[i] == b) 
                removedSurfCount[i]++;

    // Collect bodies that would become invalidated and return the collection
    std::unordered_set<Body*> result;
    for(int i = 0; i < affectedBodies.size(); i++) {
        Body *b = affectedBodies[i];
        if(b->getSurfaces().size() - removedSurfCount[i] < 4) 
            result.insert(b);
    }
    return result;
}

/**
 * @brief Determines all bodies necessarily invalidated by removing a set of surfaces
 * 
 * @param toRemove a set of surfaces to remove
 */
static std::vector<Body*> removedChildrenByRemovedParents(const std::vector<Surface*> &toRemove) {
    std::unordered_set<Body*> result = removedChildrenByRemovedParents(std::unordered_set<Surface*>(toRemove.begin(), toRemove.end()));
    return std::vector<Body*>(result.begin(), result.end());
}

/**
 * @brief Determines all surfaces necessarily invalidated by converting a surface to a vertex. 
 * 
 * Does not account for removing bodies (e.g., removing orphaned surfaces). 
 * 
 * @param toRemove a surface to be converted to a vertex
 */
static std::unordered_set<Surface*> removedSurfacesByS2V(const Surface *toRemove) {
    std::unordered_set<Surface*> result;
    for(auto &s : toRemove->connectedSurfaces()) 
        if(s->getVertices().size() - s->connectingVertices(toRemove).size() < 2) 
            result.insert(s);
    return result;
}

/**
 * @brief Determines all surfaces affected, but not removed, by converting a surface to a vertex
 * 
 * @param removed a set of surfaces to be removed
 */
static std::unordered_set<Surface*> connectedSurfacesToS2V(const std::unordered_set<Surface*> &removed) {
    std::unordered_set<Surface*> result;
    for(auto &s : removed) 
        for(auto &ns : s->connectedSurfaces()) 
            result.insert(ns);
    for(auto &s : removed) 
        result.erase(s);
    return result;
}

/**
 * @brief Determines all surfaces affected, but not removed, by converting a surface to a vertex
 * 
 * @param toRemove a surface to be converted to a vertex
 */
static std::unordered_set<Surface*> connectedSurfacesToS2V(const Surface *toRemove) {
    std::unordered_set<Surface*> result = connectedSurfacesToS2V(removedSurfacesByS2V(toRemove));
    result.erase(const_cast<Surface*>(toRemove));
    return result;
}

static std::unordered_map<Surface*, std::unordered_set<Vertex*> > mapSurfaceReplacementsByB2V(
    std::unordered_set<Body*> &removedBodies, 
    std::unordered_set<Surface*> &removedSurfaces) 
{
    std::unordered_map<Surface*, std::unordered_set<Vertex*> > result;

    std::unordered_set<Surface*> connectingSurfs;
    std::unordered_set<Vertex*> connectingVerts;
    connectingThrough(removedBodies, connectingSurfs, connectingVerts);

    std::unordered_multimap<Body*, std::pair<Surface*, std::unordered_set<Vertex*> > > bsmap;
    for(auto &v : connectingVerts) 
        for(auto &s : v->getSurfaces()) 
            for(auto &b : s->getBodies()) 
                bsmap.insert({b, {s, {v}}});
    for(auto &s : connectingSurfs) 
        for(auto &ns : s->connectedSurfaces()) {
            std::vector<Vertex*> connectingVertices = s->connectingVertices(ns);
            std::pair<Surface*, std::unordered_set<Vertex*> > p = {ns, {connectingVertices.begin(), connectingVertices.end()}};
            for(auto &b : ns->getBodies()) 
                bsmap.insert({b, p});
        }
    for(auto &b : removedBodies) 
        bsmap.erase(b);
    std::unordered_multimap<Surface*, std::unordered_set<Vertex*> > svmap;
    for(auto &p : bsmap) 
        svmap.insert({p.second.first, p.second.second});
    for(auto &s : removedSurfaces) 
        svmap.erase(s);
    for(unsigned int i = 0; i < svmap.bucket_count(); i++) {
        Surface *s = NULL;
        std::unordered_set<Vertex*> vs;
        for(auto itr = svmap.begin(i); itr != svmap.end(i); itr++) {
            if(!s) 
                s = itr->first;
            for(auto &v : itr->second) 
                vs.insert(v);
        }
        if(!s) 
            continue;
        if(s->getVertices().size() - vs.size() < 2) 
            removedSurfaces.insert(s);
        else 
            result.insert({s, vs});
    }
    
    return result;
}

/**
 * @brief Determines all vertices, surfaces and bodies transformed by converting a body to a vertex.
 * 
 * @param body a body to be converted to a vertex
 * @param removedVertices vertices to be removed, excluding those removed by replacing surfaces and vertices
 * @param removedSurfaces surfaces to be removed, excluding those removed by replacing surfaces
 * @param removedBodies bodies to be removed
 * @param replacedVertices vertices to be replaced
 * @param replacedSurfaces surfaces to be replaced
 */
static HRESULT transformedByB2V(
    const Body *body, 
    std::unordered_set<Vertex*> &removedVertices, 
    std::unordered_set<Surface*> &removedSurfaces, 
    std::unordered_set<Body*> &removedBodies, 
    std::unordered_map<Surface*, std::unordered_set<Vertex*> > &replacementMap) 
{
    std::unordered_set<Body*> _removedBodies = {const_cast<Body*>(body)};
    std::unordered_set<Vertex*> _removedVertices;
    std::unordered_set<Surface*> _removedSurfaces;

    definedBy(_removedBodies, _removedVertices, _removedSurfaces);
    size_t num_surfs = _removedSurfaces.size();
    std::unordered_map<Surface*, std::unordered_set<Vertex*> > _replacementMap = mapSurfaceReplacementsByB2V(_removedBodies, _removedSurfaces);

    while(_removedSurfaces.size() > num_surfs) {
        num_surfs = _removedSurfaces.size();
        _removedBodies = removedChildrenByRemovedParents(_removedSurfaces);
        definedBy(_removedBodies, _removedVertices, _removedSurfaces);
        _replacementMap = mapSurfaceReplacementsByB2V(_removedBodies, _removedSurfaces);
    }

    for(auto &v : _removedVertices) 
        removedVertices.insert(v);
    for(auto &s : _removedSurfaces) 
        removedSurfaces.insert(s);
    for(auto &b : _removedBodies) 
        removedBodies.insert(b);
    for(auto &p : _replacementMap) 
        replacementMap.insert(p);

    return S_OK;
}

/**
 * @brief Determines all vertices removed by converting a body to a vertex.
 * 
 * @param body a body to be converted to a vertex
 */
static std::unordered_set<Vertex*> removedVerticesByB2V(const Body *body) {
    std::unordered_set<Body*> _removedBodies = {const_cast<Body*>(body)};
    std::unordered_set<Vertex*> _removedVertices;
    std::unordered_set<Surface*> _removedSurfaces;

    definedBy(_removedBodies, _removedVertices, _removedSurfaces);
    size_t num_surfs = _removedSurfaces.size();
    mapSurfaceReplacementsByB2V(_removedBodies, _removedSurfaces);

    while(_removedSurfaces.size() > num_surfs) {
        num_surfs = _removedSurfaces.size();
        _removedBodies = removedChildrenByRemovedParents(_removedSurfaces);
        definedBy(_removedBodies, _removedVertices, _removedSurfaces);
        mapSurfaceReplacementsByB2V(_removedBodies, _removedSurfaces);
    }

    return _removedVertices;
}

/**
 * @brief Determines all surfaces removed by converting a body to a vertex.
 * 
 * @param toRemove a body to be converted to a vertex
 */
static std::unordered_set<Surface*> removedSurfacesByB2V(const Body *body) {
    std::unordered_set<Body*> _removedBodies = {const_cast<Body*>(body)};
    std::unordered_set<Vertex*> _removedVertices;
    std::unordered_set<Surface*> _removedSurfaces;

    definedBy(_removedBodies, _removedVertices, _removedSurfaces);
    size_t num_surfs = _removedSurfaces.size();
    mapSurfaceReplacementsByB2V(_removedBodies, _removedSurfaces);

    while(_removedSurfaces.size() > num_surfs) {
        num_surfs = _removedSurfaces.size();
        _removedBodies = removedChildrenByRemovedParents(_removedSurfaces);
        definedBy(_removedBodies, _removedVertices, _removedSurfaces);
        mapSurfaceReplacementsByB2V(_removedBodies, _removedSurfaces);
    }

    return _removedSurfaces;
}

/**
 * @brief Determines all bodies necessarily invalidated by converting a body to a vertex.
 * 
 * @param toRemove a body to be converted to a vertex
 */
static std::unordered_set<Body*> removedBodiesByB2V(const Body *body) {
    std::unordered_set<Body*> _removedBodies = {const_cast<Body*>(body)};
    std::unordered_set<Vertex*> _removedVertices;
    std::unordered_set<Surface*> _removedSurfaces;

    definedBy(_removedBodies, _removedVertices, _removedSurfaces);
    size_t num_surfs = _removedSurfaces.size();
    mapSurfaceReplacementsByB2V(_removedBodies, _removedSurfaces);

    while(_removedSurfaces.size() > num_surfs) {
        num_surfs = _removedSurfaces.size();
        _removedBodies = removedChildrenByRemovedParents(_removedSurfaces);
        definedBy(_removedBodies, _removedVertices, _removedSurfaces);
        mapSurfaceReplacementsByB2V(_removedBodies, _removedSurfaces);
    }

    return _removedBodies;
}

static std::unordered_map<Surface*, std::unordered_set<Vertex*> > surfaceReplacementsByB2V(const Body *body) 
{
    std::unordered_set<Body*> _removedBodies = {const_cast<Body*>(body)};
    std::unordered_set<Vertex*> _removedVertices;
    std::unordered_set<Surface*> _removedSurfaces;

    definedBy(_removedBodies, _removedVertices, _removedSurfaces);
    size_t num_surfs = _removedSurfaces.size();
    std::unordered_map<Surface*, std::unordered_set<Vertex*> > _replacementMap = mapSurfaceReplacementsByB2V(_removedBodies, _removedSurfaces);

    while(_removedSurfaces.size() > num_surfs) {
        num_surfs = _removedSurfaces.size();
        _removedBodies = removedChildrenByRemovedParents(_removedSurfaces);
        definedBy(_removedBodies, _removedVertices, _removedSurfaces);
        _replacementMap = mapSurfaceReplacementsByB2V(_removedBodies, _removedSurfaces);
    }

    return _replacementMap;
}

/**
 * @brief Determines all vertices removed by converting a surface to a vertex.
 * 
 * Does not account for removing bodies.
 * 
 * @param toRemove a surface to be converted to a vertex
 */
static std::unordered_set<Vertex*> removedVerticesByS2V(const Surface *toRemove) {
    std::unordered_set<Vertex*> result;
    definedBy(removedSurfacesByS2V(toRemove), result);
    return result;
}


};


#endif // _MODELS_VERTEX_SOLVER_TF_MESH_OPS_H_