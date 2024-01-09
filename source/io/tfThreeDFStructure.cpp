/*******************************************************************************
 * This file is part of Tissue Forge.
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

#include <assimp/Importer.hpp>
#include <assimp/Exporter.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <Magnum/Trade/MeshData.h>
#include <Magnum/Primitives/Icosphere.h>

#include <tfLogger.h>
#include <tfError.h>
#include <types/tfVector3.h>

#include <algorithm>
#include <limits>
#include <numeric>
#include <random>

#include "tfThreeDFStructure.h"
#include "tfTaskScheduler.h"


namespace TissueForge {


    FMatrix4 cast(const aiMatrix4x4 &m) {
        return FMatrix4(
            {m.a1, m.a2, m.a3, m.a4}, 
            {m.b1, m.b2, m.b3, m.b4}, 
            {m.c1, m.c2, m.c3, m.c4}, 
            {m.d1, m.d2, m.d3, m.d4}
        );
    }

    aiMatrix4x4 cast(const FMatrix4 &m) {
        return aiMatrix4x4(
            m[0][0], m[0][1], m[0][2], m[0][3], 
            m[1][0], m[1][1], m[1][2], m[1][3], 
            m[2][0], m[2][1], m[2][2], m[2][3], 
            m[3][0], m[3][1], m[3][2], m[3][3]
        );
    }

    template <>
    FVector3 cast(const aiVector3D &v) {
        return {v.x, v.y, v.z};
    }

    template <>
    aiVector3D cast(const FVector3 &v) {
        return {(ai_real)v.x(), (ai_real)v.y(), (ai_real)v.z()};
    }

    template <>
    FVector3 cast(const aiColor3D &v) {
        return {v.r, v.g, v.b};
    }

    template <>
    aiColor3D cast(const FVector3 &v) {
        return {(ai_real)v.x(), (ai_real)v.y(), (ai_real)v.z()};
    }

};


const aiScene *getScene(Assimp::Importer &importer, const std::string &filePath, unsigned int *pFlags=NULL) {
    unsigned int _pFlags;
    if(pFlags == NULL) {
        _pFlags = aiProcess_ValidateDataStructure | 
                  aiProcess_JoinIdenticalVertices | 
                  aiProcess_GenNormals;
    }
    else _pFlags = *pFlags;

    return importer.ReadFile(filePath, _pFlags);
}


namespace TissueForge::io {


    ThreeDFStructure::~ThreeDFStructure() {
        for(auto v : this->inventory.vertices) 
            this->remove(v);

        this->flush();
    }

    HRESULT getGlobalTransform(aiNode *node, aiMatrix4x4 *result) {
        if(node->mParent == NULL) 
            return S_OK;

        aiMatrix4x4 nt = aiMatrix4x4(node->mTransformation);
        getGlobalTransform(node->mParent, &nt);
        *result = *result * nt;
        return S_OK;
    }

    std::vector<unsigned int> buildVertexConsolidation(const std::vector<FVector3>& vPos, const FloatP_t &tol=1E-6) {

        unsigned int numV = vPos.size();
        std::vector<unsigned int> result(numV);
        std::iota(result.begin(), result.end(), 0);

        auto func = [&vPos, &result, &tol, &numV](int tid) -> void {
            const FloatP_t& tol2 = tol * tol;
            for(unsigned int vIdx = tid; vIdx < numV; vIdx += ThreadPool::size()) {
                auto vp = vPos[vIdx];

                for(unsigned int vvIdx = 0; vvIdx < vIdx; vvIdx++) {
                    if((vp - vPos[vvIdx]).dot() < tol2) {
                        result[vIdx] = vvIdx;
                        break;
                    }
                }
            }
        };
        parallel_for(ThreadPool::size(), func);

        TF_Log(LOG_TRACE) << "Consolidated vertices";

        return result;
    }

    /**
     * @brief Calculate face normals from vertices
     * 
     * @param vIndices 
     * @param vPos 
     * @return std::vector<FVector3> 
     */
    FVector3 faceNormal(const std::vector<unsigned int>& vIndices, const std::vector<FVector3>& vPos, const std::vector<FVector3>& vNorm) {
        FVector3 vap = vPos[vIndices[0]],  vbp = vPos[vIndices[1]],  vcp = vPos[vIndices[2]];
        FVector3 vbn = vNorm[vIndices[1]];

        FVector3 result = Magnum::Math::cross(vcp - vbp, vap - vbp);
        FloatP_t result_len = result.length();
        if(result_len == 0.f) 
            tf_error(E_FAIL, "Division by zero");
        result = result / result_len;

        // Check orientation

        if(result.dot(vbn) < 0) 
            result *= -1.0f;

        return result;
    }

    HRESULT ThreeDFStructure::fromFile(const std::string &filePath) {
        
        TF_Log(LOG_DEBUG) << "Importing " << filePath;

        Assimp::Importer importer;
        auto scene = getScene(importer, filePath);
        if(scene == NULL) {
            std::string msg("Import failed (" + filePath + ")");
            tf_error(E_FAIL, msg.c_str());
        }

        if(!scene->HasMeshes()) { 
            TF_Log(LOG_DEBUG) << "No meshes found";
            return S_OK;
        }

        // Get transforms to global frame
        // Either there is one root node, or a node hierarchy

        TF_Log(LOG_DEBUG) << "Getting transforms";

        auto numNodes = scene->mRootNode->mNumChildren;
        auto numMeshes = scene->mNumMeshes;
        std::vector<FMatrix4> meshTransforms(scene->mNumMeshes);
        if(numNodes > 0) {
            TF_Log(LOG_TRACE);

            for(unsigned int nIdx = 0; nIdx < numNodes; nIdx++) {
                auto node = scene->mRootNode->mChildren[nIdx];
                for(unsigned int nmIdx = 0; nmIdx < node->mNumMeshes; nmIdx++) {
                    aiMatrix4x4 mt;
                    getGlobalTransform(node, &mt);
                    meshTransforms[node->mMeshes[nmIdx]] = cast(mt);
                }
            }
        }
        else {
            TF_Log(LOG_TRACE);

            for(unsigned int nmIdx = 0; nmIdx < scene->mNumMeshes; nmIdx++) {
                aiMatrix4x4 mt;
                getGlobalTransform(scene->mRootNode, &mt);
                meshTransforms[scene->mRootNode->mMeshes[nmIdx]] = cast(mt);
            }
        }

        // Build data
        TF_Log(LOG_DEBUG) << "Building data";

        for(unsigned int mIdx = 0; mIdx < numMeshes; mIdx++) {

            const aiMesh *aim = scene->mMeshes[mIdx];
            auto meshTransform = meshTransforms[mIdx];
            
            TF_Log(LOG_TRACE) << "Mesh " << mIdx << ": " << aim->mNumVertices << " vertices, " << aim->mNumFaces << " faces";

            // Construct and add every vertex while keeping original ordering for future indexing
            std::vector<ThreeDFVertexData*> vertices(aim->mNumVertices);
            std::vector<FVector3> vPos(aim->mNumVertices), vNorm(aim->mNumVertices);
            parallel_for(
                aim->mNumVertices, 
                [&aim, &meshTransform, &vertices, &vNorm, &vPos](int vIdx) -> void {
                    auto aiv = aim->mVertices[vIdx];
                    auto ain = aim->mNormals[vIdx];
                    
                    FVector4 ait = {(FloatP_t)aiv.x, (FloatP_t)aiv.y, (FloatP_t)aiv.z, 1.f};
                    FVector3 position = (meshTransform * ait).xyz();
                    ThreeDFVertexData* vertex = new ThreeDFVertexData(position);

                    vPos[vIdx] = position;

                    FVector3 aiu = {(FloatP_t)ain.x, (FloatP_t)ain.y, (FloatP_t)ain.z};
                    FVector3 normal = meshTransform.rotation() * aiu;
                    vNorm[vIdx] = normal;
                    
                    vertices[vIdx] = vertex;
                }
            );

            // Consolidate

            auto vcIndices = buildVertexConsolidation(vPos);
            for(unsigned int vIdx = 0; vIdx < vPos.size(); vIdx++) 
                if(vcIndices[vIdx] == vIdx) 
                    this->add(vertices[vIdx]);
                else 
                    delete vertices[vIdx];

            // If there are faces, construct them and their edges and add them
            if(aim->HasFaces()) {
                TF_Log(LOG_TRACE) << "Importing " << std::to_string(aim->mNumFaces) << " faces";

                ThreeDFMeshData *mesh = new ThreeDFMeshData();
                mesh->name = std::string(aim->mName.C_Str());

                // Build and add faces and edges

                std::vector<ThreeDFFaceData*> newFaces(aim->mNumFaces, 0);
                std::vector<std::vector<ThreeDFVertexData*> > newFacesVertexData(aim->mNumFaces);
                std::vector<std::vector<unsigned int> > newFacesVertexIndexData(aim->mNumFaces);
                std::vector<std::vector<ThreeDFEdgeData*> > newFacesEdgeData(aim->mNumFaces);

                parallel_for(
                    aim->mNumFaces, 
                    [&newFacesVertexData, &newFacesVertexIndexData, &newFaces, &aim, &vertices, &vcIndices](int fIdx) -> void {
                        auto aif = aim->mFaces[fIdx];

                        std::vector<ThreeDFVertexData*>& vertices_f = newFacesVertexData[fIdx];
                        std::vector<unsigned int>& fvIndices = newFacesVertexIndexData[fIdx];
                        vertices_f.reserve(aif.mNumIndices + 1);
                        fvIndices.reserve(aif.mNumIndices);
                        for(unsigned int fvIdx = 0; fvIdx < aif.mNumIndices; fvIdx++) { 
                            unsigned int vIdx = aif.mIndices[fvIdx];
                            vertices_f.push_back(vertices[vcIndices[vIdx]]);
                            fvIndices.push_back(vcIndices[vIdx]);
                        }

                        vertices_f.push_back(vertices_f[0]);
                        newFaces[fIdx] = new ThreeDFFaceData();
                    }
                );

                std::vector<std::tuple<ThreeDFVertexData*, ThreeDFVertexData*, unsigned int> > _newFacesNewEdgeData;
                std::vector<std::vector<std::tuple<ThreeDFVertexData*, ThreeDFVertexData*, unsigned int> > > newFacesNewEdgeData(ThreadPool::size());
                parallel_for(
                    ThreadPool::size(), 
                    [&newFacesEdgeData, &newFacesNewEdgeData, &newFacesVertexData, &newFaces, &aim](int tid) -> void {
                        std::vector<std::tuple<ThreeDFVertexData*, ThreeDFVertexData*, unsigned int> >& newFacesNewEdgeDataThread = newFacesNewEdgeData[tid];

                        for(unsigned int fIdx = tid; fIdx < aim->mNumFaces; fIdx += ThreadPool::size()) {
                            auto aif = aim->mFaces[fIdx];

                            std::vector<ThreeDFVertexData*> vertices_f = newFacesVertexData[fIdx];

                            ThreeDFVertexData *va, *vb;
                            ThreeDFEdgeData *edge;
                            ThreeDFFaceData *face = newFaces[fIdx];
                            for(unsigned int fvIdx = 0; fvIdx < aif.mNumIndices; fvIdx++) {
                                va = vertices_f[fvIdx];
                                vb = vertices_f[fvIdx + 1];
                                edge = NULL;
                                for(auto e : va->edges) 
                                    if(e->has(vb)) {
                                        edge = e;
                                        newFacesEdgeData[fIdx].push_back(e);
                                        break;
                                    }
                                if(edge == NULL) {
                                    newFacesNewEdgeDataThread.push_back({va, vb, fIdx});
                                }
                            }
                        }
                    }
                );

                for(auto& newFacesNewEdgeDataThread : newFacesNewEdgeData) 
                    for(auto& newEdgeData : newFacesNewEdgeDataThread) {
                        ThreeDFVertexData* va, *vb;
                        unsigned int fIdx;
                        std::tie(va, vb, fIdx) = newEdgeData;
                        ThreeDFEdgeData* edge = new ThreeDFEdgeData(va, vb);
                        this->add(edge);
                        newFacesEdgeData[fIdx].push_back(edge);
                    }

                for(unsigned int fIdx = 0; fIdx < aim->mNumFaces; fIdx++) {
                    ThreeDFFaceData *face = newFaces[fIdx];
                    for(auto& edge : newFacesEdgeData[fIdx]) {
                        face->edges.push_back(edge);
                        edge->faces.push_back(face);
                    }
                    mesh->faces.push_back(face);
                    face->meshes.push_back(mesh);
                    this->add(face);
                }

                parallel_for(
                    aim->mNumFaces, 
                    [&newFaces, &vPos, &vNorm, &newFacesVertexIndexData](int fIdx) -> void {
                        newFaces[fIdx]->normal = faceNormal(newFacesVertexIndexData[fIdx], vPos, vNorm);
                    }
                );

                // Add final data for this mesh

                this->add(mesh);

            }

        }

        TF_Log(LOG_INFORMATION) << "Successfully imported " << filePath;

        return S_OK;
    }

    std::vector<ThreeDFVertexData*> assembleFaceVertices(const std::vector<ThreeDFEdgeData*> &edges) {
        auto numEdges = edges.size();
        if(numEdges < 3) 
            tf_error(E_FAIL, "Invalid face definition from edges");
        auto numVertices = numEdges + 1;

        std::vector<bool> edgeIntegrated(edges.size(), false);
        std::vector<ThreeDFVertexData*> result(numVertices, 0);

        result[0] = edges[0]->va;
        result[1] = edges[0]->vb;
        edgeIntegrated[0] = true;

        ThreeDFVertexData *va = result[1];
        for(unsigned int vIdx = 2; vIdx < numVertices; vIdx++) {
            
            bool foundV = false;

            for(unsigned int eIdx = 0; eIdx < numEdges; eIdx++) {

                if(edgeIntegrated[eIdx]) 
                    continue;

                auto e = edges[eIdx];
                if(e->va == va) { 
                    foundV = true;
                    edgeIntegrated[eIdx] = true;
                    va = e->vb;
                    result[vIdx] = va;
                    break;
                } 
                else if(e->vb == va) { 
                    foundV = true;
                    edgeIntegrated[eIdx] = true;
                    va = e->va;
                    result[vIdx] = va;
                    break;
                }

            }

            if(!foundV) 
                tf_error(E_FAIL, "Face assembly failed");
        }

        // Validate that last is first, then remove last

        if(result[0] != result[result.size() - 1]) 
            tf_error(E_FAIL, "Face result error");

        result.pop_back();

        return result;
    }

    HRESULT naiveNormalsCheck(std::vector<FVector3> &positions, std::vector<FVector3> &normals) {
        unsigned int i, numPos = positions.size();

        if(numPos != normals.size()) 
            tf_error(E_FAIL, "Positions and normals differ in size");

        // Build indices
        
        std::vector<unsigned int> iA, iB, iC;
        for(i = 0; i < positions.size(); i++) {

            iA.push_back(i-1); 
            iB.push_back(i); 
            iC.push_back(i+1);

        }
        iA[0] = positions.size() - 1;
        iC[positions.size() - 1] = 0;

        // Calculate norms and populate/correct as necessary

        FVector3 pA, pB, pC, nB, nBCalc;
        bool flipIt = false;
        for(i = 0; i < positions.size() - 1; i++) {
            
            pA = positions[iA[i]]; 
            pB = positions[iB[i]]; 
            pC = positions[iC[i]];
            nB = normals[iB[i]];

            if(nB.length() < std::numeric_limits<FloatP_t>::epsilon()) 
                normals[i] = nBCalc;
            else {
                nBCalc = Magnum::Math::cross(pC - pB, pA - pB);
                if(nB.dot(nBCalc) < 0) {
                    flipIt = true;
                    break;
                }
            }

        }

        // Do final correction: flip order of vertices if normals face inward

        if(flipIt) {

            std::vector<FVector3> tmp_p, tmp_n;

            for(i = 0; i < numPos; i++) {
                tmp_p.push_back(positions[numPos - i - 1]);
                tmp_n.push_back(normals[numPos - i - 1]);
            }
            for(i = 0; i < numPos; i++) {
                positions[i] = tmp_p[i];
                normals[i] = tmp_n[i];
            }

        }

        return S_OK;
    }

    void uploadMesh(aiMesh *aiMesh, ThreeDFMeshData *tfMesh, const unsigned int &mIdx=0) { 
        auto tfFaces = tfMesh->getFaces();
        auto tfEdges = tfMesh->getEdges();

        std::vector<std::vector<ThreeDFVertexData*> > verticesByFace;
        verticesByFace.reserve(tfFaces.size());

        aiMesh->mName = tfMesh->name.size() > 0 ? tfMesh->name : std::string("Mesh " + std::to_string(mIdx));
        aiMesh->mPrimitiveTypes = aiPrimitiveType_TRIANGLE | aiPrimitiveType_POLYGON;
        aiMesh->mNumFaces = tfFaces.size();

        TF_Log(LOG_TRACE) << "... " << aiMesh->mNumFaces << " faces";

        unsigned int numVertices = 0;
        for(auto f : tfFaces) {
            
            auto fvertices = assembleFaceVertices(f->getEdges());
            verticesByFace.push_back(fvertices);
            numVertices += fvertices.size();

        }
        aiMesh->mNumVertices = numVertices;

        TF_Log(LOG_TRACE) << "... prepping mesh";

        TF_Log(LOG_TRACE) << "... " << aiMesh->mNumVertices << " vertices";

        // Allocate faces and vertices

        aiMesh->mFaces = new aiFace[aiMesh->mNumFaces];
        aiMesh->mVertices = new aiVector3D[aiMesh->mNumVertices];
        aiMesh->mNormals = new aiVector3D[aiMesh->mNumVertices];

        // Build faces and vertices; vertices take normals from their faces, if any

        for(unsigned int fIdx = 0, vIdx = 0; fIdx < aiMesh->mNumFaces; fIdx++) {

            aiFace &face = aiMesh->mFaces[fIdx];
            ThreeDFFaceData *tfFace = tfFaces[fIdx];
            
            auto fvertices = verticesByFace[fIdx];
            auto numfvertices = fvertices.size();

            face.mNumIndices = numfvertices;
            face.mIndices = new unsigned int[numfvertices];

            std::vector<FVector3> fpos, fnorm;
            for(unsigned int fvIdx = 0; fvIdx < numfvertices; fvIdx++) {
                fpos.push_back(fvertices[fvIdx]->position);
                fnorm.push_back(tfFaces[fIdx]->normal);
            }

            if(naiveNormalsCheck(fpos, fnorm) != S_OK) 
                return;

            for(unsigned int fvIdx = 0; fvIdx < numfvertices; fvIdx++, vIdx++) {
                face.mIndices[fvIdx] = vIdx;
                aiMesh->mVertices[vIdx] = cast<FVector3, aiVector3D>(fpos[fvIdx]);
                aiMesh->mNormals[vIdx] = cast<FVector3, aiVector3D>(fnorm[fvIdx]);
            }

        }
    }

    aiMaterial *generate3DFMaterial(ThreeDFRenderData *renderData) {

        if(renderData == NULL) 
            tf_error(E_FAIL, "NULL render data");

        aiMaterial *mtl = new aiMaterial();

        aiColor3D color = cast<FVector3, aiColor3D>(renderData->color);

        mtl->AddProperty(&color, 1, AI_MATKEY_COLOR_DIFFUSE);

        return mtl;
    }

    HRESULT ThreeDFStructure::toFile(const std::string &format, const std::string &filePath) {

        TF_Log(LOG_DEBUG) << "Exporting " << format << ", " << filePath;

        Assimp::Exporter exporter;

        aiScene *scene = new aiScene();
        scene->mMetaData = new aiMetadata();

        // Create a root node with no child nodes

        aiNode *rootNode = new aiNode();

        // Set materials, if any; otherwise set one material

        // scene->mMaterials = new aiMaterial*[1];
        // scene->mNumMaterials = 1;
        // scene->mMaterials[0] = new aiMaterial();

        unsigned int numRenders = 0;
        auto tfMeshes = this->getMeshes();
        std::vector<aiMaterial*> meshMtls;
        std::vector<unsigned int> meshMtlIndices(tfMeshes.size(), 0);

        meshMtls.push_back(new aiMaterial());

        for(unsigned int i = 0; i < tfMeshes.size(); i++) {

            auto m = tfMeshes[i];
            if(m->renderData != NULL) {
                meshMtlIndices[i] = meshMtls.size();
                meshMtls.push_back(generate3DFMaterial(m->renderData));
            }

        }

        scene->mNumMaterials = meshMtls.size();
        scene->mMaterials = new aiMaterial*[scene->mNumMaterials];
        for(unsigned int i = 0; i < scene->mNumMaterials; i++) 
            scene->mMaterials[i] = meshMtls[i];

        // Create meshes

        scene->mNumMeshes = this->getNumMeshes();

        if(scene->mNumMeshes == 0) 
            tf_error(E_FAIL, "No data to export");

        TF_Log(LOG_TRACE) << "number of meshes: " << scene->mNumMeshes;

        scene->mMeshes = new aiMesh*[scene->mNumMeshes];
        rootNode->mNumMeshes = scene->mNumMeshes;
        rootNode->mMeshes = new unsigned int[scene->mNumMeshes];
        for(unsigned int i = 0; i < scene->mNumMeshes; i++) { 
            scene->mMeshes[i] = new aiMesh();
            scene->mMeshes[i]->mMaterialIndex = meshMtlIndices[i];
            rootNode->mMeshes[i] = i;
        }
        scene->mRootNode = rootNode;

        for(unsigned int mIdx = 0; mIdx < scene->mNumMeshes; mIdx++) {
            TF_Log(LOG_TRACE) << "generating mesh " << mIdx;
            
            uploadMesh(scene->mMeshes[mIdx], this->inventory.meshes[mIdx], mIdx);
        }

        TF_Log(LOG_TRACE) << "Exporting";

        // Export

        if(exporter.Export(scene, format, filePath) != aiReturn_SUCCESS) {
            tf_error(E_FAIL, exporter.GetErrorString());
            return E_FAIL;
        }

        // Bug in MSVC: deleting a scene in debug builds invokes some issue with aiNode destructor
        #if !defined(_MSC_VER) || !defined(_DEBUG)
        delete scene;
        #endif

        return S_OK;
    }

    HRESULT ThreeDFStructure::flush() {
        
        for(auto v : this->queueRemove.vertices) 
            delete v;

        for(auto e : this->queueRemove.edges) 
            delete e;

        for(auto f : this->queueRemove.faces) 
            delete f;

        for(auto m : this->queueRemove.meshes) 
            delete m;

        this->queueRemove.vertices.clear();
        this->queueRemove.edges.clear();
        this->queueRemove.faces.clear();
        this->queueRemove.meshes.clear();

        return S_OK;
    }

    HRESULT ThreeDFStructure::extend(const ThreeDFStructure &s) {
        this->inventory.vertices.insert(this->inventory.vertices.end(), s.inventory.vertices.begin(), s.inventory.vertices.end());
        this->inventory.edges.insert(this->inventory.edges.end(),       s.inventory.edges.begin(),    s.inventory.edges.end());
        this->inventory.faces.insert(this->inventory.faces.end(),       s.inventory.faces.begin(),    s.inventory.faces.end());
        this->inventory.meshes.insert(this->inventory.meshes.end(),     s.inventory.meshes.begin(),   s.inventory.meshes.end());

        return S_OK;
    }

    HRESULT ThreeDFStructure::clear() {
        for(auto m : this->getMeshes()) 
            this->remove(m);

        for(auto f : this->getFaces()) 
            this->remove(f);

        for(auto e : this->getEdges()) 
            this->remove(e);

        for(auto v : this->getVertices()) 
            this->remove(v);

        return this->flush();
    }

    std::vector<ThreeDFVertexData*> ThreeDFStructure::getVertices() {
        return this->inventory.vertices;
    }

    std::vector<ThreeDFEdgeData*> ThreeDFStructure::getEdges() {
        return this->inventory.edges;
    }

    std::vector<ThreeDFFaceData*> ThreeDFStructure::getFaces() {
        return this->inventory.faces;
    }

    std::vector<ThreeDFMeshData*> ThreeDFStructure::getMeshes() {
        return this->inventory.meshes;
    }

    unsigned int ThreeDFStructure::getNumVertices() {
        return this->inventory.vertices.size();
    }

    unsigned int ThreeDFStructure::getNumEdges() {
        return this->inventory.edges.size();
    }

    unsigned int ThreeDFStructure::getNumFaces() {
        return this->inventory.faces.size();
    }

    unsigned int ThreeDFStructure::getNumMeshes() {
        return this->inventory.meshes.size();
    }

    bool ThreeDFStructure::has(ThreeDFVertexData *v) {
        auto itr = std::find(this->inventory.vertices.begin(), this->inventory.vertices.end(), v);
        return itr != std::end(this->inventory.vertices);
    }

    bool ThreeDFStructure::has(ThreeDFEdgeData *e) {
        auto itr = std::find(this->inventory.edges.begin(), this->inventory.edges.end(), e);
        return itr != std::end(this->inventory.edges);
    }

    bool ThreeDFStructure::has(ThreeDFFaceData *f) {
        auto itr = std::find(this->inventory.faces.begin(), this->inventory.faces.end(), f);
        return itr != std::end(this->inventory.faces);
    }

    bool ThreeDFStructure::has(ThreeDFMeshData *m) {
        auto itr = std::find(this->inventory.meshes.begin(), this->inventory.meshes.end(), m);
        return itr != std::end(this->inventory.meshes);
    }

    void ThreeDFStructure::add(ThreeDFVertexData *v) {
        if(v->structure == this) 
            return;
        else if(v->structure != NULL) 
            tf_error(E_FAIL, "Vertex already owned by a structure");

        v->structure = this;
        v->id = this->id_vertex++;
        this->inventory.vertices.push_back(v);
    }

    void ThreeDFStructure::add(ThreeDFEdgeData *e) {
        if(e->structure == this) 
            return;
        else if(e->structure != NULL) {
            std::stringstream msg_str;
            msg_str << "Edge already owned by a structure: ";
            msg_str << e->structure;
            tf_error(E_FAIL, msg_str.str().c_str());
        }
        else if(!e->va || !e->vb) 
            tf_error(E_FAIL, "Invalid definition");

        for(auto v : e->getVertices()) 
            if(v->structure != this) 
                this->add(v);

        e->structure = this;
        e->id = this->id_edge++;
        this->inventory.edges.push_back(e);
    }

    void ThreeDFStructure::add(ThreeDFFaceData *f) {
        if(f->structure == this) 
            return;
        else if(f->structure != NULL) 
            tf_error(E_FAIL, "Face already owned by a structure");
        else if(f->edges.size() < 3) 
            tf_error(E_FAIL, "Invalid definition");

        for(auto e : f->getEdges()) 
            if(e->structure != this) 
                this->add(e);

        f->structure = this;
        f->id = this->id_face++;
        this->inventory.faces.push_back(f);
    }

    void ThreeDFStructure::add(ThreeDFMeshData *m) {
        if(m->structure == this) 
            return;
        else if(m->structure != NULL) 
            tf_error(E_FAIL, "Mesh already owned by a structure");
        else if(m->faces.size() < 3) 
            tf_error(E_FAIL, "Invalid definition");

        for(auto f : m->faces) 
            if(f->structure != this) 
                this->add(f);

        m->structure = this;
        m->id = this->id_mesh++;
        this->inventory.meshes.push_back(m);
    }

    void ThreeDFStructure::remove(ThreeDFVertexData *v) {
        if(v->structure == NULL) 
            return;
        else if(v->structure != this) 
            tf_error(E_FAIL, "Vertex owned by different structure");

        auto itr = std::find(this->inventory.vertices.begin(), this->inventory.vertices.end(), v);
        if(itr == std::end(this->inventory.vertices)) 
            tf_error(E_FAIL, "Could not find vertex");
        
        this->onRemoved(v);

        v->structure = NULL;
        this->inventory.vertices.erase(itr);
        this->queueRemove.vertices.push_back(v);
    }

    void ThreeDFStructure::remove(ThreeDFEdgeData *e) {
        if(e->structure == NULL) 
            return;
        else if(e->structure != this) 
            tf_error(E_FAIL, "Edge owned by different structure");

        auto itr = std::find(this->inventory.edges.begin(), this->inventory.edges.end(), e);
        if(itr == std::end(this->inventory.edges)) 
            tf_error(E_FAIL, "Could not find edge");
        
        this->onRemoved(e);

        e->structure = NULL;
        this->inventory.edges.erase(itr);
        this->queueRemove.edges.push_back(e);
    }

    void ThreeDFStructure::remove(ThreeDFFaceData *f) {
        if(f->structure == NULL) 
            return;
        else if(f->structure != this) 
            tf_error(E_FAIL, "Face owned by different structure");

        auto itr = std::find(this->inventory.faces.begin(), this->inventory.faces.end(), f);
        if(itr == std::end(this->inventory.faces)) 
            tf_error(E_FAIL, "Could not find face");
        
        this->onRemoved(f);

        f->structure = NULL;
        this->inventory.faces.erase(itr);
        this->queueRemove.faces.push_back(f);
    }

    void ThreeDFStructure::remove(ThreeDFMeshData *m) {
        if(m->structure == NULL) 
            return;
        else if(m->structure != this) 
            tf_error(E_FAIL, "Mesh owned by different structure");

        auto itr = std::find(this->inventory.meshes.begin(), this->inventory.meshes.end(), m);
        if(itr == std::end(this->inventory.meshes)) 
            tf_error(E_FAIL, "Could not find mesh");
        
        m->structure = NULL;
        this->inventory.meshes.erase(itr);
        this->queueRemove.meshes.push_back(m);
    }

    void ThreeDFStructure::onRemoved(ThreeDFVertexData *v) {
        for(auto e : v->getEdges()) 
            this->remove(e);
    }

    void ThreeDFStructure::onRemoved(ThreeDFEdgeData *e) {
        for(auto f : e->getFaces()) { 
            auto itr = std::find(f->edges.begin(), f->edges.end(), e);
            f->edges.erase(itr);
            if(f->edges.size() < 3) 
                this->remove(f);
        }
    }

    void ThreeDFStructure::onRemoved(ThreeDFFaceData *f) {
        for(auto m : f->getMeshes()) {
            auto itr = std::find(m->faces.begin(), m->faces.end(), f);
            m->faces.erase(itr);
            if(m->faces.size() < 3) 
                this->remove(m);
        }
    }

    FVector3 ThreeDFStructure::getCentroid() {
        auto vertices = this->getVertices();
        auto numV = vertices.size();

        if(numV == 0) 
            tf_error(E_FAIL, "No vertices");

        FVector3 result = {0.f, 0.f, 0.f};

        for(unsigned int i = 0; i < numV; i++) 
            result += vertices[i]->position;

        result /= numV;
        return result;
    }

    HRESULT ThreeDFStructure::translate(const FVector3 &displacement) {
        for(auto m : this->getMeshes()) 
            m->translate(displacement);

        return S_OK;
    }

    HRESULT ThreeDFStructure::translateTo(const FVector3 &position) {
        return this->translate(position - this->getCentroid());
    }

    HRESULT ThreeDFStructure::rotateAt(const FMatrix3 &rotMat, const FVector3 &rotPt) {
        for(auto m : this->getMeshes()) 
            m->rotateAt(rotMat, rotPt);
        return S_OK;
    }

    HRESULT ThreeDFStructure::rotate(const FMatrix3 &rotMat) {
        return this->rotateAt(rotMat, this->getCentroid());
    }

    HRESULT ThreeDFStructure::scaleFrom(const FVector3 &scales, const FVector3 &scalePt) {
        for(auto m : this->getMeshes()) {
            m->scaleFrom(scales, scalePt);
        }
        return S_OK;
    }

    HRESULT ThreeDFStructure::scaleFrom(const FloatP_t &scale, const FVector3 &scalePt) {
        return this->scaleFrom(FVector3(scale), scalePt);
    }

    HRESULT ThreeDFStructure::scale(const FVector3 &scales) {
        return this->scaleFrom(scales, this->getCentroid());
    }

    HRESULT ThreeDFStructure::scale(const FloatP_t &scale) {
        return this->scale(FVector3(scale));
    }

};
