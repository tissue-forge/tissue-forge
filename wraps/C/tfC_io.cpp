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

#include "tfC_io.h"

#include "TissueForge_c_private.h"

#include <io/tf_io.h>
#include <io/tfIO.h>
#include <io/tfFIO.h>
#include <io/tfThreeDFStructure.h>


using namespace TissueForge;


//////////////////
// Module casts //
//////////////////


namespace TissueForge { 
    

    void castC(const io::MetaData &obj, struct tfIoMetaDataHandle *handle) {
        handle->versionMajor = obj.versionMajor;
        handle->versionMinor = obj.versionMinor;
        handle->versionPatch = obj.versionPatch;
    }

    io::IOElement *castC(struct tfIoIOElementHandle *handle) {
        return castC<io::IOElement, tfIoIOElementHandle>(handle);
    }

    io::ThreeDFRenderData *castC(struct tfIoThreeDFRenderDataHandle *handle) {
        return castC<io::ThreeDFRenderData, tfIoThreeDFRenderDataHandle>(handle);
    }

    io::ThreeDFVertexData *castC(struct tfIoThreeDFVertexDataHandle *handle) {
        return castC<io::ThreeDFVertexData, tfIoThreeDFVertexDataHandle>(handle);
    }

    io::ThreeDFEdgeData *castC(struct tfIoThreeDFEdgeDataHandle *handle) {
        return castC<io::ThreeDFEdgeData, tfIoThreeDFEdgeDataHandle>(handle);
    }

    io::ThreeDFFaceData *castC(struct tfIoThreeDFFaceDataHandle *handle) {
        return castC<io::ThreeDFFaceData, tfIoThreeDFFaceDataHandle>(handle);
    }

    io::ThreeDFMeshData *castC(struct tfIoThreeDFMeshDataHandle *handle) {
        return castC<io::ThreeDFMeshData, tfIoThreeDFMeshDataHandle>(handle);
    }

    io::ThreeDFStructure *castC(struct tfIoThreeDFStructureHandle *handle) {
        return castC<io::ThreeDFStructure, tfIoThreeDFStructureHandle>(handle);
    }

}

#define TFC_IOELEMENTHANDLE_GET(handle, varname) \
    io::IOElement *varname = TissueForge::castC<io::IOElement, tfIoIOElementHandle>(handle); \
    if(!varname) \
        return E_FAIL;

#define TFC_3DFRENDERDATAHANDLE_GET(handle, varname) \
    io::ThreeDFRenderData *varname = TissueForge::castC<io::ThreeDFRenderData, tfIoThreeDFRenderDataHandle>(handle); \
    if(!varname) \
        return E_FAIL;

#define TFC_3DFVERTEXDATAHANDLE_GET(handle, varname) \
    io::ThreeDFVertexData *varname = TissueForge::castC<io::ThreeDFVertexData, tfIoThreeDFVertexDataHandle>(handle); \
    if(!varname) \
        return E_FAIL;

#define TFC_3DFEDGEDATAHANDLE_GET(handle, varname) \
    io::ThreeDFEdgeData *varname = TissueForge::castC<io::ThreeDFEdgeData, tfIoThreeDFEdgeDataHandle>(handle); \
    if(!varname) \
        return E_FAIL;

#define TFC_3DFFACEDATAHANDLE_GET(handle, varname) \
    io::ThreeDFFaceData *varname = TissueForge::castC<io::ThreeDFFaceData, tfIoThreeDFFaceDataHandle>(handle); \
    if(!varname) \
        return E_FAIL;

#define TFC_3DFMESHDATAHANDLE_GET(handle, varname) \
    io::ThreeDFMeshData *varname = TissueForge::castC<io::ThreeDFMeshData, tfIoThreeDFMeshDataHandle>(handle); \
    if(!varname) \
        return E_FAIL;

#define TFC_3DFSTRUCTUREHANDLE_GET(handle, varname) \
    io::ThreeDFStructure *varname = TissueForge::castC<io::ThreeDFStructure, tfIoThreeDFStructureHandle>(handle); \
    if(!varname) \
        return E_FAIL;


//////////////
// Generics //
//////////////


namespace TissueForge::capi {


    template <typename O, typename H> 
    HRESULT meshObj_hasStructure(H *handle, bool *hasStructure) {
        O *obj = TissueForge::castC(handle);
        TFC_PTRCHECK(obj);
        TFC_PTRCHECK(hasStructure);
        *hasStructure = obj->structure != NULL;
        return S_OK;
    }

    template <typename O, typename H> 
    HRESULT meshObj_getStructure(H *handle, struct tfIoThreeDFStructureHandle *structure) {
        O *obj = TissueForge::castC(handle);
        TFC_PTRCHECK(obj); TFC_PTRCHECK(obj->structure);
        TFC_PTRCHECK(structure);
        structure->tfObj = (void*)obj->structure;
        return S_OK;
    }

    template <typename O, typename H> 
    HRESULT meshObj_setStructure(H *handle, struct tfIoThreeDFStructureHandle *structure) {
        TFC_PTRCHECK(handle);
        TFC_PTRCHECK(structure);
        O *obj = TissueForge::castC(handle);
        io::ThreeDFStructure *_structure = TissueForge::castC(structure);
        TFC_PTRCHECK(obj); TFC_PTRCHECK(_structure);
        obj->structure = _structure;
        return S_OK;
    }

    template <typename O, typename H>
    HRESULT copyMeshObjFromVec(const std::vector<O*> &vec, H **arr, unsigned int *numArr) {
        TFC_PTRCHECK(arr);
        TFC_PTRCHECK(numArr);
        *numArr = vec.size();
        if(*numArr > 0) {
            H *_arr = (H*)malloc(*numArr * sizeof(H));
            if(!_arr) 
                return E_OUTOFMEMORY;
            for(unsigned int i = 0; i < *numArr; i++) 
                _arr[i].tfObj = (void*)vec[i];
            *arr = _arr;
        }
        return S_OK;
    }

    template <typename O, typename H>
    HRESULT copyMeshObjToVec(std::vector<O*> &vec, H *arr, unsigned int numArr) {
        TFC_PTRCHECK(arr);
        vec.clear();
        H _el;
        for(unsigned int i = 0; i < numArr; i++) {
            _el = arr[i];
            if(!_el.tfObj) 
                return E_FAIL;
            vec.push_back((O*)_el.tfObj);
        }
        return S_OK;
    }

}


////////////////////////
// Function factories //
////////////////////////


struct tfIoFIOModule : io::FIOModule {
    std::string _moduleName;
    tfIoFIOModuleToFileFcn _toFile;
    tfIoFIOModuleFromFileFcn _fromFile;

    std::string moduleName() { return this->_moduleName; }

    HRESULT toFile(const io::MetaData &metaData, io::IOElement &fileElement) {
        tfIoMetaDataHandle _metaData;
        tfIoIOElementHandle _fileElement;
        TissueForge::castC(metaData, &_metaData);
        TissueForge::castC(fileElement, &_fileElement);
        return this->_toFile(_metaData, &_fileElement);
    }

    HRESULT fromFile(const io::MetaData &metaData, const io::IOElement &fileElement) {
        tfIoMetaDataHandle _metaData;
        tfIoIOElementHandle _fileElement;
        TissueForge::castC(metaData, &_metaData);
        TissueForge::castC(fileElement, &_fileElement);
        return this->_fromFile(_metaData, _fileElement);
    }
};


////////////////////////
// io::MetaDataHandle //
////////////////////////


HRESULT tfIoMetaData_init(struct tfIoMetaDataHandle *handle) {
    io::MetaData md;
    handle->versionMajor = md.versionMajor;
    handle->versionMinor = md.versionMinor;
    handle->versionPatch = md.versionPatch;
    return S_OK;
}


////////////////////////
// io::FIOStorageKeys //
////////////////////////


HRESULT tfIoFIOStorageKeys_init(struct tfIoFIOStorageKeysHandle *handle) {
    TFC_PTRCHECK(handle);

    handle->KEY_TYPE = new char(io::FIO::KEY_TYPE.length());
    std::strcpy(handle->KEY_TYPE, io::FIO::KEY_TYPE.c_str());

    handle->KEY_VALUE = new char(io::FIO::KEY_VALUE.length());
    std::strcpy(handle->KEY_VALUE, io::FIO::KEY_VALUE.c_str());

    handle->KEY_METADATA = new char(io::FIO::KEY_METADATA.length());
    std::strcpy(handle->KEY_METADATA, io::FIO::KEY_METADATA.c_str());

    handle->KEY_SIMULATOR = new char(io::FIO::KEY_SIMULATOR.length());
    std::strcpy(handle->KEY_SIMULATOR, io::FIO::KEY_SIMULATOR.c_str());

    handle->KEY_UNIVERSE = new char(io::FIO::KEY_UNIVERSE.length());
    std::strcpy(handle->KEY_UNIVERSE, io::FIO::KEY_UNIVERSE.c_str());

    handle->KEY_MODULES = new char(io::FIO::KEY_MODULES.length());
    std::strcpy(handle->KEY_MODULES, io::FIO::KEY_MODULES.c_str());

    return S_OK;
}

HRESULT tfIoFIOStorageKeys_destroy(struct tfIoFIOStorageKeysHandle *handle) {
    delete handle->KEY_TYPE;
    delete handle->KEY_VALUE;
    delete handle->KEY_METADATA;
    delete handle->KEY_SIMULATOR;
    delete handle->KEY_UNIVERSE;
    delete handle->KEY_MODULES;

    handle->KEY_TYPE = NULL;
    handle->KEY_VALUE = NULL;
    handle->KEY_METADATA = NULL;
    handle->KEY_SIMULATOR = NULL;
    handle->KEY_UNIVERSE = NULL;
    handle->KEY_MODULES = NULL;
    return S_OK;
}


///////////////////////////
// io::ThreeDFRenderData //
///////////////////////////


HRESULT tfIoThreeDFRenderData_init(struct tfIoThreeDFRenderDataHandle *handle) {
    io::ThreeDFRenderData *rd = new io::ThreeDFRenderData();
    handle->tfObj = (void*)rd;
    return S_OK;
}

HRESULT tfIoThreeDFRenderData_destroy(struct tfIoThreeDFRenderDataHandle *handle) {
    return TissueForge::capi::destroyHandle<io::ThreeDFRenderData, tfIoThreeDFRenderDataHandle>(handle) ? S_OK : E_FAIL;
}

HRESULT tfIoThreeDFRenderData_getColor(struct tfIoThreeDFRenderDataHandle *handle, tfFloatP_t **color) {
    TFC_3DFRENDERDATAHANDLE_GET(handle, rd);
    if(!color) 
        return E_FAIL;
    TFC_VECTOR3_COPYFROM(rd->color, (*color));
    return S_OK;
}

HRESULT tfIoThreeDFRenderData_setColor(struct tfIoThreeDFRenderDataHandle *handle, tfFloatP_t *color) {
    TFC_3DFRENDERDATAHANDLE_GET(handle, rd);
    if(!color) 
        return E_FAIL;
    TFC_VECTOR3_COPYTO(color, rd->color);
    return S_OK;
}


///////////////////
// io::IOElement //
///////////////////


HRESULT tfIoIOElement_init(struct tfIoIOElementHandle *handle) {
    io::IOElement *ioel = new io::IOElement();
    handle->tfObj = (void*)ioel;
    return S_OK;
}

HRESULT tfIoIOElement_destroy(struct tfIoIOElementHandle *handle) {
    return TissueForge::capi::destroyHandle<io::IOElement, tfIoIOElementHandle>(handle) ? S_OK : E_FAIL;
}

HRESULT tfIoIOElement_getType(struct tfIoIOElementHandle *handle, char **type, unsigned int *numChars) {
    TFC_IOELEMENTHANDLE_GET(handle, ioel);
    TFC_PTRCHECK(numChars);
    TissueForge::capi::str2Char(ioel->get()->type, type, numChars);
    return S_OK;
}

HRESULT tfIoIOElement_setType(struct tfIoIOElementHandle *handle, const char *type) {
    TFC_IOELEMENTHANDLE_GET(handle, ioel);
    ioel->get()->type = type;
    return S_OK;
}

HRESULT tfIoIOElement_getValue(struct tfIoIOElementHandle *handle, char **value, unsigned int *numChars) {
    TFC_IOELEMENTHANDLE_GET(handle, ioel);
    TFC_PTRCHECK(value);
    TFC_PTRCHECK(numChars);
    return TissueForge::capi::str2Char(ioel->get()->value, value, numChars);
}

HRESULT tfIoIOElement_setValue(struct tfIoIOElementHandle *handle, const char *value) {
    TFC_IOELEMENTHANDLE_GET(handle, ioel);
    TFC_PTRCHECK(value);
    ioel->get()->value = value;
    return S_OK;
}

HRESULT tfIoIOElement_hasParent(struct tfIoIOElementHandle *handle, bool *hasParent) {
    TFC_IOELEMENTHANDLE_GET(handle, ioel);
    TFC_PTRCHECK(hasParent);
    *hasParent = ioel->get()->parent.get()->type.size() > 0;
    return S_OK;
}

HRESULT tfIoIOElement_getParent(struct tfIoIOElementHandle *handle, struct tfIoIOElementHandle *parent) {
    TFC_IOELEMENTHANDLE_GET(handle, ioel);
    TFC_PTRCHECK(parent);
    TissueForge::io::IOElement *_parent = new TissueForge::io::IOElement(ioel->get()->parent);
    parent->tfObj = (void*)_parent;
    return S_OK;
}

HRESULT tfIoIOElement_setParent(struct tfIoIOElementHandle *handle, struct tfIoIOElementHandle *parent) {
    TFC_IOELEMENTHANDLE_GET(handle, ioel);
    TFC_IOELEMENTHANDLE_GET(parent, pioel);
    ioel->get()->parent = *pioel;
    return S_OK;
}

HRESULT tfIoIOElement_getNumChildren(struct tfIoIOElementHandle *handle, unsigned int *numChildren) {
    TFC_IOELEMENTHANDLE_GET(handle, ioel);
    TFC_PTRCHECK(numChildren);
    *numChildren = ioel->get()->children.size();
    return S_OK;
}

HRESULT tfIoIOElement_getKeys(struct tfIoIOElementHandle *handle, char ***keys) {
    TFC_IOELEMENTHANDLE_GET(handle, ioel);
    if(!keys) 
        return E_FAIL;
    auto numChildren = ioel->get()->children.size();
    if(numChildren > 0) {
        char **_keys = (char**)malloc(numChildren * sizeof(char*));
        if(!_keys) 
            return E_OUTOFMEMORY;
        unsigned int i = 0;
        for(auto &itr : ioel->get()->children) {
            char *_c = new char[itr.first.size() + 1];
            std::strcpy(_c, itr.first.c_str());
            _keys[i] = _c;
            i++;
        }
        *keys = _keys;
    }
    return S_OK;
}

HRESULT tfIoIOElement_getChild(struct tfIoIOElementHandle *handle, const char *key, struct tfIoIOElementHandle *child) {
    TFC_IOELEMENTHANDLE_GET(handle, ioel);
    TFC_PTRCHECK(child);
    auto itr = ioel->get()->children.find(key);
    if(itr == ioel->get()->children.end()) 
        return E_FAIL;
    TissueForge::io::IOElement *_child = new TissueForge::io::IOElement(itr->second);
    child->tfObj = (void*)_child;
    return S_OK;
}

HRESULT tfIoIOElement_setChild(struct tfIoIOElementHandle *handle, const char *key, struct tfIoIOElementHandle *child) {
    TFC_IOELEMENTHANDLE_GET(handle, ioel);
    TFC_PTRCHECK(key);
    TFC_IOELEMENTHANDLE_GET(child, cioel);
    ioel->get()->children.insert({key, *cioel});
    return S_OK;
}


///////////////////////////
// io::ThreeDFVertexData //
///////////////////////////


HRESULT tfIoThreeDFVertexData_init(struct tfIoThreeDFVertexDataHandle *handle, tfFloatP_t *position) {
    if(!position) 
        return E_FAIL;
    io::ThreeDFVertexData *vert = new io::ThreeDFVertexData(FVector3::from(position));
    handle->tfObj = (void*)vert;
    return S_OK;
}

HRESULT tfIoThreeDFVertexData_destroy(struct tfIoThreeDFVertexDataHandle *handle) {
    return TissueForge::capi::destroyHandle<io::ThreeDFVertexData, tfIoThreeDFVertexDataHandle>(handle) ? S_OK : E_FAIL;
}

HRESULT tfIoThreeDFVertexData_hasStructure(struct tfIoThreeDFVertexDataHandle *handle, bool *hasStructure) {
    return TissueForge::capi::meshObj_hasStructure<io::ThreeDFVertexData, tfIoThreeDFVertexDataHandle>(handle, hasStructure);
}

HRESULT tfIoThreeDFVertexData_getStructure(struct tfIoThreeDFVertexDataHandle *handle, struct tfIoThreeDFStructureHandle *structure) {
    return TissueForge::capi::meshObj_getStructure<io::ThreeDFVertexData, tfIoThreeDFVertexDataHandle>(handle, structure);
}

HRESULT tfIoThreeDFVertexData_setStructure(struct tfIoThreeDFVertexDataHandle *handle, struct tfIoThreeDFStructureHandle *structure) {
    return TissueForge::capi::meshObj_setStructure<io::ThreeDFVertexData, tfIoThreeDFVertexDataHandle>(handle, structure);
}

HRESULT tfIoThreeDFVertexData_getPosition(struct tfIoThreeDFVertexDataHandle *handle, tfFloatP_t **position) {
    TFC_3DFVERTEXDATAHANDLE_GET(handle, vert);
    if(!position) 
        return E_FAIL;
    TFC_VECTOR3_COPYFROM(vert->position, (*position));
    return S_OK;
}

HRESULT tfIoThreeDFVertexData_setPosition(struct tfIoThreeDFVertexDataHandle *handle, tfFloatP_t *position) {
    TFC_3DFVERTEXDATAHANDLE_GET(handle, vert);
    if(!position) 
        return E_FAIL;
    TFC_VECTOR3_COPYTO(position, vert->position);
    return S_OK;
}

HRESULT tfIoThreeDFVertexData_getId(struct tfIoThreeDFVertexDataHandle *handle, int *value) {
    TFC_3DFVERTEXDATAHANDLE_GET(handle, vert);
    TFC_PTRCHECK(value);
    *value = vert->id;
    return S_OK;
}

HRESULT tfIoThreeDFVertexData_setId(struct tfIoThreeDFVertexDataHandle *handle, unsigned int value) {
    TFC_3DFVERTEXDATAHANDLE_GET(handle, vert);
    vert->id = value;
    return S_OK;
}

HRESULT tfIoThreeDFVertexData_getEdges(struct tfIoThreeDFVertexDataHandle *handle, struct tfIoThreeDFEdgeDataHandle **edges, unsigned int *numEdges) {
    TFC_3DFVERTEXDATAHANDLE_GET(handle, vert);
    return TissueForge::capi::copyMeshObjFromVec(vert->edges, edges, numEdges);
}

HRESULT tfIoThreeDFVertexData_setEdges(struct tfIoThreeDFVertexDataHandle *handle, struct tfIoThreeDFEdgeDataHandle *edges, unsigned int numEdges) {
    TFC_3DFVERTEXDATAHANDLE_GET(handle, vert);
    return TissueForge::capi::copyMeshObjToVec(vert->edges, edges, numEdges);
}

HRESULT tfIoThreeDFVertexData_getFaces(struct tfIoThreeDFVertexDataHandle *handle, struct tfIoThreeDFFaceDataHandle **faces, unsigned int *numFaces) {
    TFC_3DFVERTEXDATAHANDLE_GET(handle, vert);
    return TissueForge::capi::copyMeshObjFromVec(vert->getFaces(), faces, numFaces);
}

HRESULT tfIoThreeDFVertexData_getMeshes(struct tfIoThreeDFVertexDataHandle *handle, struct tfIoThreeDFMeshDataHandle **meshes, unsigned int *numMeshes) {
    TFC_3DFVERTEXDATAHANDLE_GET(handle, vert);
    return TissueForge::capi::copyMeshObjFromVec(vert->getMeshes(), meshes, numMeshes);
}

HRESULT tfIoThreeDFVertexData_getNumEdges(struct tfIoThreeDFVertexDataHandle *handle, unsigned int *value) {
    TFC_3DFVERTEXDATAHANDLE_GET(handle, vert);
    TFC_PTRCHECK(value);
    *value = vert->getNumEdges();
    return S_OK;
}

HRESULT tfIoThreeDFVertexData_getNumFaces(struct tfIoThreeDFVertexDataHandle *handle, unsigned int *value) {
    TFC_3DFVERTEXDATAHANDLE_GET(handle, vert);
    TFC_PTRCHECK(value);
    *value = vert->getNumFaces();
    return S_OK;
}

HRESULT tfIoThreeDFVertexData_getNumMeshes(struct tfIoThreeDFVertexDataHandle *handle, unsigned int *value) {
    TFC_3DFVERTEXDATAHANDLE_GET(handle, vert);
    TFC_PTRCHECK(value);
    *value = vert->getNumMeshes();
    return S_OK;
}

HRESULT tfIoThreeDFVertexData_inEdge(struct tfIoThreeDFVertexDataHandle *handle, struct tfIoThreeDFEdgeDataHandle *edge, bool *isIn) {
    TFC_3DFVERTEXDATAHANDLE_GET(handle, vert);
    TFC_3DFEDGEDATAHANDLE_GET(edge, _edge);
    TFC_PTRCHECK(isIn);
    *isIn = vert->in(_edge);
    return S_OK;
}

HRESULT tfIoThreeDFVertexData_inFace(struct tfIoThreeDFVertexDataHandle *handle, struct tfIoThreeDFFaceDataHandle *face, bool *isIn) {
    TFC_3DFVERTEXDATAHANDLE_GET(handle, vert);
    TFC_3DFFACEDATAHANDLE_GET(face, _face);
    TFC_PTRCHECK(isIn);
    *isIn = vert->in(_face);
    return S_OK;
}

HRESULT tfIoThreeDFVertexData_inMesh(struct tfIoThreeDFVertexDataHandle *handle, struct tfIoThreeDFMeshDataHandle *mesh, bool *isIn) {
    TFC_3DFVERTEXDATAHANDLE_GET(handle, vert);
    TFC_3DFMESHDATAHANDLE_GET(mesh, _mesh);
    TFC_PTRCHECK(isIn);
    *isIn = vert->in(_mesh);
    return S_OK;
}

HRESULT tfIoThreeDFVertexData_inStructure(struct tfIoThreeDFVertexDataHandle *handle, struct tfIoThreeDFStructureHandle *structure, bool *isIn) {
    TFC_3DFVERTEXDATAHANDLE_GET(handle, vert);
    TFC_3DFSTRUCTUREHANDLE_GET(structure, _structure);
    TFC_PTRCHECK(isIn);
    *isIn = vert->in(_structure);
    return S_OK;
}


/////////////////////////
// io::ThreeDFEdgeData //
/////////////////////////


HRESULT tfIoThreeDFEdgeData_init(struct tfIoThreeDFEdgeDataHandle *handle, struct tfIoThreeDFVertexDataHandle *va, struct tfIoThreeDFVertexDataHandle *vb) {
    TFC_PTRCHECK(handle);
    TFC_3DFVERTEXDATAHANDLE_GET(va, _va);
    TFC_3DFVERTEXDATAHANDLE_GET(vb, _vb);
    io::ThreeDFEdgeData *edge = new io::ThreeDFEdgeData(_va, _vb);
    handle->tfObj = (void*)edge;
    return S_OK;
}

HRESULT tfIoThreeDFEdgeData_destroy(struct tfIoThreeDFEdgeDataHandle *handle) {
    return TissueForge::capi::destroyHandle<io::ThreeDFEdgeData, tfIoThreeDFEdgeDataHandle>(handle) ? S_OK : E_FAIL;
}

HRESULT tfIoThreeDFEdgeData_hasStructure(struct tfIoThreeDFEdgeDataHandle *handle, bool *hasStructure) {
    return TissueForge::capi::meshObj_hasStructure<io::ThreeDFEdgeData, tfIoThreeDFEdgeDataHandle>(handle, hasStructure);
}

HRESULT tfIoThreeDFEdgeData_getStructure(struct tfIoThreeDFEdgeDataHandle *handle, struct tfIoThreeDFStructureHandle *structure) {
    return TissueForge::capi::meshObj_getStructure<io::ThreeDFEdgeData, tfIoThreeDFEdgeDataHandle>(handle, structure);
}

HRESULT tfIoThreeDFEdgeData_setStructure(struct tfIoThreeDFEdgeDataHandle *handle, struct tfIoThreeDFStructureHandle *structure) {
    return TissueForge::capi::meshObj_setStructure<io::ThreeDFEdgeData, tfIoThreeDFEdgeDataHandle>(handle, structure);
}

HRESULT tfIoThreeDFEdgeData_getId(struct tfIoThreeDFEdgeDataHandle *handle, int *value) {
    TFC_3DFEDGEDATAHANDLE_GET(handle, edge);
    TFC_PTRCHECK(value);
    *value = edge->id;
    return S_OK;
}

HRESULT tfIoThreeDFEdgeData_setId(struct tfIoThreeDFEdgeDataHandle *handle, unsigned int value) {
    TFC_3DFEDGEDATAHANDLE_GET(handle, edge);
    edge->id = value;
    return S_OK;
}

HRESULT tfIoThreeDFEdgeData_getVertices(struct tfIoThreeDFEdgeDataHandle *handle, struct tfIoThreeDFVertexDataHandle *va, struct tfIoThreeDFVertexDataHandle *vb) {
    TFC_3DFEDGEDATAHANDLE_GET(handle, edge);
    TFC_PTRCHECK(va);
    TFC_PTRCHECK(vb);
    va->tfObj = (void*)edge->va;
    vb->tfObj = (void*)edge->vb;
    return S_OK;
}

HRESULT tfIoThreeDFEdgeData_setVertices(struct tfIoThreeDFEdgeDataHandle *handle, struct tfIoThreeDFVertexDataHandle *va, struct tfIoThreeDFVertexDataHandle *vb) {
    TFC_3DFEDGEDATAHANDLE_GET(handle, edge);
    TFC_3DFVERTEXDATAHANDLE_GET(va, _va);
    TFC_3DFVERTEXDATAHANDLE_GET(vb, _vb);
    edge->va = _va;
    edge->vb = _vb;
    return S_OK;
}

HRESULT tfIoThreeDFEdgeData_getFaces(struct tfIoThreeDFEdgeDataHandle *handle, struct tfIoThreeDFFaceDataHandle **faces, unsigned int *numFaces) {
    TFC_3DFEDGEDATAHANDLE_GET(handle, edge);
    return TissueForge::capi::copyMeshObjFromVec(edge->getFaces(), faces, numFaces);
}

HRESULT tfIoThreeDFEdgeData_setFaces(struct tfIoThreeDFEdgeDataHandle *handle, struct tfIoThreeDFFaceDataHandle *faces, unsigned int numFaces) {
    TFC_3DFEDGEDATAHANDLE_GET(handle, edge);
    return TissueForge::capi::copyMeshObjToVec(edge->faces, faces, numFaces);
}

HRESULT tfIoThreeDFEdgeData_getMeshes(struct tfIoThreeDFEdgeDataHandle *handle, struct tfIoThreeDFMeshDataHandle **meshes, unsigned int *numMeshes) {
    TFC_3DFEDGEDATAHANDLE_GET(handle, edge);
    return TissueForge::capi::copyMeshObjFromVec(edge->getMeshes(), meshes, numMeshes);
}

HRESULT tfIoThreeDFEdgeData_getNumFaces(struct tfIoThreeDFEdgeDataHandle *handle, unsigned int *value) {
    TFC_3DFEDGEDATAHANDLE_GET(handle, edge);
    TFC_PTRCHECK(value);
    *value = edge->getNumFaces();
    return S_OK;
}

HRESULT tfIoThreeDFEdgeData_getNumMeshes(struct tfIoThreeDFEdgeDataHandle *handle, unsigned int *value) {
    TFC_3DFEDGEDATAHANDLE_GET(handle, edge);
    TFC_PTRCHECK(value);
    *value = edge->getNumMeshes();
    return S_OK;
}

HRESULT tfIoThreeDFEdgeData_hasVertex(struct tfIoThreeDFEdgeDataHandle *handle, struct tfIoThreeDFVertexDataHandle *vertex, bool *isIn) {
    TFC_3DFEDGEDATAHANDLE_GET(handle, edge);
    TFC_3DFVERTEXDATAHANDLE_GET(vertex, _vertex);
    TFC_PTRCHECK(isIn);
    *isIn = edge->has(_vertex);
    return S_OK;
}

HRESULT tfIoThreeDFEdgeData_inFace(struct tfIoThreeDFEdgeDataHandle *handle, struct tfIoThreeDFFaceDataHandle *face, bool *isIn) {
    TFC_3DFEDGEDATAHANDLE_GET(handle, edge);
    TFC_3DFFACEDATAHANDLE_GET(face, _face);
    TFC_PTRCHECK(isIn);
    *isIn = edge->in(_face);
    return S_OK;
}

HRESULT tfIoThreeDFEdgeData_inMesh(struct tfIoThreeDFEdgeDataHandle *handle, struct tfIoThreeDFMeshDataHandle *mesh, bool *isIn) {
    TFC_3DFEDGEDATAHANDLE_GET(handle, edge);
    TFC_3DFMESHDATAHANDLE_GET(mesh, _mesh);
    TFC_PTRCHECK(isIn);
    *isIn = edge->in(_mesh);
    return S_OK;
}

HRESULT tfIoThreeDFEdgeData_inStructure(struct tfIoThreeDFEdgeDataHandle *handle, struct tfIoThreeDFStructureHandle *structure, bool *isIn) {
    TFC_3DFEDGEDATAHANDLE_GET(handle, edge);
    TFC_3DFSTRUCTUREHANDLE_GET(structure, _structure);
    TFC_PTRCHECK(isIn);
    *isIn = edge->in(_structure);
    return S_OK;
}


/////////////////////////
// io::ThreeDFFaceData //
/////////////////////////


HRESULT tfIoThreeDFFaceData_init(struct tfIoThreeDFFaceDataHandle *handle) {
    io::ThreeDFFaceData *face = new io::ThreeDFFaceData();
    handle->tfObj = (void*)face;
    return S_OK;
}

HRESULT tfIoThreeDFFaceData_destroy(struct tfIoThreeDFFaceDataHandle *handle) {
    return TissueForge::capi::destroyHandle<io::ThreeDFFaceData, tfIoThreeDFFaceDataHandle>(handle) ? S_OK : E_FAIL;
}

HRESULT tfIoThreeDFFaceData_hasStructure(struct tfIoThreeDFFaceDataHandle *handle, bool *hasStructure) {
    return TissueForge::capi::meshObj_hasStructure<io::ThreeDFFaceData, tfIoThreeDFFaceDataHandle>(handle, hasStructure);
}

HRESULT tfIoThreeDFFaceData_getStructure(struct tfIoThreeDFFaceDataHandle *handle, struct tfIoThreeDFStructureHandle *structure) {
    return TissueForge::capi::meshObj_getStructure<io::ThreeDFFaceData, tfIoThreeDFFaceDataHandle>(handle, structure);
}

HRESULT tfIoThreeDFFaceData_setStructure(struct tfIoThreeDFFaceDataHandle *handle, struct tfIoThreeDFStructureHandle *structure) {
    return TissueForge::capi::meshObj_setStructure<io::ThreeDFFaceData, tfIoThreeDFFaceDataHandle>(handle, structure);
}

HRESULT tfIoThreeDFFaceData_getId(struct tfIoThreeDFFaceDataHandle *handle, int *value) {
    TFC_3DFFACEDATAHANDLE_GET(handle, face);
    TFC_PTRCHECK(value);
    *value = face->id;
    return S_OK;
}

HRESULT tfIoThreeDFFaceData_setId(struct tfIoThreeDFFaceDataHandle *handle, unsigned int value) {
    TFC_3DFFACEDATAHANDLE_GET(handle, face);
    face->id = value;
    return S_OK;
}

HRESULT tfIoThreeDFFaceData_getVertices(struct tfIoThreeDFFaceDataHandle *handle, struct tfIoThreeDFVertexDataHandle **vertices, unsigned int *numVertices) {
    TFC_3DFFACEDATAHANDLE_GET(handle, face);
    return TissueForge::capi::copyMeshObjFromVec(face->getVertices(), vertices, numVertices);
}

HRESULT tfIoThreeDFFaceData_getNumVertices(struct tfIoThreeDFFaceDataHandle *handle, unsigned int *value) {
    TFC_3DFFACEDATAHANDLE_GET(handle, face);
    TFC_PTRCHECK(value);
    *value = face->getNumVertices();
    return S_OK;
}

HRESULT tfIoThreeDFFaceData_getEdges(struct tfIoThreeDFFaceDataHandle *handle, struct tfIoThreeDFEdgeDataHandle **edges, unsigned int *numEdges) {
    TFC_3DFFACEDATAHANDLE_GET(handle, face);
    return TissueForge::capi::copyMeshObjFromVec(face->edges, edges, numEdges);
}

HRESULT tfIoThreeDFFaceData_setEdges(struct tfIoThreeDFFaceDataHandle *handle, struct tfIoThreeDFEdgeDataHandle *edges, unsigned int numEdges) {
    TFC_3DFFACEDATAHANDLE_GET(handle, face);
    if(!edges) 
        return E_FAIL;
    return TissueForge::capi::copyMeshObjToVec(face->edges, edges, numEdges);
}

HRESULT tfIoThreeDFFaceData_getNumEdges(struct tfIoThreeDFFaceDataHandle *handle, unsigned int *value) {
    TFC_3DFFACEDATAHANDLE_GET(handle, face);
    TFC_PTRCHECK(value);
    *value = face->getNumEdges();
    return S_OK;
}

HRESULT tfIoThreeDFFaceData_getMeshes(struct tfIoThreeDFFaceDataHandle *handle, struct tfIoThreeDFMeshDataHandle **meshes, unsigned int *numMeshes) {
    TFC_3DFFACEDATAHANDLE_GET(handle, face);
    return TissueForge::capi::copyMeshObjFromVec(face->meshes, meshes, numMeshes);
}

HRESULT tfIoThreeDFFaceData_setMeshes(struct tfIoThreeDFFaceDataHandle *handle, struct tfIoThreeDFMeshDataHandle *meshes, unsigned int numMeshes) {
    TFC_3DFFACEDATAHANDLE_GET(handle, face);
    if(!meshes) 
        return E_FAIL;
    return TissueForge::capi::copyMeshObjToVec(face->meshes, meshes, numMeshes);
}

HRESULT tfIoThreeDFFaceData_getNumMeshes(struct tfIoThreeDFFaceDataHandle *handle, unsigned int *value) {
    TFC_3DFFACEDATAHANDLE_GET(handle, face);
    TFC_PTRCHECK(value);
    *value = face->getNumMeshes();
    return S_OK;
}

HRESULT tfIoThreeDFFaceData_hasVertex(struct tfIoThreeDFFaceDataHandle *handle, struct tfIoThreeDFVertexDataHandle *vertex, bool *isIn) {
    TFC_3DFFACEDATAHANDLE_GET(handle, face);
    TFC_3DFVERTEXDATAHANDLE_GET(vertex, _vertex);
    TFC_PTRCHECK(isIn);
    *isIn = face->has(_vertex);
    return S_OK;
}

HRESULT tfIoThreeDFFaceData_hasEdge(struct tfIoThreeDFFaceDataHandle *handle, struct tfIoThreeDFEdgeDataHandle *edge, bool *isIn) {
    TFC_3DFFACEDATAHANDLE_GET(handle, face);
    TFC_3DFEDGEDATAHANDLE_GET(edge, _edge);
    TFC_PTRCHECK(isIn);
    *isIn = face->has(_edge);
    return S_OK;
}

HRESULT tfIoThreeDFFaceData_inMesh(struct tfIoThreeDFFaceDataHandle *handle, struct tfIoThreeDFMeshDataHandle *mesh, bool *isIn) {
    TFC_3DFFACEDATAHANDLE_GET(handle, face);
    TFC_3DFMESHDATAHANDLE_GET(mesh, _mesh);
    TFC_PTRCHECK(isIn);
    *isIn = face->in(_mesh);
    return S_OK;
}

HRESULT tfIoThreeDFFaceData_inStructure(struct tfIoThreeDFFaceDataHandle *handle, struct tfIoThreeDFStructureHandle *structure, bool *isIn) {
    TFC_3DFFACEDATAHANDLE_GET(handle, face);
    TFC_3DFSTRUCTUREHANDLE_GET(structure, _structure);
    TFC_PTRCHECK(isIn);
    *isIn = face->in(_structure);
    return S_OK;
}

HRESULT tfIoThreeDFFaceData_getNormal(struct tfIoThreeDFFaceDataHandle *handle, tfFloatP_t **normal) {
    TFC_3DFFACEDATAHANDLE_GET(handle, face);
    TFC_PTRCHECK(normal);
    TFC_VECTOR3_COPYFROM(face->normal, (*normal));
    return S_OK;
}

HRESULT tfIoThreeDFFaceData_setNormal(struct tfIoThreeDFFaceDataHandle *handle, tfFloatP_t *normal) {
    TFC_3DFFACEDATAHANDLE_GET(handle, face);
    TFC_PTRCHECK(normal);
    TFC_VECTOR3_COPYTO(normal, face->normal);
    return S_OK;
}


/////////////////////////
// io::ThreeDFMeshData //
/////////////////////////


HRESULT tfIoThreeDFMeshData_init(struct tfIoThreeDFMeshDataHandle *handle) {
    io::ThreeDFMeshData *mesh = new io::ThreeDFMeshData();
    handle->tfObj = (void*)mesh;
    return S_OK;
}

HRESULT tfIoThreeDFMeshData_destroy(struct tfIoThreeDFMeshDataHandle *handle) {
    return TissueForge::capi::destroyHandle<io::ThreeDFMeshData, tfIoThreeDFMeshDataHandle>(handle) ? S_OK : E_FAIL;
}

HRESULT tfIoThreeDFMeshData_hasStructure(struct tfIoThreeDFMeshDataHandle *handle, bool *hasStructure) {
    return TissueForge::capi::meshObj_hasStructure<io::ThreeDFMeshData, tfIoThreeDFMeshDataHandle>(handle, hasStructure);
}

HRESULT tfIoThreeDFMeshData_getStructure(struct tfIoThreeDFMeshDataHandle *handle, struct tfIoThreeDFStructureHandle *structure) {
    return TissueForge::capi::meshObj_getStructure<io::ThreeDFMeshData, tfIoThreeDFMeshDataHandle>(handle, structure);
}

HRESULT tfIoThreeDFMeshData_setStructure(struct tfIoThreeDFMeshDataHandle *handle, struct tfIoThreeDFStructureHandle *structure) {
    return TissueForge::capi::meshObj_setStructure<io::ThreeDFMeshData, tfIoThreeDFMeshDataHandle>(handle, structure);
}

HRESULT tfIoThreeDFMeshData_getId(struct tfIoThreeDFMeshDataHandle *handle, int *value) {
    TFC_3DFMESHDATAHANDLE_GET(handle, mesh);
    TFC_PTRCHECK(value);
    *value = mesh->id;
    return S_OK;
}

HRESULT tfIoThreeDFMeshData_setId(struct tfIoThreeDFMeshDataHandle *handle, unsigned int value) {
    TFC_3DFMESHDATAHANDLE_GET(handle, mesh);
    mesh->id = value;
    return S_OK;
}

HRESULT tfIoThreeDFMeshData_getVertices(struct tfIoThreeDFMeshDataHandle *handle, struct tfIoThreeDFVertexDataHandle **vertices, unsigned int *numVertices) {
    TFC_3DFMESHDATAHANDLE_GET(handle, mesh);
    return TissueForge::capi::copyMeshObjFromVec(mesh->getVertices(), vertices, numVertices);
}

HRESULT tfIoThreeDFMeshData_getNumVertices(struct tfIoThreeDFMeshDataHandle *handle, unsigned int *value) {
    TFC_3DFMESHDATAHANDLE_GET(handle, mesh);
    TFC_PTRCHECK(value);
    *value = mesh->getNumVertices();
    return S_OK;
}

HRESULT tfIoThreeDFMeshData_getEdges(struct tfIoThreeDFMeshDataHandle *handle, struct tfIoThreeDFEdgeDataHandle **edges, unsigned int *numEdges) {
    TFC_3DFMESHDATAHANDLE_GET(handle, mesh);
    return TissueForge::capi::copyMeshObjFromVec(mesh->getEdges(), edges, numEdges);
}

HRESULT tfIoThreeDFMeshData_getNumEdges(struct tfIoThreeDFMeshDataHandle *handle, unsigned int *value) {
    TFC_3DFMESHDATAHANDLE_GET(handle, mesh);
    TFC_PTRCHECK(value);
    *value = mesh->getNumEdges();
    return S_OK;
}

HRESULT tfIoThreeDFMeshData_getFaces(struct tfIoThreeDFMeshDataHandle *handle, struct tfIoThreeDFFaceDataHandle **faces, unsigned int *numFaces) {
    TFC_3DFMESHDATAHANDLE_GET(handle, mesh);
    return TissueForge::capi::copyMeshObjFromVec(mesh->getFaces(), faces, numFaces);
}

HRESULT tfIoThreeDFMeshData_setFaces(struct tfIoThreeDFMeshDataHandle *handle, struct tfIoThreeDFFaceDataHandle *faces, unsigned int numFaces) {
    TFC_3DFMESHDATAHANDLE_GET(handle, mesh);
    if(!faces)
        return E_FAIL;
    return TissueForge::capi::copyMeshObjToVec(mesh->faces, faces, numFaces);
}

HRESULT tfIoThreeDFMeshData_getNumFaces(struct tfIoThreeDFMeshDataHandle *handle, unsigned int *value) {
    TFC_3DFMESHDATAHANDLE_GET(handle, mesh);
    TFC_PTRCHECK(value);
    *value = mesh->getNumFaces();
    return S_OK;
}

HRESULT tfIoThreeDFMeshData_hasVertex(struct tfIoThreeDFMeshDataHandle *handle, struct tfIoThreeDFVertexDataHandle *vertex, bool *isIn) {
    TFC_3DFMESHDATAHANDLE_GET(handle, mesh);
    TFC_3DFVERTEXDATAHANDLE_GET(vertex, _vertex);
    TFC_PTRCHECK(isIn);
    *isIn = mesh->has(_vertex);
    return S_OK;
}

HRESULT tfIoThreeDFMeshData_hasEdge(struct tfIoThreeDFMeshDataHandle *handle, struct tfIoThreeDFEdgeDataHandle *edge, bool *isIn) {
    TFC_3DFMESHDATAHANDLE_GET(handle, mesh);
    TFC_3DFEDGEDATAHANDLE_GET(edge, _edge);
    TFC_PTRCHECK(isIn);
    *isIn = mesh->has(_edge);
    return S_OK;
}

HRESULT tfIoThreeDFMeshData_hasFace(struct tfIoThreeDFMeshDataHandle *handle, struct tfIoThreeDFFaceDataHandle *face, bool *isIn) {
    TFC_3DFMESHDATAHANDLE_GET(handle, mesh);
    TFC_3DFFACEDATAHANDLE_GET(face, _face);
    TFC_PTRCHECK(isIn);
    *isIn = mesh->has(_face);
    return S_OK;
}

HRESULT tfIoThreeDFMeshData_inStructure(struct tfIoThreeDFMeshDataHandle *handle, struct tfIoThreeDFStructureHandle *structure, bool *isIn) {
    TFC_3DFMESHDATAHANDLE_GET(handle, mesh);
    TFC_3DFSTRUCTUREHANDLE_GET(structure, _structure);
    TFC_PTRCHECK(isIn);
    *isIn = mesh->in(_structure);
    return S_OK;
}

HRESULT tfIoThreeDFMeshData_getName(struct tfIoThreeDFMeshDataHandle *handle, char **name, unsigned int *numChars) {
    TFC_3DFMESHDATAHANDLE_GET(handle, mesh);
    TFC_PTRCHECK(name);
    TFC_PTRCHECK(numChars);
    char *_name = new char[mesh->name.size() + 1];
    std::strcpy(_name, mesh->name.c_str());
    *numChars = mesh->name.size() + 1;
    return S_OK;
}

HRESULT tfIoThreeDFMeshData_setName(struct tfIoThreeDFMeshDataHandle *handle, const char *name) {
    TFC_3DFMESHDATAHANDLE_GET(handle, mesh);
    TFC_PTRCHECK(name);
    mesh->name = name;
    return S_OK;
}

HRESULT tfIoThreeDFMeshData_hasRenderData(struct tfIoThreeDFMeshDataHandle *handle, bool *hasData) {
    TFC_3DFMESHDATAHANDLE_GET(handle, mesh);
    TFC_PTRCHECK(hasData);
    *hasData = mesh->renderData != NULL;
    return S_OK;
}

HRESULT tfIoThreeDFMeshData_getRenderData(struct tfIoThreeDFMeshDataHandle *handle, struct tfIoThreeDFRenderDataHandle *renderData) {
    TFC_3DFMESHDATAHANDLE_GET(handle, mesh);
    TFC_PTRCHECK(mesh->renderData);
    TFC_PTRCHECK(renderData);
    renderData->tfObj = (void*)mesh->renderData;
    return S_OK;
}

HRESULT tfIoThreeDFMeshData_setRenderData(struct tfIoThreeDFMeshDataHandle *handle, struct tfIoThreeDFRenderDataHandle *renderData) {
    TFC_3DFMESHDATAHANDLE_GET(handle, mesh);
    TFC_3DFRENDERDATAHANDLE_GET(renderData, _renderData);
    mesh->renderData = _renderData;
    return S_OK;
}

HRESULT tfIoThreeDFMeshData_getCentroid(struct tfIoThreeDFMeshDataHandle *handle, tfFloatP_t **centroid) {
    TFC_3DFMESHDATAHANDLE_GET(handle, mesh);
    if(!centroid) 
        return E_FAIL;
    auto _centroid = mesh->getCentroid();
    TFC_VECTOR3_COPYFROM(_centroid, (*centroid));
    return S_OK;
}

HRESULT tfIoThreeDFMeshData_translate(struct tfIoThreeDFMeshDataHandle *handle, tfFloatP_t *displacement) {
    TFC_3DFMESHDATAHANDLE_GET(handle, mesh);
    if(!displacement) 
        return E_FAIL;
    return mesh->translate(FVector3::from(displacement));
}

HRESULT tfIoThreeDFMeshData_translateTo(struct tfIoThreeDFMeshDataHandle *handle, tfFloatP_t *position) {
    TFC_3DFMESHDATAHANDLE_GET(handle, mesh);
    if(!position) 
        return E_FAIL;
    return mesh->translateTo(FVector3::from(position));
}

HRESULT tfIoThreeDFMeshData_rotateAt(struct tfIoThreeDFMeshDataHandle *handle, tfFloatP_t *rotMat, tfFloatP_t *rotPot) {
    TFC_3DFMESHDATAHANDLE_GET(handle, mesh);
    if(!rotMat || !rotPot) 
        return E_FAIL;
    FMatrix3 _rotMat;
    TFC_MATRIX3_COPYTO(rotMat, _rotMat);
    return mesh->rotateAt(_rotMat, FVector3::from(rotPot));
}

HRESULT tfIoThreeDFMeshData_rotate(struct tfIoThreeDFMeshDataHandle *handle, tfFloatP_t *rotMat) {
    TFC_3DFMESHDATAHANDLE_GET(handle, mesh);
    if(!rotMat) 
        return E_FAIL;
    FMatrix3 _rotMat;
    TFC_MATRIX3_COPYTO(rotMat, _rotMat);
    return mesh->rotate(_rotMat);
}

HRESULT tfIoThreeDFMeshData_scaleFrom(struct tfIoThreeDFMeshDataHandle *handle, tfFloatP_t *scales, tfFloatP_t *scalePt) {
    TFC_3DFMESHDATAHANDLE_GET(handle, mesh);
    if(!scales || !scalePt) 
        return E_FAIL;
    return mesh->scaleFrom(FVector3::from(scales), FVector3::from(scalePt));
}

HRESULT tfIoThreeDFMeshData_scaleFromS(struct tfIoThreeDFMeshDataHandle *handle, tfFloatP_t scale, tfFloatP_t *scalePt) {
    TFC_3DFMESHDATAHANDLE_GET(handle, mesh);
    if(!scalePt) 
        return E_FAIL;
    return mesh->scaleFrom(scale, FVector3::from(scalePt));
}

HRESULT tfIoThreeDFMeshData_scale(struct tfIoThreeDFMeshDataHandle *handle, tfFloatP_t *scales) {
    TFC_3DFMESHDATAHANDLE_GET(handle, mesh);
    if(!scales) 
        return E_FAIL;
    return mesh->scale(FVector3::from(scales));
}

HRESULT tfIoThreeDFMeshData_scaleS(struct tfIoThreeDFMeshDataHandle *handle, tfFloatP_t scale) {
    TFC_3DFMESHDATAHANDLE_GET(handle, mesh);
    return mesh->scale(scale);
}


//////////////////////////
// io::ThreeDFStructure //
//////////////////////////


HRESULT tfIoThreeDFStructure_init(struct tfIoThreeDFStructureHandle *handle) {
    io::ThreeDFStructure *strt = new io::ThreeDFStructure();
    handle->tfObj = (void*)strt;
    return S_OK;
}

HRESULT tfIoThreeDFStructure_destroy(struct tfIoThreeDFStructureHandle *handle) {
    return TissueForge::capi::destroyHandle<io::ThreeDFStructure, tfIoThreeDFStructureHandle>(handle) ? S_OK : E_FAIL;
}

HRESULT tfIoThreeDFStructure_getRadiusDef(struct tfIoThreeDFStructureHandle *handle, tfFloatP_t *vRadiusDef) {
    TFC_3DFSTRUCTUREHANDLE_GET(handle, strt);
    TFC_PTRCHECK(vRadiusDef);
    *vRadiusDef = strt->vRadiusDef;
    return S_OK;
}

HRESULT tfIoThreeDFStructure_setRadiusDef(struct tfIoThreeDFStructureHandle *handle, tfFloatP_t vRadiusDef) {
    TFC_3DFSTRUCTUREHANDLE_GET(handle, strt);
    strt->vRadiusDef = vRadiusDef;
    return S_OK;
}

HRESULT tfIoThreeDFStructure_fromFile(struct tfIoThreeDFStructureHandle *handle, const char *filePath) {
    TFC_3DFSTRUCTUREHANDLE_GET(handle, strt);
    return strt->fromFile(filePath);
}

HRESULT tfIoThreeDFStructure_toFile(struct tfIoThreeDFStructureHandle *handle, const char *format, const char *filePath) {
    TFC_3DFSTRUCTUREHANDLE_GET(handle, strt);
    return strt->toFile(format, filePath);
}

HRESULT tfIoThreeDFStructure_flush(struct tfIoThreeDFStructureHandle *handle) {
    TFC_3DFSTRUCTUREHANDLE_GET(handle, strt);
    return strt->flush();
}

HRESULT tfIoThreeDFStructure_extend(struct tfIoThreeDFStructureHandle *handle, struct tfIoThreeDFStructureHandle *s) {
    TFC_3DFSTRUCTUREHANDLE_GET(handle, strt);
    TFC_3DFSTRUCTUREHANDLE_GET(s, _s);
    return strt->extend(*_s);
}

HRESULT tfIoThreeDFStructure_clear(struct tfIoThreeDFStructureHandle *handle) {
    TFC_3DFSTRUCTUREHANDLE_GET(handle, strt);
    return strt->clear();
}

HRESULT tfIoThreeDFStructure_getVertices(struct tfIoThreeDFStructureHandle *handle, struct tfIoThreeDFVertexDataHandle **vertices, unsigned int *numVertices) {
    TFC_3DFSTRUCTUREHANDLE_GET(handle, strt);
    return TissueForge::capi::copyMeshObjFromVec(strt->getVertices(), vertices, numVertices);
}

HRESULT tfIoThreeDFStructure_getEdges(struct tfIoThreeDFStructureHandle *handle, struct tfIoThreeDFEdgeDataHandle **edges, unsigned int *numEdges) {
    TFC_3DFSTRUCTUREHANDLE_GET(handle, strt);
    return TissueForge::capi::copyMeshObjFromVec(strt->getEdges(), edges, numEdges);
}

HRESULT tfIoThreeDFStructure_getFaces(struct tfIoThreeDFStructureHandle *handle, struct tfIoThreeDFFaceDataHandle **faces, unsigned int *numFaces) {
    TFC_3DFSTRUCTUREHANDLE_GET(handle, strt);
    return TissueForge::capi::copyMeshObjFromVec(strt->getFaces(), faces, numFaces);
}

HRESULT tfIoThreeDFStructure_getMeshes(struct tfIoThreeDFStructureHandle *handle, struct tfIoThreeDFMeshDataHandle **meshes, unsigned int *numMeshes) {
    TFC_3DFSTRUCTUREHANDLE_GET(handle, strt);
    return TissueForge::capi::copyMeshObjFromVec(strt->getMeshes(), meshes, numMeshes);
}

HRESULT tfIoThreeDFStructure_getNumVertices(struct tfIoThreeDFStructureHandle *handle, unsigned int *value) {
    TFC_3DFSTRUCTUREHANDLE_GET(handle, strt);
    TFC_PTRCHECK(value);
    *value = strt->getNumVertices();
    return S_OK;
}

HRESULT tfIoThreeDFStructure_getNumEdges(struct tfIoThreeDFStructureHandle *handle, unsigned int *value) {
    TFC_3DFSTRUCTUREHANDLE_GET(handle, strt);
    TFC_PTRCHECK(value);
    *value = strt->getNumEdges();
    return S_OK;
}

HRESULT tfIoThreeDFStructure_getNumFaces(struct tfIoThreeDFStructureHandle *handle, unsigned int *value) {
    TFC_3DFSTRUCTUREHANDLE_GET(handle, strt);
    TFC_PTRCHECK(value);
    *value = strt->getNumFaces();
    return S_OK;
}

HRESULT tfIoThreeDFStructure_getNumMeshes(struct tfIoThreeDFStructureHandle *handle, unsigned int *value) {
    TFC_3DFSTRUCTUREHANDLE_GET(handle, strt);
    TFC_PTRCHECK(value);
    *value = strt->getNumMeshes();
    return S_OK;
}

HRESULT tfIoThreeDFStructure_hasVertex(struct tfIoThreeDFStructureHandle *handle, struct tfIoThreeDFVertexDataHandle *vertex, bool *isIn) {
    TFC_3DFSTRUCTUREHANDLE_GET(handle, strt);
    TFC_3DFVERTEXDATAHANDLE_GET(vertex, _vertex);
    TFC_PTRCHECK(isIn);
    *isIn = strt->has(_vertex);
    return S_OK;
}

HRESULT tfIoThreeDFStructure_hasEdge(struct tfIoThreeDFStructureHandle *handle, struct tfIoThreeDFEdgeDataHandle *edge, bool *isIn) {
    TFC_3DFSTRUCTUREHANDLE_GET(handle, strt);
    TFC_3DFEDGEDATAHANDLE_GET(edge, _edge);
    TFC_PTRCHECK(isIn);
    *isIn = strt->has(_edge);
    return S_OK;
}

HRESULT tfIoThreeDFStructure_hasFace(struct tfIoThreeDFStructureHandle *handle, struct tfIoThreeDFFaceDataHandle *face, bool *isIn) {
    TFC_3DFSTRUCTUREHANDLE_GET(handle, strt);
    TFC_3DFFACEDATAHANDLE_GET(face, _face);
    TFC_PTRCHECK(isIn);
    *isIn = strt->has(_face);
    return S_OK;
}

HRESULT tfIoThreeDFStructure_hasMesh(struct tfIoThreeDFStructureHandle *handle, struct tfIoThreeDFMeshDataHandle *mesh, bool *isIn) {
    TFC_3DFSTRUCTUREHANDLE_GET(handle, strt);
    TFC_3DFMESHDATAHANDLE_GET(mesh, _mesh);
    TFC_PTRCHECK(isIn);
    *isIn = strt->has(_mesh);
    return S_OK;
}

HRESULT tfIoThreeDFStructure_addVertex(struct tfIoThreeDFStructureHandle *handle, struct tfIoThreeDFVertexDataHandle *vertex) {
    TFC_3DFSTRUCTUREHANDLE_GET(handle, strt);
    TFC_3DFVERTEXDATAHANDLE_GET(vertex, _vertex);
    strt->add(_vertex);
    return S_OK;
}

HRESULT tfIoThreeDFStructure_addEdge(struct tfIoThreeDFStructureHandle *handle, struct tfIoThreeDFEdgeDataHandle *edge) {
    TFC_3DFSTRUCTUREHANDLE_GET(handle, strt);
    TFC_3DFEDGEDATAHANDLE_GET(edge, _edge);
    strt->add(_edge);
    return S_OK;
}

HRESULT tfIoThreeDFStructure_addFace(struct tfIoThreeDFStructureHandle *handle, struct tfIoThreeDFFaceDataHandle *face) {
    TFC_3DFSTRUCTUREHANDLE_GET(handle, strt);
    TFC_3DFFACEDATAHANDLE_GET(face, _face);
    strt->add(_face);
    return S_OK;
}

HRESULT tfIoThreeDFStructure_addMesh(struct tfIoThreeDFStructureHandle *handle, struct tfIoThreeDFMeshDataHandle *mesh) {
    TFC_3DFSTRUCTUREHANDLE_GET(handle, strt);
    TFC_3DFMESHDATAHANDLE_GET(mesh, _mesh);
    strt->add(_mesh);
    return S_OK;
}

HRESULT tfIoThreeDFStructure_removeVertex(struct tfIoThreeDFStructureHandle *handle, struct tfIoThreeDFVertexDataHandle *vertex) {
    TFC_3DFSTRUCTUREHANDLE_GET(handle, strt);
    TFC_3DFVERTEXDATAHANDLE_GET(vertex, _vertex);
    strt->remove(_vertex);
    return S_OK;
}

HRESULT tfIoThreeDFStructure_removeEdge(struct tfIoThreeDFStructureHandle *handle, struct tfIoThreeDFEdgeDataHandle *edge) {
    TFC_3DFSTRUCTUREHANDLE_GET(handle, strt);
    TFC_3DFEDGEDATAHANDLE_GET(edge, _edge);
    strt->remove(_edge);
    return S_OK;
}

HRESULT tfIoThreeDFStructure_removeFace(struct tfIoThreeDFStructureHandle *handle, struct tfIoThreeDFFaceDataHandle *face) {
    TFC_3DFSTRUCTUREHANDLE_GET(handle, strt);
    TFC_3DFFACEDATAHANDLE_GET(face, _face);
    strt->remove(_face);
    return S_OK;
}

HRESULT tfIoThreeDFStructure_removeMesh(struct tfIoThreeDFStructureHandle *handle, struct tfIoThreeDFMeshDataHandle *mesh) {
    TFC_3DFSTRUCTUREHANDLE_GET(handle, strt);
    TFC_3DFMESHDATAHANDLE_GET(mesh, _mesh);
    strt->remove(_mesh);
    return S_OK;
}

HRESULT tfIoThreeDFStructure_getCentroid(struct tfIoThreeDFStructureHandle *handle, tfFloatP_t **centroid) {
    TFC_3DFSTRUCTUREHANDLE_GET(handle, strt);
    if(!centroid) 
        return E_FAIL;
    auto _centroid = strt->getCentroid();
    TFC_VECTOR3_COPYFROM(_centroid, (*centroid));
    return S_OK;
}

HRESULT tfIoThreeDFStructure_translate(struct tfIoThreeDFStructureHandle *handle, tfFloatP_t *displacement) {
    TFC_3DFSTRUCTUREHANDLE_GET(handle, strt);
    if(!displacement) 
        return E_FAIL;
    return strt->translate(FVector3::from(displacement));;
}

HRESULT tfIoThreeDFStructure_translateTo(struct tfIoThreeDFStructureHandle *handle, tfFloatP_t *position) {
    TFC_3DFSTRUCTUREHANDLE_GET(handle, strt);
    if(!position) 
        return E_FAIL;
    return strt->translateTo(FVector3::from(position));
}

HRESULT tfIoThreeDFStructure_rotateAt(struct tfIoThreeDFStructureHandle *handle, tfFloatP_t *rotMat, tfFloatP_t *rotPot) {
    TFC_3DFSTRUCTUREHANDLE_GET(handle, strt);
    if(!rotMat || !rotPot) 
        return E_FAIL;
    FMatrix3 _rotMat;
    TFC_MATRIX3_COPYTO(rotMat, _rotMat);
    return strt->rotateAt(_rotMat, FVector3::from(rotPot));
}

HRESULT tfIoThreeDFStructure_rotate(struct tfIoThreeDFStructureHandle *handle, tfFloatP_t *rotMat) {
    TFC_3DFSTRUCTUREHANDLE_GET(handle, strt);
    if(!rotMat) 
        return E_FAIL;
    FMatrix3 _rotMat;
    TFC_MATRIX3_COPYTO(rotMat, _rotMat);
    return strt->rotate(_rotMat);
}

HRESULT tfIoThreeDFStructure_scaleFrom(struct tfIoThreeDFStructureHandle *handle, tfFloatP_t *scales, tfFloatP_t *scalePt) {
    TFC_3DFSTRUCTUREHANDLE_GET(handle, strt);
    if(!scales || !scalePt) 
        return E_FAIL;
    return strt->scaleFrom(FVector3::from(scales), FVector3::from(scalePt));
}

HRESULT tfIoThreeDFStructure_scaleFromS(struct tfIoThreeDFStructureHandle *handle, tfFloatP_t scale, tfFloatP_t *scalePt) {
    TFC_3DFSTRUCTUREHANDLE_GET(handle, strt);
    if(!scalePt) 
        return E_FAIL;
    return strt->scaleFrom(scale, FVector3::from(scalePt));
}

HRESULT tfIoThreeDFStructure_scale(struct tfIoThreeDFStructureHandle *handle, tfFloatP_t *scales) {
    TFC_3DFSTRUCTUREHANDLE_GET(handle, strt);
    if(!scales) 
        return E_FAIL;
    return strt->scale(FVector3::from(scales));
}

HRESULT tfIoThreeDFStructure_scaleS(struct tfIoThreeDFStructureHandle *handle, tfFloatP_t scale) {
    TFC_3DFSTRUCTUREHANDLE_GET(handle, strt);
    return strt->scale(scale);
}


///////////////////
// io::FIOModule //
///////////////////


HRESULT tfIoFIOModule_init(struct tfIoFIOModuleHandle *handle, const char *moduleName, tfIoFIOModuleToFileFcn toFile, tfIoFIOModuleFromFileFcn fromFile) {
    tfIoFIOModule *cmodule = new tfIoFIOModule();
    cmodule->_moduleName = moduleName;
    cmodule->_toFile = toFile;
    cmodule->_fromFile = fromFile;
    handle->tfObj = (void*)cmodule;
    return S_OK;
}

HRESULT tfIoFIOModule_destroy(struct tfIoFIOModuleHandle *handle) {
    return TissueForge::capi::destroyHandle<tfIoFIOModule, tfIoFIOModuleHandle>(handle) ? S_OK : E_FAIL;
}

HRESULT tfIoFIOModule_registerIOModule(struct tfIoFIOModuleHandle *handle) {
    TFC_PTRCHECK(handle); TFC_PTRCHECK(handle->tfObj);
    tfIoFIOModule *cmodule = (tfIoFIOModule*)handle->tfObj;
    cmodule->registerIOModule();
    return S_OK;
}

HRESULT tfIoFIOModule_load(struct tfIoFIOModuleHandle *handle) {
    TFC_PTRCHECK(handle); TFC_PTRCHECK(handle->tfObj);
    tfIoFIOModule *cmodule = (tfIoFIOModule*)handle->tfObj;
    cmodule->load();
    return S_OK;
}


//////////////////////
// Module functions //
//////////////////////


HRESULT tfIoFIO_getIORootElement(struct tfIoIOElementHandle *handle) {
    io::IOElement *rootElement;
    if(!io::FIO::hasImport()) 
        io::FIO::generateIORootElement();
    io::FIO::getCurrentIORootElement(rootElement);
    if(!rootElement) 
        return E_FAIL;
    handle->tfObj = (void*)rootElement;
    return S_OK;
}

HRESULT tfIoFIO_releaseIORootElement() {
    return io::FIO::releaseIORootElement();
}

HRESULT tfIoFIO_hasImport(bool *value) {
    TFC_PTRCHECK(value);
    *value = io::FIO::hasImport();
    return S_OK;
}

HRESULT tfIo_mapImportParticleId(unsigned int pid, unsigned int *mapId) {
    TFC_PTRCHECK(io::FIO::importSummary);
    TFC_PTRCHECK(mapId);
    
    auto itr = io::FIO::importSummary->particleIdMap.find(pid);
    if(itr == io::FIO::importSummary->particleIdMap.end()) 
        return E_FAIL;
    *mapId = itr->second;
    return S_OK;
}

HRESULT tfIo_mapImportParticleTypeId(unsigned int ptid, unsigned int *mapId) {
    TFC_PTRCHECK(io::FIO::importSummary);
    TFC_PTRCHECK(mapId);
    
    auto itr = io::FIO::importSummary->particleTypeIdMap.find(ptid);
    if(itr == io::FIO::importSummary->particleTypeIdMap.end()) 
        return E_FAIL;
    *mapId = itr->second;
    return S_OK;
}

HRESULT tfIo_fromFile3DF(const char *filePath, struct tfIoThreeDFStructureHandle *strt) {
    TFC_PTRCHECK(filePath);
    io::ThreeDFStructure *_strt = io::fromFile3DF(filePath);
    TFC_PTRCHECK(_strt);
    TFC_PTRCHECK(strt);
    strt->tfObj = (void*)_strt;
    return S_OK;
}

HRESULT tfIo_toFile3DF(const char *format, const char *filePath, unsigned int pRefinements) {
    TFC_PTRCHECK(format);
    TFC_PTRCHECK(filePath);
    return io::toFile3DF(format, filePath, pRefinements);
}

HRESULT tfIo_toFile(const char *saveFilePath) {
    TFC_PTRCHECK(saveFilePath);
    return io::toFile(saveFilePath);
}

HRESULT tfIo_toString(char **str, unsigned int *numChars) {
    return TissueForge::capi::str2Char(io::toString(), str, numChars);
}
