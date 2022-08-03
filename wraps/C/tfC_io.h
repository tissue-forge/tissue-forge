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

#ifndef _WRAPS_C_TFC_IO_H_
#define _WRAPS_C_TFC_IO_H_

#include "tf_port_c.h"

typedef HRESULT (*tfIoFIOModuleToFileFcn)(struct tfIoMetaDataHandle, struct tfIoIOElementHandle*);
typedef HRESULT (*tfIoFIOModuleFromFileFcn)(struct tfIoMetaDataHandle, struct tfIoIOElementHandle);

// Handles

struct CAPI_EXPORT tfIoMetaDataHandle {
    unsigned int versionMajor;
    unsigned int versionMinor;
    unsigned int versionPatch;
};

struct CAPI_EXPORT tfIoFIOStorageKeysHandle {
    char *KEY_TYPE;
    char *KEY_VALUE;
    char *KEY_METADATA;
    char *KEY_SIMULATOR;
    char *KEY_UNIVERSE;
    char *KEY_MODULES;
};

/**
 * @brief Handle to a @ref io::IOElement instance
 * 
 */
struct CAPI_EXPORT tfIoIOElementHandle {
    void *tfObj;
};

/**
 * @brief Handle to a @ref io::ThreeDFRenderData instance
 * 
 */
struct CAPI_EXPORT tfIoThreeDFRenderDataHandle {
    void *tfObj;
};

/**
 * @brief Handle to a @ref io::ThreeDFVertexData instance
 * 
 */
struct CAPI_EXPORT tfIoThreeDFVertexDataHandle {
    void *tfObj;
};

/**
 * @brief Handle to a @ref io::ThreeDFEdgeData instance
 * 
 */
struct CAPI_EXPORT tfIoThreeDFEdgeDataHandle {
    void *tfObj;
};

/**
 * @brief Handle to a @ref io::ThreeDFFaceData instance
 * 
 */
struct CAPI_EXPORT tfIoThreeDFFaceDataHandle {
    void *tfObj;
};

/**
 * @brief Handle to a @ref io::ThreeDFMeshData instance
 * 
 */
struct CAPI_EXPORT tfIoThreeDFMeshDataHandle {
    void *tfObj;
};

/**
 * @brief Handle to a @ref io::ThreeDFStructure instance
 * 
 */
struct CAPI_EXPORT tfIoThreeDFStructureHandle {
    void *tfObj;
};

/**
 * @brief Handle to a @ref io::FIOModule instance. 
 * 
 */
struct CAPI_EXPORT tfIoFIOModuleHandle {
    void *tfObj;
};



//////////////////
// io::MetaData //
//////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoMetaData_init(struct tfIoMetaDataHandle *handle);


////////////////////////
// io::FIOStorageKeys //
////////////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoFIOStorageKeys_init(struct tfIoFIOStorageKeysHandle *handle);

/**
 * @brief Destroy an instance
 * 
 * @param handle populated handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoFIOStorageKeys_destroy(struct tfIoFIOStorageKeysHandle *handle);


///////////////////
// io::IOElement //
///////////////////


/**
 * @brief Initialize an empty instance
 * 
 * @param handle handle to populate
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoIOElement_init(struct tfIoIOElementHandle *handle);

/**
 * @brief Destroy an instance
 * 
 * @param handle populated handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoIOElement_destroy(struct tfIoIOElementHandle *handle);

/**
 * @brief Get the instance value type
 * 
 * @param handle populated handle
 * @param type value type
 * @param numChars number of characters
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoIOElement_getType(struct tfIoIOElementHandle *handle, char **type, unsigned int *numChars);

/**
 * @brief Set the instance value type
 * 
 * @param handle populated handle
 * @param type value type
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoIOElement_setType(struct tfIoIOElementHandle *handle, const char *type);

/**
 * @brief Get the instance value
 * 
 * @param handle populated handle
 * @param value value string
 * @param numChars number of characters
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoIOElement_getValue(struct tfIoIOElementHandle *handle, char **value, unsigned int *numChars);

/**
 * @brief Set the instance value
 * 
 * @param handle populated handle
 * @param value value string
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoIOElement_setValue(struct tfIoIOElementHandle *handle, const char *value);

/**
 * @brief Test whether an instance has a parent element
 * 
 * @param handle populated handle
 * @param hasParent true when instance has a parent element
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoIOElement_hasParent(struct tfIoIOElementHandle *handle, bool *hasParent);

/**
 * @brief Get an instance parent
 * 
 * @param handle populated handle
 * @param parent instance parent
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoIOElement_getParent(struct tfIoIOElementHandle *handle, struct tfIoIOElementHandle *parent);

/**
 * @brief Set an instance parent
 * 
 * @param handle populated handle
 * @param parent instance parent
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoIOElement_setParent(struct tfIoIOElementHandle *handle, struct tfIoIOElementHandle *parent);

/**
 * @brief Get the number of child elements
 * 
 * @param handle populated handle
 * @param numChildren number of child elements
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoIOElement_getNumChildren(struct tfIoIOElementHandle *handle, unsigned int *numChildren);

/**
 * @brief Get the child element keys
 * 
 * @param handle populated handle
 * @param keys child element keys
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoIOElement_getKeys(struct tfIoIOElementHandle *handle, char ***keys);

/**
 * @brief Get a child
 * 
 * @param handle populated handle
 * @param key child key
 * @param child child element
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoIOElement_getChild(struct tfIoIOElementHandle *handle, const char *key, struct tfIoIOElementHandle *child);

/**
 * @brief Set a child
 * 
 * @param handle populated handle
 * @param key child key
 * @param child child element
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoIOElement_setChild(struct tfIoIOElementHandle *handle, const char *key, struct tfIoIOElementHandle *child);


///////////////////////////
// io::ThreeDFRenderData //
///////////////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFRenderData_init(struct tfIoThreeDFRenderDataHandle *handle);

/**
 * @brief Destroy an instance
 * 
 * @param handle populated handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFRenderData_destroy(struct tfIoThreeDFRenderDataHandle *handle);

/**
 * @brief Get the color
 * 
 * @param handle populated handle
 * @param color data color
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFRenderData_getColor(struct tfIoThreeDFRenderDataHandle *handle, tfFloatP_t **color);

/**
 * @brief Set the color
 * 
 * @param handle populated handle
 * @param color data color
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFRenderData_setColor(struct tfIoThreeDFRenderDataHandle *handle, tfFloatP_t *color);


///////////////////////////
// io::ThreeDFVertexData //
///////////////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @param position global position
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFVertexData_init(struct tfIoThreeDFVertexDataHandle *handle, tfFloatP_t *position);

/**
 * @brief Destroy an instance
 * 
 * @param handle populated handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFVertexData_destroy(struct tfIoThreeDFVertexDataHandle *handle);

/**
 * @brief Test whether an instance has a parent structure
 * 
 * @param handle populated handle
 * @param hasStructure flag signifying whether an instance has a parent structure
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFVertexData_hasStructure(struct tfIoThreeDFVertexDataHandle *handle, bool *hasStructure);

/**
 * @brief Get the parent structure
 * 
 * @param handle populated handle
 * @param structure parent structure
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFVertexData_getStructure(struct tfIoThreeDFVertexDataHandle *handle, struct tfIoThreeDFStructureHandle *structure);

/**
 * @brief Set the parent structure
 * 
 * @param handle populated handle
 * @param structure parent structure
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFVertexData_setStructure(struct tfIoThreeDFVertexDataHandle *handle, struct tfIoThreeDFStructureHandle *structure);

/**
 * @brief Get the global position
 * 
 * @param handle populated handle
 * @param position global position
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFVertexData_getPosition(struct tfIoThreeDFVertexDataHandle *handle, tfFloatP_t **position);

/**
 * @brief Set the global position
 * 
 * @param handle populated handle
 * @param position global position
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFVertexData_setPosition(struct tfIoThreeDFVertexDataHandle *handle, tfFloatP_t *position);

/**
 * @brief Get the ID, if any. Unique to its structure and type. -1 if not set.
 * 
 * @param handle populated handle
 * @param value ID, if any; otherwise -1
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFVertexData_getId(struct tfIoThreeDFVertexDataHandle *handle, int *value);

/**
 * @brief Set the ID
 * 
 * @param handle populated handle
 * @param value ID
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFVertexData_setId(struct tfIoThreeDFVertexDataHandle *handle, unsigned int value);

/**
 * @brief Get all parent edges
 * 
 * @param handle populated handle
 * @param edges parent edges
 * @param numEdges number of edges
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFVertexData_getEdges(struct tfIoThreeDFVertexDataHandle *handle, struct tfIoThreeDFEdgeDataHandle **edges, unsigned int *numEdges);

/**
 * @brief Set all parent edges
 * 
 * @param handle populated handle
 * @param edges parent edges
 * @param numEdges number of edges
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFVertexData_setEdges(struct tfIoThreeDFVertexDataHandle *handle, struct tfIoThreeDFEdgeDataHandle *edges, unsigned int numEdges);

/**
 * @brief Get all parent faces
 * 
 * @param handle populated handle
 * @param faces parent faces
 * @param numFaces number of faces
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFVertexData_getFaces(struct tfIoThreeDFVertexDataHandle *handle, struct tfIoThreeDFFaceDataHandle **faces, unsigned int *numFaces);

/**
 * @brief Get all parent meshes
 * 
 * @param handle populated handle
 * @param meshes parent meshes
 * @param numMeshes number of meshes
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFVertexData_getMeshes(struct tfIoThreeDFVertexDataHandle *handle, struct tfIoThreeDFMeshDataHandle **meshes, unsigned int *numMeshes);

/**
 * @brief Get the number of parent edges
 * 
 * @param handle populated handle
 * @param value number of parent edges
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFVertexData_getNumEdges(struct tfIoThreeDFVertexDataHandle *handle, unsigned int *value);

/**
 * @brief Get the number of parent faces
 * 
 * @param handle populated handle
 * @param value number of parent faces
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFVertexData_getNumFaces(struct tfIoThreeDFVertexDataHandle *handle, unsigned int *value);

/**
 * @brief Get the number of parent meshes
 * 
 * @param handle populated handle
 * @param value number of parent meshes
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFVertexData_getNumMeshes(struct tfIoThreeDFVertexDataHandle *handle, unsigned int *value);

/**
 * @brief Test whether in an edge
 * 
 * @param handle populated handle
 * @param edge edge to test
 * @param isIn flag signifying whether in an edge
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFVertexData_inEdge(struct tfIoThreeDFVertexDataHandle *handle, struct tfIoThreeDFEdgeDataHandle *edge, bool *isIn);

/**
 * @brief Test whether in a face
 * 
 * @param handle populated handle
 * @param face face to test
 * @param isIn flag signifying whether in a face
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFVertexData_inFace(struct tfIoThreeDFVertexDataHandle *handle, struct tfIoThreeDFFaceDataHandle *face, bool *isIn);

/**
 * @brief Test whether in a mesh
 * 
 * @param handle populated handle
 * @param mesh mesh to test
 * @param isIn flag signifying whether in a mesh
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFVertexData_inMesh(struct tfIoThreeDFVertexDataHandle *handle, struct tfIoThreeDFMeshDataHandle *mesh, bool *isIn);

/**
 * @brief Test whether in a structure
 * 
 * @param handle populated handle
 * @param structure structure to test
 * @param isIn flag signifying whether in a structure
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFVertexData_inStructure(struct tfIoThreeDFVertexDataHandle *handle, struct tfIoThreeDFStructureHandle *structure, bool *isIn);


/////////////////////////
// io::ThreeDFEdgeData //
/////////////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @param va first vertex
 * @param vb second vertex
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFEdgeData_init(struct tfIoThreeDFEdgeDataHandle *handle, struct tfIoThreeDFVertexDataHandle *va, struct tfIoThreeDFVertexDataHandle *vb);

/**
 * @brief Destroy an instance
 * 
 * @param handle populated handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFEdgeData_destroy(struct tfIoThreeDFEdgeDataHandle *handle);

/**
 * @brief Test whether an instance has a parent structure
 * 
 * @param handle populated handle
 * @param hasStructure flag signifying whether an instance has a parent structure
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFEdgeData_hasStructure(struct tfIoThreeDFEdgeDataHandle *handle, bool *hasStructure);

/**
 * @brief Get the parent structure
 * 
 * @param handle populated handle
 * @param structure parent structure
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFEdgeData_getStructure(struct tfIoThreeDFEdgeDataHandle *handle, struct tfIoThreeDFStructureHandle *structure);

/**
 * @brief Set the parent structure
 * 
 * @param handle populated handle
 * @param structure parent structure
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFEdgeData_setStructure(struct tfIoThreeDFEdgeDataHandle *handle, struct tfIoThreeDFStructureHandle *structure);

/**
 * @brief Get the ID, if any. Unique to its structure and type. -1 if not set.
 * 
 * @param handle populated handle
 * @param value ID, if any; otherwise -1
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFEdgeData_getId(struct tfIoThreeDFEdgeDataHandle *handle, int *value);

/**
 * @brief Set the ID
 * 
 * @param handle populated handle
 * @param value ID
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFEdgeData_setId(struct tfIoThreeDFEdgeDataHandle *handle, unsigned int value);

/**
 * @brief Get all child vertices
 * 
 * @param handle populated handle
 * @param va first vertex
 * @param vb second vertex
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFEdgeData_getVertices(struct tfIoThreeDFEdgeDataHandle *handle, struct tfIoThreeDFVertexDataHandle *va, struct tfIoThreeDFVertexDataHandle *vb);

/**
 * @brief Set all child vertices
 * 
 * @param handle populated handle
 * @param va first vertex
 * @param vb second vertex
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFEdgeData_setVertices(struct tfIoThreeDFEdgeDataHandle *handle, struct tfIoThreeDFVertexDataHandle *va, struct tfIoThreeDFVertexDataHandle *vb);

/**
 * @brief Get all parent faces
 * 
 * @param handle populated handle
 * @param faces parent faces
 * @param numFaces number of faces
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFEdgeData_getFaces(struct tfIoThreeDFEdgeDataHandle *handle, struct tfIoThreeDFFaceDataHandle **faces, unsigned int *numFaces);

/**
 * @brief Set all parent faces
 * 
 * @param handle populated handle
 * @param faces parent faces
 * @param numFaces number of faces
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFEdgeData_setFaces(struct tfIoThreeDFEdgeDataHandle *handle, struct tfIoThreeDFFaceDataHandle *faces, unsigned int numFaces);

/**
 * @brief Get all parent meshes
 * 
 * @param handle populated handle
 * @param meshes parent meshes
 * @param numMeshes number of meshes
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFEdgeData_getMeshes(struct tfIoThreeDFEdgeDataHandle *handle, struct tfIoThreeDFMeshDataHandle **meshes, unsigned int *numMeshes);

/**
 * @brief Get the number of parent faces
 * 
 * @param handle populated handle
 * @param value number of parent faces
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFEdgeData_getNumFaces(struct tfIoThreeDFEdgeDataHandle *handle, unsigned int *value);

/**
 * @brief Get the number of parent meshes
 * 
 * @param handle populated handle
 * @param value number of parent meshes
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFEdgeData_getNumMeshes(struct tfIoThreeDFEdgeDataHandle *handle, unsigned int *value);

/**
 * @brief Test whether has a vertex
 * 
 * @param handle populated handle
 * @param vertex vertex to test
 * @param isIn flag signifying whether has a vertex
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFEdgeData_hasVertex(struct tfIoThreeDFEdgeDataHandle *handle, struct tfIoThreeDFVertexDataHandle *vertex, bool *isIn);

/**
 * @brief Test whether in a face
 * 
 * @param handle populated handle
 * @param face face to test
 * @param isIn flag signifying whether in a face
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFEdgeData_inFace(struct tfIoThreeDFEdgeDataHandle *handle, struct tfIoThreeDFFaceDataHandle *face, bool *isIn);

/**
 * @brief Test whether in a mesh
 * 
 * @param handle populated handle
 * @param mesh mesh to test
 * @param isIn flag signifying whether in a mesh
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFEdgeData_inMesh(struct tfIoThreeDFEdgeDataHandle *handle, struct tfIoThreeDFMeshDataHandle *mesh, bool *isIn);

/**
 * @brief Test whether in a structure
 * 
 * @param handle populated handle
 * @param structure structure to test
 * @param isIn flag signifying whether in a structure
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFEdgeData_inStructure(struct tfIoThreeDFEdgeDataHandle *handle, struct tfIoThreeDFStructureHandle *structure, bool *isIn);


/////////////////////////
// io::ThreeDFFaceData //
/////////////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFFaceData_init(struct tfIoThreeDFFaceDataHandle *handle);

/**
 * @brief Destroy an instance
 * 
 * @param handle populated handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFFaceData_destroy(struct tfIoThreeDFFaceDataHandle *handle);

/**
 * @brief Test whether an instance has a parent structure
 * 
 * @param handle populated handle
 * @param hasStructure flag signifying whether an instance has a parent structure
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFFaceData_hasStructure(struct tfIoThreeDFFaceDataHandle *handle, bool *hasStructure);

/**
 * @brief Get the parent structure
 * 
 * @param handle populated handle
 * @param structure parent structure
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFFaceData_getStructure(struct tfIoThreeDFFaceDataHandle *handle, struct tfIoThreeDFStructureHandle *structure);

/**
 * @brief Set the parent structure
 * 
 * @param handle populated handle
 * @param structure parent structure
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFFaceData_setStructure(struct tfIoThreeDFFaceDataHandle *handle, struct tfIoThreeDFStructureHandle *structure);

/**
 * @brief Get the ID, if any. Unique to its structure and type. -1 if not set.
 * 
 * @param handle populated handle
 * @param value ID, if any; otherwise -1
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFFaceData_getId(struct tfIoThreeDFFaceDataHandle *handle, int *value);

/**
 * @brief Set the ID
 * 
 * @param handle populated handle
 * @param value ID
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFFaceData_setId(struct tfIoThreeDFFaceDataHandle *handle, unsigned int value);

/**
 * @brief Get all child vertices
 * 
 * @param handle populated handle
 * @param vertices child vertices
 * @param numVertices number of vertices
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFFaceData_getVertices(struct tfIoThreeDFFaceDataHandle *handle, struct tfIoThreeDFVertexDataHandle **vertices, unsigned int *numVertices);

/**
 * @brief Get the number of child vertices
 * 
 * @param handle populated handle
 * @param value number of child vertices
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFFaceData_getNumVertices(struct tfIoThreeDFFaceDataHandle *handle, unsigned int *value);

/**
 * @brief Get all child edges
 * 
 * @param handle populated handle
 * @param edges child edges
 * @param numEdges number of child edges
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFFaceData_getEdges(struct tfIoThreeDFFaceDataHandle *handle, struct tfIoThreeDFEdgeDataHandle **edges, unsigned int *numEdges);

/**
 * @brief Set all child edges
 * 
 * @param handle populated handle
 * @param edges child edges
 * @param numEdges number of child edges
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFFaceData_setEdges(struct tfIoThreeDFFaceDataHandle *handle, struct tfIoThreeDFEdgeDataHandle *edges, unsigned int numEdges);

/**
 * @brief Get the number of child edges
 * 
 * @param handle populated handle
 * @param value number of child edges
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFFaceData_getNumEdges(struct tfIoThreeDFFaceDataHandle *handle, unsigned int *value);

/**
 * @brief Get all parent meshes
 * 
 * @param handle populated handle
 * @param meshes parent meshes
 * @param numMeshes number of meshes
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFFaceData_getMeshes(struct tfIoThreeDFFaceDataHandle *handle, struct tfIoThreeDFMeshDataHandle **meshes, unsigned int *numMeshes);

/**
 * @brief Set all parent meshes
 * 
 * @param handle populated handle
 * @param meshes parent meshes
 * @param numMeshes number of meshes
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFFaceData_setMeshes(struct tfIoThreeDFFaceDataHandle *handle, struct tfIoThreeDFMeshDataHandle *meshes, unsigned int numMeshes);

/**
 * @brief Get the number of parent meshes
 * 
 * @param handle populated handle
 * @param value number of parent meshes
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFFaceData_getNumMeshes(struct tfIoThreeDFFaceDataHandle *handle, unsigned int *value);

/**
 * @brief Test whether has a vertex
 * 
 * @param handle populated handle
 * @param vertex vertex to test
 * @param isIn flag signifying whether has a vertex
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFFaceData_hasVertex(struct tfIoThreeDFFaceDataHandle *handle, struct tfIoThreeDFVertexDataHandle *vertex, bool *isIn);

/**
 * @brief Test whether has an edge
 * 
 * @param handle populated handle
 * @param edge edge to test
 * @param isIn flag signifying whether has an edge
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFFaceData_hasEdge(struct tfIoThreeDFFaceDataHandle *handle, struct tfIoThreeDFEdgeDataHandle *edge, bool *isIn);

/**
 * @brief Test whether in a mesh
 * 
 * @param handle populated handle
 * @param mesh mesh to test
 * @param isIn flag signifying whether in a mesh
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFFaceData_inMesh(struct tfIoThreeDFFaceDataHandle *handle, struct tfIoThreeDFMeshDataHandle *mesh, bool *isIn);

/**
 * @brief Test whether in a structure
 * 
 * @param handle populated handle
 * @param structure structure to test
 * @param isIn flag signifying whether in a structure
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFFaceData_inStructure(struct tfIoThreeDFFaceDataHandle *handle, struct tfIoThreeDFStructureHandle *structure, bool *isIn);

/**
 * @brief Get the face normal vector
 * 
 * @param handle populated handle
 * @param normal normal vector
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFFaceData_getNormal(struct tfIoThreeDFFaceDataHandle *handle, tfFloatP_t **normal);

/**
 * @brief Set the face normal vector
 * 
 * @param handle populated handle
 * @param normal normal vector
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFFaceData_setNormal(struct tfIoThreeDFFaceDataHandle *handle, tfFloatP_t *normal);


/////////////////////////
// io::ThreeDFMeshData //
/////////////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFMeshData_init(struct tfIoThreeDFMeshDataHandle *handle);

/**
 * @brief Destroy an instance
 * 
 * @param handle populated handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFMeshData_destroy(struct tfIoThreeDFMeshDataHandle *handle);

/**
 * @brief Test whether an instance has a parent structure
 * 
 * @param handle populated handle
 * @param hasStructure flag signifying whether an instance has a parent structure
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFMeshData_hasStructure(struct tfIoThreeDFMeshDataHandle *handle, bool *hasStructure);

/**
 * @brief Get the parent structure
 * 
 * @param handle populated handle
 * @param structure parent structure
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFMeshData_getStructure(struct tfIoThreeDFMeshDataHandle *handle, struct tfIoThreeDFStructureHandle *structure);

/**
 * @brief Set the parent structure
 * 
 * @param handle populated handle
 * @param structure parent structure
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFMeshData_setStructure(struct tfIoThreeDFMeshDataHandle *handle, struct tfIoThreeDFStructureHandle *structure);

/**
 * @brief Get the ID, if any. Unique to its structure and type. -1 if not set.
 * 
 * @param handle populated handle
 * @param value ID, if any; otherwise -1
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFMeshData_getId(struct tfIoThreeDFMeshDataHandle *handle, int *value);

/**
 * @brief Set the ID
 * 
 * @param handle populated handle
 * @param value ID
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFMeshData_setId(struct tfIoThreeDFMeshDataHandle *handle, unsigned int value);

/**
 * @brief Get all child vertices
 * 
 * @param handle populated handle
 * @param vertices child vertices
 * @param numVertices number of vertices
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFMeshData_getVertices(struct tfIoThreeDFMeshDataHandle *handle, struct tfIoThreeDFVertexDataHandle **vertices, unsigned int *numVertices);

/**
 * @brief Get the number of child vertices
 * 
 * @param handle populated handle
 * @param value number of child vertices
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFMeshData_getNumVertices(struct tfIoThreeDFMeshDataHandle *handle, unsigned int *value);

/**
 * @brief Get all child edges
 * 
 * @param handle populated handle
 * @param edges child edges
 * @param numEdges number of child edges
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFMeshData_getEdges(struct tfIoThreeDFMeshDataHandle *handle, struct tfIoThreeDFEdgeDataHandle **edges, unsigned int *numEdges);

/**
 * @brief Get the number of child edges
 * 
 * @param handle populated handle
 * @param value number of child edges
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFMeshData_getNumEdges(struct tfIoThreeDFMeshDataHandle *handle, unsigned int *value);

/**
 * @brief Get all child faces
 * 
 * @param handle populated handle
 * @param faces child faces
 * @param numFaces number of child faces
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFMeshData_getFaces(struct tfIoThreeDFMeshDataHandle *handle, struct tfIoThreeDFFaceDataHandle **faces, unsigned int *numFaces);

/**
 * @brief Get all child faces
 * 
 * @param handle populated handle
 * @param faces child faces
 * @param numFaces number of child faces
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFMeshData_setFaces(struct tfIoThreeDFMeshDataHandle *handle, struct tfIoThreeDFFaceDataHandle *faces, unsigned int numFaces);

/**
 * @brief Get the number of child faces
 * 
 * @param handle populated handle
 * @param value number of child faces
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFMeshData_getNumFaces(struct tfIoThreeDFMeshDataHandle *handle, unsigned int *value);

/**
 * @brief Test whether has a vertex
 * 
 * @param handle populated handle
 * @param vertex vertex to test
 * @param isIn flag signifying whether has a vertex
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFMeshData_hasVertex(struct tfIoThreeDFMeshDataHandle *handle, struct tfIoThreeDFVertexDataHandle *vertex, bool *isIn);

/**
 * @brief Test whether has an edge
 * 
 * @param handle populated handle
 * @param edge edge to test
 * @param isIn flag signifying whether has an edge
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFMeshData_hasEdge(struct tfIoThreeDFMeshDataHandle *handle, struct tfIoThreeDFEdgeDataHandle *edge, bool *isIn);

/**
 * @brief Test whether has a face
 * 
 * @param handle populated handle
 * @param face face to test
 * @param isIn flag signifying whether has a face
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFMeshData_hasFace(struct tfIoThreeDFMeshDataHandle *handle, struct tfIoThreeDFFaceDataHandle *face, bool *isIn);

/**
 * @brief Test whether in a structure
 * 
 * @param handle populated handle
 * @param structure structure to test
 * @param isIn flag signifying whether in a structure
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFMeshData_inStructure(struct tfIoThreeDFMeshDataHandle *handle, struct tfIoThreeDFStructureHandle *structure, bool *isIn);

/**
 * @brief Get the mesh name
 * 
 * @param handle populated handle
 * @param name mesh name
 * @param numChars number of characters
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFMeshData_getName(struct tfIoThreeDFMeshDataHandle *handle, char **name, unsigned int *numChars);

/**
 * @brief Set the mesh name
 * 
 * @param handle populated handle
 * @param name mesh name
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFMeshData_setName(struct tfIoThreeDFMeshDataHandle *handle, const char *name);

/**
 * @brief Test whether has render data
 * 
 * @param handle populated handle
 * @param hasData flag signifying whether has render data
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFMeshData_hasRenderData(struct tfIoThreeDFMeshDataHandle *handle, bool *hasData);

/**
 * @brief Get render data
 * 
 * @param handle populated handle
 * @param renderData render data
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFMeshData_getRenderData(struct tfIoThreeDFMeshDataHandle *handle, struct tfIoThreeDFRenderDataHandle *renderData);

/**
 * @brief Set render data
 * 
 * @param handle populated handle
 * @param renderData render data
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFMeshData_setRenderData(struct tfIoThreeDFMeshDataHandle *handle, struct tfIoThreeDFRenderDataHandle *renderData);

/**
 * @brief Get the centroid of the mesh
 * 
 * @param handle populated handle
 * @param centroid mesh centroid
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFMeshData_getCentroid(struct tfIoThreeDFMeshDataHandle *handle, tfFloatP_t **centroid);

/**
 * @brief Translate the mesh by a displacement
 * 
 * @param handle populated handle
 * @param displacement translation displacement
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFMeshData_translate(struct tfIoThreeDFMeshDataHandle *handle, tfFloatP_t *displacement);

/**
 * @brief Translate the mesh to a position
 * 
 * @param handle populated handle
 * @param displacement translation displacement
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFMeshData_translateTo(struct tfIoThreeDFMeshDataHandle *handle, tfFloatP_t *position);

/**
 * @brief Rotate the mesh about a point
 * 
 * @param handle populated handle
 * @param rotMat rotation matrix
 * @param rotPt rotation point
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFMeshData_rotateAt(struct tfIoThreeDFMeshDataHandle *handle, tfFloatP_t *rotMat, tfFloatP_t *rotPot);

/**
 * @brief Rotate the mesh about its centroid
 * 
 * @param handle populated handle
 * @param rotMat rotation matrix
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFMeshData_rotate(struct tfIoThreeDFMeshDataHandle *handle, tfFloatP_t *rotMat);

/**
 * @brief Scale the mesh about a point
 * 
 * @param handle populated handle
 * @param scales scale coefficients
 * @param scalePt scale point
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFMeshData_scaleFrom(struct tfIoThreeDFMeshDataHandle *handle, tfFloatP_t *scales, tfFloatP_t *scalePt);

/**
 * @brief Scale the mesh uniformly about a point
 * 
 * @param handle populated handle
 * @param scale scale coefficient
 * @param scalePt scale point
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFMeshData_scaleFromS(struct tfIoThreeDFMeshDataHandle *handle, tfFloatP_t scale, tfFloatP_t *scalePt);

/**
 * @brief Scale the structure about its centroid
 * 
 * @param handle populated handle
 * @param scales scale components
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFMeshData_scale(struct tfIoThreeDFMeshDataHandle *handle, tfFloatP_t *scales);

/**
 * @brief Scale the structure uniformly about its centroid
 * 
 * @param handle populated handle
 * @param scale scale coefficient
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFMeshData_scaleS(struct tfIoThreeDFMeshDataHandle *handle, tfFloatP_t scale);


//////////////////////////
// io::ThreeDFStructure //
//////////////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFStructure_init(struct tfIoThreeDFStructureHandle *handle);

/**
 * @brief Destroy an instance
 * 
 * @param handle populated handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFStructure_destroy(struct tfIoThreeDFStructureHandle *handle);

/**
 * @brief Get the default radius applied to vertices when generating meshes from point clouds
 * 
 * @param handle populated handle
 * @param vRadiusDef default radius
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFStructure_getRadiusDef(struct tfIoThreeDFStructureHandle *handle, tfFloatP_t *vRadiusDef);

/**
 * @brief Set the default radius applied to vertices when generating meshes from point clouds
 * 
 * @param handle populated handle
 * @param vRadiusDef default radius
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFStructure_setRadiusDef(struct tfIoThreeDFStructureHandle *handle, tfFloatP_t vRadiusDef);

/**
 * @brief Load from file
 * 
 * @param handle populated handle
 * @param filePath file absolute path
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFStructure_fromFile(struct tfIoThreeDFStructureHandle *handle, const char *filePath);

/**
 * @brief Write to file
 * 
 * @param handle populated handle
 * @param format output format of file
 * @param filePath file absolute path
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFStructure_toFile(struct tfIoThreeDFStructureHandle *handle, const char *format, const char *filePath);

/**
 * @brief Flush stucture. All scheduled processes are executed. 
 * 
 * @param handle populated handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFStructure_flush(struct tfIoThreeDFStructureHandle *handle);

/**
 * @brief Extend a structure
 * 
 * @param handle populated handle
 * @param s stucture to extend with
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFStructure_extend(struct tfIoThreeDFStructureHandle *handle, struct tfIoThreeDFStructureHandle *s);

/**
 * @brief Clear all data of the structure
 * 
 * @param handle populated handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFStructure_clear(struct tfIoThreeDFStructureHandle *handle);

/**
 * @brief Get all constituent vertices
 * 
 * @param handle populated handle
 * @param vertices child vertices
 * @param numVertices number of vertices
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFStructure_getVertices(struct tfIoThreeDFStructureHandle *handle, struct tfIoThreeDFVertexDataHandle **vertices, unsigned int *numVertices);

/**
 * @brief Get all constituent edges
 * 
 * @param handle populated handle
 * @param edges child edges
 * @param numEdges number of child edges
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFStructure_getEdges(struct tfIoThreeDFStructureHandle *handle, struct tfIoThreeDFEdgeDataHandle **edges, unsigned int *numEdges);

/**
 * @brief Get all constituent faces
 * 
 * @param handle populated handle
 * @param faces child faces
 * @param numFaces number of child faces
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFStructure_getFaces(struct tfIoThreeDFStructureHandle *handle, struct tfIoThreeDFFaceDataHandle **faces, unsigned int *numFaces);

/**
 * @brief Get all constituent meshes
 * 
 * @param handle populated handle
 * @param meshes child meshes
 * @param numMeshes number of meshes
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFStructure_getMeshes(struct tfIoThreeDFStructureHandle *handle, struct tfIoThreeDFMeshDataHandle **meshes, unsigned int *numMeshes);

/**
 * @brief Get the number of constituent vertices
 * 
 * @param handle populated handle
 * @param value number of constituent vertices
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFStructure_getNumVertices(struct tfIoThreeDFStructureHandle *handle, unsigned int *value);

/**
 * @brief Get the number of constituent edges
 * 
 * @param handle populated handle
 * @param value number of constituent edges
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFStructure_getNumEdges(struct tfIoThreeDFStructureHandle *handle, unsigned int *value);

/**
 * @brief Get the number of constituent faces
 * 
 * @param handle populated handle
 * @param value number of constituent faces
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFStructure_getNumFaces(struct tfIoThreeDFStructureHandle *handle, unsigned int *value);

/**
 * @brief Get the number of constituent meshes
 * 
 * @param handle populated handle
 * @param value number of constituent meshes
 */
CAPI_FUNC(HRESULT) tfIoThreeDFStructure_getNumMeshes(struct tfIoThreeDFStructureHandle *handle, unsigned int *value);

/**
 * @brief Test whether a vertex is a constituent
 * 
 * @param handle populated handle
 * @param vertex vertex to test
 * @param isIn flag signifying whether has a vertex
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFStructure_hasVertex(struct tfIoThreeDFStructureHandle *handle, struct tfIoThreeDFVertexDataHandle *vertex, bool *isIn);

/**
 * @brief Test whether an edge is a constituent
 * 
 * @param handle populated handle
 * @param edge edge to test
 * @param isIn flag signifying whether has a vertex
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFStructure_hasEdge(struct tfIoThreeDFStructureHandle *handle, struct tfIoThreeDFEdgeDataHandle *edge, bool *isIn);

/**
 * @brief Test whether a face is a constituent
 * 
 * @param handle populated handle
 * @param face face to test
 * @param isIn flag signifying whether has a vertex
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFStructure_hasFace(struct tfIoThreeDFStructureHandle *handle, struct tfIoThreeDFFaceDataHandle *face, bool *isIn);

/**
 * @brief Test whether a mesh is a constituent
 * 
 * @param handle populated handle
 * @param mesh mesh to test
 * @param isIn flag signifying whether has a vertex
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFStructure_hasMesh(struct tfIoThreeDFStructureHandle *handle, struct tfIoThreeDFMeshDataHandle *mesh, bool *isIn);

/**
 * @brief Add a vertex
 * 
 * @param handle populated handle
 * @param vertex vertex to add
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFStructure_addVertex(struct tfIoThreeDFStructureHandle *handle, struct tfIoThreeDFVertexDataHandle *vertex);

/**
 * @brief Add an edge and all constituent data
 * 
 * @param handle populated handle
 * @param edge edge to add
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFStructure_addEdge(struct tfIoThreeDFStructureHandle *handle, struct tfIoThreeDFEdgeDataHandle *edge);

/**
 * @brief Add a face and all constituent data
 * 
 * @param handle populated handle
 * @param face face to add
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFStructure_addFace(struct tfIoThreeDFStructureHandle *handle, struct tfIoThreeDFFaceDataHandle *face);

/**
 * @brief Add a mesh and all constituent data
 * 
 * @param handle populated handle
 * @param mesh mesh to add
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFStructure_addMesh(struct tfIoThreeDFStructureHandle *handle, struct tfIoThreeDFMeshDataHandle *mesh);

/**
 * @brief Remove a vertex
 * 
 * @param handle populated handle
 * @param vertex vertex to remove
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFStructure_removeVertex(struct tfIoThreeDFStructureHandle *handle, struct tfIoThreeDFVertexDataHandle *vertex);

/**
 * @brief Remove a edge and all constituent data
 * 
 * @param handle populated handle
 * @param edge edge to remove
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFStructure_removeEdge(struct tfIoThreeDFStructureHandle *handle, struct tfIoThreeDFEdgeDataHandle *edge);

/**
 * @brief Remove a face and all constituent data
 * 
 * @param handle populated handle
 * @param face face to remove
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFStructure_removeFace(struct tfIoThreeDFStructureHandle *handle, struct tfIoThreeDFFaceDataHandle *face);

/**
 * @brief Remove a mesh and all constituent data
 * 
 * @param handle populated handle
 * @param mesh mesh to remove
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFStructure_removeMesh(struct tfIoThreeDFStructureHandle *handle, struct tfIoThreeDFMeshDataHandle *mesh);

/**
 * @brief Get the centroid of the structure
 * 
 * @param handle populated handle
 * @param centroid structure centroid
 * @return S_OK on success  
 */
CAPI_FUNC(HRESULT) tfIoThreeDFStructure_getCentroid(struct tfIoThreeDFStructureHandle *handle, tfFloatP_t **centroid);

/**
 * @brief Translate the structure by a displacement
 * 
 * @param handle populated handle
 * @param displacement translation displacement
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFStructure_translate(struct tfIoThreeDFStructureHandle *handle, tfFloatP_t *displacement);

/**
 * @brief Translate the structure to a position
 * 
 * @param handle populated handle
 * @param position translation displacement
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFStructure_translateTo(struct tfIoThreeDFStructureHandle *handle, tfFloatP_t *position);

/**
 * @brief Rotate the structure about a point
 * 
 * @param handle populated handle
 * @param rotMat rotation matrix
 * @param rotPt rotation point
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFStructure_rotateAt(struct tfIoThreeDFStructureHandle *handle, tfFloatP_t *rotMat, tfFloatP_t *rotPot);

/**
 * @brief Rotate the structure about its centroid
 * 
 * @param handle populated handle
 * @param rotMat rotation matrix
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFStructure_rotate(struct tfIoThreeDFStructureHandle *handle, tfFloatP_t *rotMat);

/**
 * @brief Scale the structure about a point
 * 
 * @param handle populated handle
 * @param scales scale coefficients
 * @param scalePt scale point
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFStructure_scaleFrom(struct tfIoThreeDFStructureHandle *handle, tfFloatP_t *scales, tfFloatP_t *scalePt);

/**
 * @brief Scale the structure uniformly about a point
 * 
 * @param handle populated handle
 * @param scale scale coefficient
 * @param scalePt scale point
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFStructure_scaleFromS(struct tfIoThreeDFStructureHandle *handle, tfFloatP_t scale, tfFloatP_t *scalePt);

/**
 * @brief Scale the structure about its centroid
 * 
 * @param handle populated handle
 * @param scales scale coefficients
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFStructure_scale(struct tfIoThreeDFStructureHandle *handle, tfFloatP_t *scales);

/**
 * @brief Scale the structure uniformly about its centroid
 * 
 * @param handle populated handle
 * @param scale scale coefficient
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoThreeDFStructure_scaleS(struct tfIoThreeDFStructureHandle *handle, tfFloatP_t scale);


///////////////////
// io::FIOModule //
///////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @param moduleName name of module
 * @param toFile callback to export module data
 * @param fromFile callback to import module data
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoFIOModule_init(struct tfIoFIOModuleHandle *handle, const char *moduleName, tfIoFIOModuleToFileFcn toFile, tfIoFIOModuleFromFileFcn fromFile);

/**
 * @brief Destroy an instance
 * 
 * @param handle 
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoFIOModule_destroy(struct tfIoFIOModuleHandle *handle);

/**
 * @brief Register a module for I/O events
 * 
 */
CAPI_FUNC(HRESULT) tfIoFIOModule_registerIOModule(struct tfIoFIOModuleHandle *handle);

/**
 * @brief User-facing function to load module data from main import. 
 * 
 * Must only be called after main import. 
 * 
 */
CAPI_FUNC(HRESULT) tfIoFIOModule_load(struct tfIoFIOModuleHandle *handle);


//////////////////////
// Module functions //
//////////////////////


/**
 * @brief Get or generate root element from current simulation state
 * 
 * @param handle root element
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoFIO_getIORootElement(struct tfIoIOElementHandle *handle);

/**
 * @brief Release current root element
 * 
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIoFIO_releaseIORootElement();

/**
 * @brief Test whether imported data is available. 
 * 
 * @param value true when imported data is available
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfIoFIO_hasImport(bool *value);

/**
 * @brief Map a particle id from currently imported file data to the created particle on import
 * 
 * @param pid particle id according to import file
 * @param mapId particle id according to simulation state
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIo_mapImportParticleId(unsigned int pid, unsigned int *mapId);

/**
 * @brief Map a particle type id from currently imported file data to the created particle type on import
 * 
 * @param ptid particle type id according to import file
 * @param mapId particle type id according to simulation state
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIo_mapImportParticleTypeId(unsigned int ptid, unsigned int *mapId);

/**
 * @brief Load a 3D format file
 * 
 * @param filePath path of file
 * @param strt 3D format data container
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIo_fromFile3DF(const char *filePath, struct tfIoThreeDFStructureHandle *strt);

/**
 * @brief Export engine state to a 3D format file
 * 
 * @param format format of file
 * @param filePath path of file
 * @param pRefinements mesh refinements applied when generating meshes
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIo_toFile3DF(const char *format, const char *filePath, unsigned int pRefinements);

/**
 * @brief Save a simulation to file
 * 
 * @param saveFilePath absolute path to file
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIo_toFile(const char *saveFilePath);

/**
 * @brief Return a simulation state as a JSON string
 * 
 * @param str string representation
 * @param numChars number of characters of string representation
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfIo_toString(char **str, unsigned int *numChars);

#endif // _WRAPS_C_TFC_IO_H_