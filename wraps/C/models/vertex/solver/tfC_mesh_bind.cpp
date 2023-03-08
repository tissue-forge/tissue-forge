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

#include "tfC_mesh_bind.h"

#include "TissueForge_c_private.h"

#include <models/vertex/solver/tf_mesh_bind.h>

#include <models/vertex/solver/tfVertex.h>
#include <models/vertex/solver/tfSurface.h>
#include <models/vertex/solver/tfBody.h>


using namespace TissueForge;
using namespace TissueForge::models::vertex;


namespace TissueForge { 


    static MeshObjActor *castC(struct tfVertexSolverMeshObjActorHandle *handle) {
        return castC<MeshObjActor, tfVertexSolverMeshObjActorHandle>(handle);
    }

    static MeshObjTypePairActor *castC(struct tfVertexSolverMeshObjTypePairActorHandle *handle) {
        return castC<MeshObjTypePairActor, tfVertexSolverMeshObjTypePairActorHandle>(handle);
    }

    static VertexHandle *castC(struct tfVertexSolverVertexHandleHandle *handle) {
        return castC<VertexHandle, tfVertexSolverVertexHandleHandle>(handle);
    }

    static SurfaceHandle *castC(struct tfVertexSolverSurfaceHandleHandle *handle) {
        return castC<SurfaceHandle, tfVertexSolverSurfaceHandleHandle>(handle);
    }

    static SurfaceType *castC(struct tfVertexSolverSurfaceTypeHandle *handle) {
        return castC<SurfaceType, tfVertexSolverSurfaceTypeHandle>(handle);
    }

    static BodyHandle *castC(struct tfVertexSolverBodyHandleHandle *handle) {
        return castC<BodyHandle, tfVertexSolverBodyHandleHandle>(handle);
    }

    static BodyType *castC(struct tfVertexSolverBodyTypeHandle *handle) {
        return castC<BodyType, tfVertexSolverBodyTypeHandle>(handle);
    }

}

#define TFC_MESHBIND_GETACTOR(handle, name) \
    MeshObjActor *name = TissueForge::castC<MeshObjActor, tfVertexSolverMeshObjActorHandle>(handle); \
    TFC_PTRCHECK(name);

#define TFC_MESHBIND_GETTYPEPAIRACTOR(handle, name) \
    MeshObjTypePairActor *name = TissueForge::castC<MeshObjTypePairActor, tfVertexSolverMeshObjTypePairActorHandle>(handle); \
    TFC_PTRCHECK(name);

#define TFC_MESHBIND_GETVERTEX(handle, name) \
    VertexHandle *name = TissueForge::castC<VertexHandle, tfVertexSolverVertexHandleHandle>(handle); \
    TFC_PTRCHECK(name);

#define TFC_MESHBIND_GETSURFACE(handle, name) \
    SurfaceHandle *name = TissueForge::castC<SurfaceHandle, tfVertexSolverSurfaceHandleHandle>(handle); \
    TFC_PTRCHECK(name);

#define TFC_MESHBIND_GETSURFACETYPE(handle, name) \
    SurfaceType *name = TissueForge::castC<SurfaceType, tfVertexSolverSurfaceTypeHandle>(handle); \
    TFC_PTRCHECK(name);

#define TFC_MESHBIND_GETBODYHANDLE(handle, name) \
    BodyHandle *name = TissueForge::castC<BodyHandle, tfVertexSolverBodyHandleHandle>(handle); \
    TFC_PTRCHECK(name);

#define TFC_MESHBIND_GETBODYTYPE(handle, name) \
    BodyType *name = TissueForge::castC<BodyType, tfVertexSolverBodyTypeHandle>(handle); \
    TFC_PTRCHECK(name);


//////////////////////
// Module functions //
//////////////////////


HRESULT tfVertexSolverBindBodyType(
    struct tfVertexSolverMeshObjActorHandle *a, 
    struct tfVertexSolverBodyTypeHandle *b
) {
    TFC_MESHBIND_GETACTOR(a, actor);
    TFC_MESHBIND_GETBODYTYPE(b, btype);
    return bind::body(actor, btype);
}

HRESULT tfVertexSolverBindBody(
    struct tfVertexSolverMeshObjActorHandle *a, 
    struct tfVertexSolverBodyHandleHandle *h
) {
    TFC_MESHBIND_GETACTOR(a, actor);
    TFC_MESHBIND_GETBODYHANDLE(h, body);
    return bind::body(actor, *body);
}

HRESULT tfVertexSolverBindSurfaceType(
    struct tfVertexSolverMeshObjActorHandle *a, 
    struct tfVertexSolverSurfaceTypeHandle *s
) {
    TFC_MESHBIND_GETACTOR(a, actor);
    TFC_MESHBIND_GETSURFACETYPE(s, stype);
    return bind::surface(actor, stype);
}

HRESULT tfVertexSolverBindSurface(
    struct tfVertexSolverMeshObjActorHandle *a, 
    struct tfVertexSolverSurfaceHandleHandle *h
) {
    TFC_MESHBIND_GETACTOR(a, actor);
    TFC_MESHBIND_GETSURFACE(h, surface);
    return bind::surface(actor, *surface);
}

HRESULT tfVertexSolverBindTypesSS(
    struct tfVertexSolverMeshObjTypePairActorHandle *a, 
    struct tfVertexSolverSurfaceTypeHandle *type1, 
    struct tfVertexSolverSurfaceTypeHandle *type2
) {
    TFC_MESHBIND_GETTYPEPAIRACTOR(a, actor);
    TFC_MESHBIND_GETSURFACETYPE(type1, stype1);
    TFC_MESHBIND_GETSURFACETYPE(type2, stype2);
    return bind::types(actor, stype1, stype2);
}

HRESULT tfVertexSolverBindTypesSB(
    struct tfVertexSolverMeshObjTypePairActorHandle *a, 
    struct tfVertexSolverSurfaceTypeHandle *type1, 
    struct tfVertexSolverBodyTypeHandle *type2
) {
    TFC_MESHBIND_GETTYPEPAIRACTOR(a, actor);
    TFC_MESHBIND_GETSURFACETYPE(type1, stype1);
    TFC_MESHBIND_GETBODYTYPE(type2, btype2);
    return bind::types(actor, stype1, btype2);
}

HRESULT tfVertexSolverBindTypesBB(
    struct tfVertexSolverMeshObjTypePairActorHandle *a, 
    struct tfVertexSolverBodyTypeHandle *type1, 
    struct tfVertexSolverBodyTypeHandle *type2
) {
    TFC_MESHBIND_GETTYPEPAIRACTOR(a, actor);
    TFC_MESHBIND_GETBODYTYPE(type1, btype1);
    TFC_MESHBIND_GETBODYTYPE(type2, btype2);
    return bind::types(actor, btype1, btype2);
}
