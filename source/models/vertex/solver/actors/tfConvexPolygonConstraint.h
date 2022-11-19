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

#ifndef _MODELS_VERTEX_SOLVER_ACTORS_TFCONVEXPOLYGONCONSTRAINT_H_
#define _MODELS_VERTEX_SOLVER_ACTORS_TFCONVEXPOLYGONCONSTRAINT_H_

#include <models/vertex/solver/tfMeshObj.h>


namespace TissueForge::models::vertex { 


    struct ConvexPolygonConstraint : MeshObjActor {

        FloatP_t lam;

        ConvexPolygonConstraint(const FloatP_t &_lam=0.1) {
            lam = _lam;
        }

        /** Name of the actor */
        virtual std::string name() const override { return "ConvexPolygonConstraint"; }

        /** Unique name of the actor */
        static std::string actorName() { return "ConvexPolygonConstraint"; }

        HRESULT energy(const MeshObj *source, const MeshObj *target, FloatP_t &e) override;

        HRESULT force(const MeshObj *source, const MeshObj *target, FloatP_t *f) override;
    };

}

#endif // _MODELS_VERTEX_SOLVER_ACTORS_TFCONVEXPOLYGONCONSTRAINT_H_