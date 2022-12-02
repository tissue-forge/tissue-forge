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

#ifndef _MODELS_VERTEX_SOLVER_ACTORS_TFBODYFORCE_H_
#define _MODELS_VERTEX_SOLVER_ACTORS_TFBODYFORCE_H_

#include <models/vertex/solver/tfMeshObj.h>

#include <types/tf_types.h>


namespace TissueForge::models::vertex { 


    struct BodyForce : MeshObjActor {

        FVector3 comps;

        BodyForce(const FVector3 &_force=FVector3(0)) {
            comps = _force;
        }

        /** Name of the actor */
        virtual std::string name() const override { return "BodyForce"; }

        /** Unique name of the actor */
        static std::string actorName() { return "BodyForce"; }

        HRESULT energy(const MeshObj *source, const MeshObj *target, FloatP_t &e) override;

        HRESULT force(const MeshObj *source, const MeshObj *target, FloatP_t *f) override;

        /**
         * @brief Create from a JSON string representation. 
         * 
         * @param str a string, as returned by ``toString``
         */
        static BodyForce *fromString(const std::string &str);

    };

}

#endif // _MODELS_VERTEX_SOLVER_ACTORS_TFBODYFORCE_H_