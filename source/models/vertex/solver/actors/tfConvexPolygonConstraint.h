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
 * @file tfConvexPolygonConstraint.h
 * 
 */

#ifndef _MODELS_VERTEX_SOLVER_ACTORS_TFCONVEXPOLYGONCONSTRAINT_H_
#define _MODELS_VERTEX_SOLVER_ACTORS_TFCONVEXPOLYGONCONSTRAINT_H_

#include <models/vertex/solver/tfMeshObj.h>


namespace TissueForge::models::vertex { 


    /**
     * @brief Imposes that surfaces are convex. 
     * 
     * When a vertex violates the convex polygon constraint, 
     * a constraint force acts on it in the direction of the 
     * line that joins its neighbor vertices and proportionally 
     * to a constant. 
     */
    struct ConvexPolygonConstraint : MeshObjActor {

        /** Constraint value */
        FloatP_t lam;

        ConvexPolygonConstraint(const FloatP_t &_lam=0.1) {
            lam = _lam;
        }

        /** Name of the actor */
        virtual std::string name() const override { return "ConvexPolygonConstraint"; }

        /** Unique name of the actor */
        static std::string actorName() { return "ConvexPolygonConstraint"; }

        /**
         * @brief Calculate the energy of a source object acting on a target object
         * 
         * @param source source object
         * @param target target object
         * @param e energy 
         */
        FloatP_t energy(const Surface *source, const Vertex *target) override;

        /**
         * @brief Calculate the force that a source object exerts on a target object
         * 
         * @param source source object
         * @param target target object
         * @param f force
         */
        FVector3 force(const Surface *source, const Vertex *target) override;

        /**
         * @brief Create from a JSON string representation. 
         * 
         * @param str a string, as returned by ``toString``
         */
        static ConvexPolygonConstraint *fromString(const std::string &str);
    };

}

#endif // _MODELS_VERTEX_SOLVER_ACTORS_TFCONVEXPOLYGONCONSTRAINT_H_