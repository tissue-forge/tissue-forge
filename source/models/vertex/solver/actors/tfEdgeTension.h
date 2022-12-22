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

#ifndef _MODELS_VERTEX_SOLVER_ACTORS_TFEDGETENSION_H_
#define _MODELS_VERTEX_SOLVER_ACTORS_TFEDGETENSION_H_

#include <models/vertex/solver/tfMeshObj.h>


namespace TissueForge::models::vertex { 


    typedef FloatP_t (*EdgeTensionEnergyFcn)(const Surface*, const Vertex*, const FloatP_t&, const unsigned int&);
    typedef FVector3 (*EdgeTensionForceFcn)(const Surface*, const Vertex*, const FloatP_t&, const unsigned int&);


    struct EdgeTension : MeshObjActor {

        FloatP_t lam;
        unsigned int order;

        EdgeTension(const FloatP_t &lam=0, const unsigned int &order=1);

        /** Name of the actor */
        virtual std::string name() const override { return "EdgeTension"; }

        /** Unique name of the actor */
        static std::string actorName() { return "EdgeTension"; }

        FloatP_t energy(const Surface *source, const Vertex *target) override;

        FVector3 force(const Surface *source, const Vertex *target) override;

        /**
         * @brief Create from a JSON string representation. 
         * 
         * @param str a string, as returned by ``toString``
         */
        static EdgeTension *fromString(const std::string &str);

    private:

        EdgeTensionEnergyFcn energyFcn;
        EdgeTensionForceFcn forceFcn;
    };

}

#endif // _MODELS_VERTEX_SOLVER_ACTORS_TFEDGETENSION_H_