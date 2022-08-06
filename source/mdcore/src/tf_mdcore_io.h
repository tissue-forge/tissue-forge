/*******************************************************************************
 * This file is part of mdcore.
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

#ifndef _MDCORE_SOURCE_TF_MDCORE_IO_H_
#define _MDCORE_SOURCE_TF_MDCORE_IO_H_

#include <mdcore_config.h>
#include <io/tf_io.h>
#include <tfAngle.h>
#include <tfBond.h>
#include <tfBoundaryConditions.h>
#include <tfDihedral.h>
#include <tfFlux.h>
#include <tfForce.h>
#include <tfParticle.h>
#include <tfPotential.h>
#include <tfDPDPotential.h>


namespace TissueForge::io {


    template <>
    HRESULT toFile(const Angle &dataElement, const MetaData &metaData, IOElement *fileElement);

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, Angle *dataElement);

    template <>
    HRESULT toFile(const Bond &dataElement, const MetaData &metaData, IOElement *fileElement);

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, Bond *dataElement);

    template <>
    HRESULT toFile(const BoundaryCondition &dataElement, const MetaData &metaData, IOElement *fileElement);

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, BoundaryCondition *dataElement);

    template <>
    HRESULT toFile(const BoundaryConditions &dataElement, const MetaData &metaData, IOElement *fileElement);

    // Requires returned value to already be initialized with cells
    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, BoundaryConditions *dataElement);

    // Takes a file element generated from BoundaryConditions
    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, BoundaryConditionsArgsContainer *dataElement);

    template <>
    HRESULT toFile(const Dihedral &dataElement, const MetaData &metaData, IOElement *fileElement);

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, Dihedral *dataElement);

    template <>
    HRESULT toFile(const TypeIdPair &dataElement, const MetaData &metaData, IOElement *fileElement);

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, TypeIdPair *dataElement);

    template <>
    HRESULT toFile(const Flux &dataElement, const MetaData &metaData, IOElement *fileElement);

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, Flux *dataElement);

    template <>
    HRESULT toFile(const Fluxes &dataElement, const MetaData &metaData, IOElement *fileElement);

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, Fluxes *dataElement);

    template <>
    HRESULT toFile(const FORCE_TYPE &dataElement, const MetaData &metaData, IOElement *fileElement);

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, FORCE_TYPE *dataElement);

    template <>
    HRESULT toFile(const CustomForce &dataElement, const MetaData &metaData, IOElement *fileElement);

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, CustomForce *dataElement);

    template <>
    HRESULT toFile(const ForceSum &dataElement, const MetaData &metaData, IOElement *fileElement);

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, ForceSum *dataElement);

    template <>
    HRESULT toFile(const Berendsen &dataElement, const MetaData &metaData, IOElement *fileElement);

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, Berendsen *dataElement);

    template <>
    HRESULT toFile(const Gaussian &dataElement, const MetaData &metaData, IOElement *fileElement);

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, Gaussian *dataElement);

    template <>
    HRESULT toFile(const Friction &dataElement, const MetaData &metaData, IOElement *fileElement);

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, Friction *dataElement);

    template <>
    HRESULT toFile(Force *dataElement, const MetaData &metaData, IOElement *fileElement);

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, Force **dataElement);

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, std::vector<Force*> *dataElement);

    template <>
    HRESULT toFile(const Particle &dataElement, const MetaData &metaData, IOElement *fileElement);

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, Particle *dataElement);

    template <>
    HRESULT toFile(const ParticleType &dataElement, const MetaData &metaData, IOElement *fileElement);

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, ParticleType *dataElement);

    template <>
    HRESULT toFile(const ParticleList &dataElement, const MetaData &metaData, IOElement *fileElement);

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, ParticleList *dataElement);

    template <>
    HRESULT toFile(const ParticleTypeList &dataElement, const MetaData &metaData, IOElement *fileElement);

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, ParticleTypeList *dataElement);

    template <>
    HRESULT toFile(Potential *dataElement, const MetaData &metaData, IOElement *fileElement);

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, Potential **dataElement);

    HRESULT toFile(DPDPotential *dataElement, const MetaData &metaData, IOElement *fileElement);

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, DPDPotential **dataElement);
};

#endif // _MDCORE_SOURCE_TF_MDCORE_IO_H_