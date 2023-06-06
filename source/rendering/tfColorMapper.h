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

/**
 * @file tfColorMapper.h
 * 
 */

#ifndef _SOURCE_RENDERING_TFCOLORMAPPER_H_
#define _SOURCE_RENDERING_TFCOLORMAPPER_H_

#include <io/tf_io.h>
#include <TissueForge_private.h>

#include "tfColorMaps.h"

#include <vector>


namespace TissueForge {


    struct ParticleType;

    struct Angle;
    struct Bond;
    struct Dihedral;
    struct Particle;


    namespace rendering {


        typedef float (*ParticleColorMapperFunc)(TissueForge::Particle* o, struct ColorMapper* mapper);
        typedef float (*AngleColorMapperFunc)(TissueForge::Angle* o, struct ColorMapper* mapper);
        typedef float (*BondColorMapperFunc)(TissueForge::Bond* o, struct ColorMapper* mapper);
        typedef float (*DihedralColorMapperFunc)(TissueForge::Dihedral* o, struct ColorMapper* mapper);


        /**
         * @brief The color mapping type
         */
        struct CAPI_EXPORT ColorMapper
        {
            ColorMapperFunc map;
            int species_index;
            
            /**
             * @brief minimum value of map
             */
            float min_val;

            /**
             * @brief maximum value of map
             */
            float max_val;

            ColorMapper(const std::string &name, const float &min=0.f, const float &max=1.f);
            ColorMapper();
            ~ColorMapper() {};

            /**
             * @brief Map a scalar value to a color
            */
            fVector4 mapScalar(const float& val) const;

            /**
             * @brief Map a particle to a color
            */
            fVector4 mapObj(Particle* o);

            /**
             * @brief Map an angle to a color
            */
            fVector4 mapObj(Angle* o);

            /**
             * @brief Map a bond to a color
            */
            fVector4 mapObj(Bond* o);

            /**
             * @brief Map a dihedral to a color
            */
            fVector4 mapObj(Dihedral* o);

            /**
             * @brief Test whether the mapper has a particle map
            */
            const bool hasMapParticle() const;

            /**
             * @brief Test whether the mapper has an angle map
            */
            const bool hasMapAngle() const;

            /**
             * @brief Test whether the mapper has a bond map
            */
            const bool hasMapBond() const;

            /**
             * @brief Test whether the mapper has a dihedral map
            */
            const bool hasMapDihedral() const;

            /**
             * @brief Clear the particle map
            */
            void clearMapParticle();

            /**
             * @brief Clear the angle map
            */
            void clearMapAngle();

            /**
             * @brief Clear the bond map
            */
            void clearMapBond();

            /**
             * @brief Clear the dihedral map
            */
            void clearMapDihedral();

            /**
             * @brief Get the particle map label
            */
            unsigned int getMapParticle() const { return map_enum_particle; }

            /**
             * @brief Get the angle map label
            */
            unsigned int getMapAngle() const { return map_enum_angle; }

            /**
             * @brief Get the bond map label
            */
            unsigned int getMapBond() const { return map_enum_bond; }

            /**
             * @brief Get the dihedral map label
            */
            unsigned int getMapDihedral() const { return map_enum_dihedral; }

            /**
             * @brief Set the particle map by a label
            */
            void setMapParticle(const unsigned int& label);

            /**
             * @brief Set the angle map by a label
            */
            void setMapAngle(const unsigned int& label);

            /**
             * @brief Set the bond map by a label
            */
            void setMapBond(const unsigned int& label);

            /**
             * @brief Set the dihedral map by a label
            */
            void setMapDihedral(const unsigned int& label);

            /**
             * @brief Set the particle map to x-coordinate of particle position
            */
            void setMapParticlePositionX();

            /**
             * @brief Set the particle map to y-coordinate of particle position
            */
            void setMapParticlePositionY();

            /**
             * @brief Set the particle map to z-coordinate of particle position
            */
            void setMapParticlePositionZ();

            /**
             * @brief Set the particle map to x-component of particle velocity
            */
            void setMapParticleVelocityX();

            /**
             * @brief Set the particle map to y-component of particle velocity
            */
            void setMapParticleVelocityY();

            /**
             * @brief Set the particle map to z-component of particle velocity
            */
            void setMapParticleVelocityZ();

            /**
             * @brief Set the particle map to particle speed
            */
            void setMapParticleSpeed();

            /**
             * @brief Set the particle map to x-component of particle force
            */
            void setMapParticleForceX();

            /**
             * @brief Set the particle map to y-component of particle force
            */
            void setMapParticleForceY();

            /**
             * @brief Set the particle map to z-component of particle force
            */
            void setMapParticleForceZ();

            /**
             * @brief Set the particle map to a species value
             * 
             * @param pType particle type
             * @param name species name
            */
            void setMapParticleSpecies(ParticleType* pType, const std::string& name);

            /**
             * @brief Set the angle map to angle
            */
            void setMapAngleAngle();

            /**
             * @brief Set the angle map to angle from equilibrium
            */
            void setMapAngleAngleEq();

            /**
             * @brief Set the bond map to length
            */
            void setMapBondLength();

            /**
             * @brief Set the bond map to length from equilibrium
            */
            void setMapBondLengthEq();

            /**
             * @brief Set the dihedral map to angle
            */
            void setMapDihedralAngle();

            /**
             * @brief Set the dihedral map to angle from equilibrium
            */
            void setMapDihedralAngleEq();
            
            /**
             * @brief Try to set the colormap. 
             * 
             * If the map doesn't exist, does not do anything and returns false.
             * 
             * @param s name of color map
             * @return true on success
             */
            bool set_colormap(const std::string& s);

            /**
             * @brief Get all available color map names
            */
            static std::vector<std::string> getNames();

            /**
             * @brief Try to get the current color map name
            */
            std::string getColorMapName() const;

        private:

            AngleColorMapperFunc mapper_angle;
            BondColorMapperFunc mapper_bond;
            DihedralColorMapperFunc mapper_dihedral;
            ParticleColorMapperFunc mapper_particle;

            unsigned int map_enum_angle;
            unsigned int map_enum_bond;
            unsigned int map_enum_dihedral;
            unsigned int map_enum_particle;
        };

    }

    namespace io {

        template <>
        HRESULT toFile(const rendering::ColorMapper &dataElement, const MetaData &metaData, IOElement &fileElement);

        template <>
        HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, rendering::ColorMapper *dataElement);

    }

};

#endif // _SOURCE_RENDERING_TFCOLORMAPPER_H_