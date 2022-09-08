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

/**
 * @file tfColorMapper.h
 * 
 */

#ifndef _SOURCE_RENDERING_TFCOLORMAPPER_H_
#define _SOURCE_RENDERING_TFCOLORMAPPER_H_

#include "tfStyle.h"

#include <vector>


namespace TissueForge {


    struct ParticleType;


    namespace rendering {


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

            ColorMapper() {}
            
            /**
             * @brief Construct a new color map for a particle type and species
             * 
             * @param partType particle type
             * @param speciesName name of species
             * @param name name of color mapper function
             * @param min minimum value of map
             * @param max maximum value of map
             */
            ColorMapper(
                struct ParticleType *partType,
                const std::string &speciesName, 
                const std::string &name="rainbow", 
                float min=0.0f, 
                float max=1.0f
            );
            ~ColorMapper() {};
            
            /**
             * @brief Try to set the colormap. 
             * 
             * If the map doesn't exist, does not do anything and returns false.
             * 
             * @param s name of color map
             * @return true on success
             */
            bool set_colormap(const std::string& s);

            static std::vector<std::string> getNames();

            std::string getColorMapName() const;
        };

    }

    namespace io {

        template <>
        HRESULT toFile(const rendering::ColorMapper &dataElement, const MetaData &metaData, IOElement *fileElement);

        template <>
        HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, rendering::ColorMapper *dataElement);

    }

};

#endif // _SOURCE_RENDERING_TFCOLORMAPPER_H_