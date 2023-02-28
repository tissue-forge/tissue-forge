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

#ifndef _SOURCE_RENDERING_TFSTYLE_H_
#define _SOURCE_RENDERING_TFSTYLE_H_

#include <io/tf_io.h>
#include <TissueForge_private.h>

#include <string>


namespace TissueForge {


    struct Particle;
    struct ParticleType;


    namespace rendering {


        typedef fVector4 (*ColorMapperFunc)(struct ColorMapper *mapper, struct Particle *p);

        /**
         * @brief The Tissue Forge style type
         */
        struct CAPI_EXPORT Style
        {
            fVector3 color;
            uint32_t flags;
            
            /**
             * @brief Color mapper of this style
             */
            struct ColorMapper *mapper = NULL;
            
            ColorMapperFunc mapper_func;

            Style(const fVector3 *color=NULL, const bool &visible=true, uint32_t flags=STYLE_VISIBLE, ColorMapper *cmap=NULL);

            /**
             * @brief Construct a new style
             * 
             * @param color name of color
             * @param visible visibility flag
             * @param flags style flags
             * @param cmap color mapper
             */
            Style(const std::string &color, const bool &visible=true, uint32_t flags=STYLE_VISIBLE, ColorMapper *cmap=NULL);
            Style(const Style &other);

            int init(const fVector3 *color=NULL, const bool &visible=true, uint32_t flags=STYLE_VISIBLE, ColorMapper *cmap=NULL);

            /**
             * @brief Set the color by name
             * 
             * @param colorName name of color
             * @return HRESULT 
             */
            HRESULT setColor(const std::string &colorName);
            HRESULT setFlag(StyleFlags flag, bool value);
            
            fVector4 map_color(struct Particle *p);

            const bool getVisible() const;
            void setVisible(const bool &visible);
            std::string getColorMap() const;
            ColorMapper *getColorMapper() const;
            void setColorMap(const std::string &colorMap);
            void setColorMapper(ColorMapper *cmap);

            /**
             * @brief Construct and apply a new color map for a particle type and species
             * 
             * @param partType particle type
             * @param speciesName name of species
             * @param name name of color map
             * @param min minimum value of map
             * @param max maximum value of map
             */
            void newColorMapper(
                struct ParticleType *partType,
                const std::string &speciesName, 
                const std::string &name="rainbow", 
                float min=0.0f, 
                float max=1.0f
            );

            /**
             * @brief Get a JSON string representation
             * 
             * @return std::string 
             */
            std::string toString();

            /**
             * @brief Create from a JSON string representation. 
             * 
             * @param str 
             * @return Style* 
             */
            static Style *fromString(const std::string &str);

        };

    }


    namespace io {

        template <>
        HRESULT toFile(const rendering::Style &dataElement, const MetaData &metaData, IOElement &fileElement);

        template <>
        HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, rendering::Style *dataElement);

    };

}

#endif // _SOURCE_RENDERING_TFSTYLE_H_