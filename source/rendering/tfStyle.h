/*******************************************************************************
 * This file is part of Tissue Forge.
 * Copyright (c) 2022-2024 T.J. Sego
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
    struct Angle;
    struct Bond;
    struct Dihedral;

    struct ParticleType;


    namespace rendering {

        /**
         * @brief The Tissue Forge style type
         */
        struct CAPI_EXPORT Style
        {
            /** Default color */
            fVector3 color;

            /** Style flags */
            uint32_t flags;
            
            /**
             * @brief Color mapper of this style
             */
            struct ColorMapper *mapper = NULL;

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

            /**
             * @brief Set a style flag
            */
            HRESULT setFlag(StyleFlags flag, bool value);
            
            /**
             * @brief Map a particle to a color
            */
            fVector4 map_color(struct Particle *p);

            /**
             * @brief Map an angle to a color
            */
            fVector4 map_color(struct Angle* a);

            /**
             * @brief Map a bond to a color
            */
            fVector4 map_color(struct Bond* b);

            /**
             * @brief Map a dihedral to a color
            */
            fVector4 map_color(struct Dihedral* d);

            /**
             * @brief Test whether visible
            */
            const bool getVisible() const;

            /**
             * @brief Set whether visible
            */
            void setVisible(const bool &visible);

            /**
             * @brief Get the name of the color map, if any
            */
            std::string getColorMap() const;

            /**
             * @brief Get the color mapper, if any
            */
            ColorMapper *getColorMapper() const;

            /**
             * @brief Create a color mapper by name
            */
            void setColorMap(const std::string &colorMap);

            /**
             * @brief Set the color mapper
            */
            void setColorMapper(ColorMapper *cmap);

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