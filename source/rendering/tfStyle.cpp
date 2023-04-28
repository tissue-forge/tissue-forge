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

#include "tfStyle.h"
#include <tfEngine.h>
#include <tfSpace.h>
#include <tf_util.h>
#include <tfError.h>
#include <io/tfFIO.h>
#include "tfColorMapper.h"


using namespace TissueForge;


HRESULT rendering::Style::setColor(const std::string &colorName) {
    color = util::Color3_Parse(colorName);
    return S_OK;
}

HRESULT rendering::Style::setFlag(StyleFlags flag, bool value) {
    if(flag == STYLE_VISIBLE) {
        if(value) this->flags |= STYLE_VISIBLE;
        else this->flags &= ~STYLE_VISIBLE;
        return space_update_style(&_Engine.s);
    }
    return tf_error(E_FAIL, "invalid flag id");
}

fVector4 rendering::Style::map_color(struct Particle *p) {
    if(mapper_func) {
        return mapper_func(mapper, p);
    }
    return fVector4{color, 1};
};

rendering::Style::Style(const fVector3 *color, const bool &visible, uint32_t flags, rendering::ColorMapper *cmap) : mapper_func(NULL) {
    init(color, visible, flags, cmap);
}

rendering::Style::Style(const std::string &color, const bool &visible, uint32_t flags, rendering::ColorMapper *cmap) : 
    rendering::Style()
{
    fVector3 c = util::Color3_Parse(color);
    init(&c, visible, flags, cmap);
}

rendering::Style::Style(const rendering::Style &other) {
    init(&other.color, true, other.flags, other.mapper);
}

const bool rendering::Style::getVisible() const {
    return flags & STYLE_VISIBLE;
}

void rendering::Style::setVisible(const bool &visible) {
    setFlag(STYLE_VISIBLE, visible);
}

std::string rendering::Style::getColorMap() const {
    return mapper ? mapper->getColorMapName() : "";
}

rendering::ColorMapper *rendering::Style::getColorMapper() const {
    return mapper;
}

void rendering::Style::setColorMap(const std::string &colorMap) {
    if(mapper) {
        mapper->set_colormap(colorMap);
        mapper_func = mapper->map;
    }
}

void rendering::Style::setColorMapper(rendering::ColorMapper *cmap) {
    if(cmap) {
        this->mapper = cmap;
        this->mapper_func = this->mapper->map;
    }
}

void rendering::Style::newColorMapper(
    struct ParticleType *partType,
    const std::string &speciesName, 
    const std::string &name, 
    float min, 
    float max) 
{
    setColorMapper(new rendering::ColorMapper(partType, speciesName, name, min, max));
}

int rendering::Style::init(const fVector3 *color, const bool &visible, uint32_t flags, rendering::ColorMapper *cmap) {
    this->flags = flags;

    this->color = color ? *color : fVector3(util::Color3_Parse("steelblue"));

    setVisible(visible);
    setColorMapper(cmap);

    return S_OK;
}

std::string rendering::Style::toString() {
    return io::toString(*this);
}

rendering::Style *rendering::Style::fromString(const std::string &str) {
    return new rendering::Style(io::fromString<rendering::Style>(str));
}


namespace TissueForge::io {

    template <>
    HRESULT toFile(const rendering::Style &dataElement, const MetaData &metaData, IOElement &fileElement) { 
        
        fVector3 color = {dataElement.color.r(), dataElement.color.g(), dataElement.color.b()};
        TF_IOTOEASY(fileElement, metaData, "color", color);
        TF_IOTOEASY(fileElement, metaData, "flags", dataElement.flags);

        if(dataElement.mapper != NULL) {
            TF_IOTOEASY(fileElement, metaData, "mapper", *dataElement.mapper);
        }

        fileElement.get()->type = "Style";

        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, rendering::Style *dataElement) { 

        fVector3 color;
        TF_IOFROMEASY(fileElement, metaData, "color", &color);
        dataElement->color = {color.x(), color.y(), color.z()};

        TF_IOFROMEASY(fileElement, metaData, "flags", &dataElement->flags);

        IOChildMap fec = IOElement::children(fileElement);
        auto feItr = fec.find("mapper");
        if(feItr != fec.end()) {
            rendering::ColorMapper *mapper = new rendering::ColorMapper();
            TF_IOFROMEASY(fileElement, metaData, "mapper", mapper);
            dataElement->setColorMapper(mapper);
        }

        return S_OK;
    }

};
