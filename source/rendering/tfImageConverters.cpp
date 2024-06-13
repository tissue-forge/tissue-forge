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

#include <Magnum/Trade/Trade.h>
#include <Corrade/Utility/ConfigurationGroup.h>
#include <Magnum/ImageView.h>
#include "tfImageConverters.h"
#include <MagnumPlugins/TgaImageConverter/TgaImageConverter.h>
#include <MagnumPlugins/StbImageConverter/StbImageConverter.h>


using namespace Magnum;
using namespace Magnum::Trade;
using namespace Corrade;
using namespace TissueForge;


Containers::Array<char> rendering::convertImageDataToBMP(const ImageView2D& image) {
    StbImageConverter conv(StbImageConverter::Format::Bmp);
    return conv.exportToData(image);
}

Containers::Array<char> rendering::convertImageDataToHDR(const ImageView2D& image) {
    StbImageConverter conv(StbImageConverter::Format::Hdr);
    return conv.exportToData(image);
}

Containers::Array<char> rendering::convertImageDataToJpeg(const ImageView2D& image, int jpegQuality) {
    StbImageConverter conv(StbImageConverter::Format::Jpeg);
    conv.configuration().setValue("jpegQuality", float(jpegQuality) / 100.f);
    return conv.exportToData(image);
}

Containers::Array<char> rendering::convertImageDataToPNG(const ImageView2D& image) {
    StbImageConverter conv(StbImageConverter::Format::Png);
    return conv.exportToData(image);
}

Containers::Array<char> rendering::convertImageDataToTGA(const ImageView2D& image) {
    TgaImageConverter conv;
    return conv.exportToData(image);
}
