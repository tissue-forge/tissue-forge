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

#ifndef _SOURCE_RENDERING_TFIMAGECONVERTERS_H_
#define _SOURCE_RENDERING_TFIMAGECONVERTERS_H_

#include <Corrade/Containers/Array.h>
#include <Magnum/ImageView.h>


namespace TissueForge { 


    namespace rendering {


        /**
         * jpegQuality shall construct JPEG quantization tables for the given quality setting.
         * The quality value ranges from 0..100.
         */
        Corrade::Containers::Array<char> convertImageDataToJpeg(const Magnum::ImageView2D& image, int jpegQuality = 100);

        Corrade::Containers::Array<char> convertImageDataToBMP(const Magnum::ImageView2D& image);

        Corrade::Containers::Array<char> convertImageDataToHDR(const Magnum::ImageView2D& image);

        Corrade::Containers::Array<char> convertImageDataToPNG(const Magnum::ImageView2D& image);

        Corrade::Containers::Array<char> convertImageDataToTGA(const Magnum::ImageView2D& image);

}}

#endif // _SOURCE_RENDERING_TFIMAGECONVERTERS_H_