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

/*
Derived from Magnum with the following notice:

    Copyright © 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019,
                2020, 2021 Vladimír Vondruš <mosra@centrum.cz>

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included
    in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.
*/

#ifndef _SOURCE_SHADERS_TFFLAT3D_H_
#define _SOURCE_SHADERS_TFFLAT3D_H_

#include <Magnum/DimensionTraits.h>
#include <Magnum/GL/AbstractShaderProgram.h>
#include <Magnum/Shaders/Generic.h>
#include <Magnum/Shaders/visibility.h>


using namespace Magnum;


namespace TissueForge::shaders { 


    class Flat3D: public GL::AbstractShaderProgram {
    public:
        /**
         * @brief Vertex position
         */
        typedef typename Magnum::Shaders::Generic3D::Position Position;

        /**
         * @brief 2D texture coordinates
         */
        typedef typename Magnum::Shaders::Generic3D::TextureCoordinates TextureCoordinates;

        /**
         * @brief Three-component vertex color
         */
        typedef typename Magnum::Shaders::Generic3D::Color3 Color3;

        /**
         * @brief Four-component vertex color
         */
        typedef typename Magnum::Shaders::Generic3D::Color4 Color4;

        #ifndef MAGNUM_TARGET_GLES2
        /**
         * @brief (Instanced) object ID
         */
        typedef typename Magnum::Shaders::Generic3D::ObjectId ObjectId;
        #endif

        /**
         * @brief (Instanced) transformation matrix
         */
        typedef typename Magnum::Shaders::Generic3D::TransformationMatrix TransformationMatrix;

        /**
         * @brief (Instanced) texture offset
         */
        typedef typename Magnum::Shaders::Generic3D::TextureOffset TextureOffset;

        enum: UnsignedInt {
            /**
             * Color shader output. 
             */
            ColorOutput = Magnum::Shaders::Generic3D::ColorOutput,

            #ifndef MAGNUM_TARGET_GLES2
            /**
             * Object ID shader output. 
             */
            ObjectIdOutput = Magnum::Shaders::Generic3D::ObjectIdOutput
            #endif
        };

        enum class Flag: UnsignedByte {
            Textured = 1 << 0,
            AlphaMask = 1 << 1,
            VertexColor = 1 << 2,
            TextureTransformation = 1 << 3,
            #ifndef MAGNUM_TARGET_GLES2
            ObjectId = 1 << 4,
            InstancedObjectId = (1 << 5)|ObjectId,
            #endif
            InstancedTransformation = 1 << 6,
            InstancedTextureOffset = (1 << 7)|TextureTransformation
        };

        typedef Containers::EnumSet<Flag> Flags;

        /**
         * @brief Constructor
         * @param flags     Flags
         */
        explicit Flat3D(Flags flags = {}, unsigned clipPlaneCount = 0);

        /**
         * @brief Construct without creating the underlying OpenGL object
         */
        explicit Flat3D(NoCreateT) noexcept: GL::AbstractShaderProgram{NoCreate} {}

        /** @brief Copying is not allowed */
        Flat3D(const Flat3D&) = delete;

        /** @brief Move constructor */
        Flat3D(Flat3D&&) noexcept = default;

        /** @brief Copying is not allowed */
        Flat3D& operator=(const Flat3D&) = delete;

        /** @brief Move assignment */
        Flat3D& operator=(Flat3D&&) noexcept = default;

        /** @brief Flags */
        Flags flags() const { return _flags; }

        /**
         * @brief Set transformation and projection matrix
         * @return Reference to self (for method chaining)
         *
         * Initial value is an identity matrix.
         */
        Flat3D& setTransformationProjectionMatrix(const MatrixTypeFor<3, Float>& matrix);

        /**
         * @brief Set texture coordinate transformation matrix
         * @return Reference to self (for method chaining)
         */
        Flat3D& setTextureMatrix(const Matrix3& matrix);

        /**
         * @brief Set color
         * @return Reference to self (for method chaining)
         */
        Flat3D& setColor(const Magnum::Color4& color);

        /**
         * @brief Bind a color texture
         * @return Reference to self (for method chaining)
         */
        Flat3D& bindTexture(GL::Texture2D& texture);

        /**
         * @brief Set alpha mask value
         * @return Reference to self (for method chaining)
         */
        Flat3D& setAlphaMask(Float mask);

        /**
         * @brief Set clip plane equation for given clip plane
         * @return Reference to self (for method chaining)
         */
        Flat3D& setclipPlaneEquation(UnsignedInt id, const Vector4& position);

        #ifndef MAGNUM_TARGET_GLES2
        /**
         * @brief Set object ID
         * @return Reference to self (for method chaining)
         */
        Flat3D& setObjectId(UnsignedInt id);
        #endif
    
        unsigned clipPlaneCount() const {
            return _clipPlaneCount;
        }

    private:
        /* Prevent accidentally calling irrelevant functions */
        #ifndef MAGNUM_TARGET_GLES
        using GL::AbstractShaderProgram::drawTransformFeedback;
        #endif
        #if !defined(MAGNUM_TARGET_GLES2) && !defined(MAGNUM_TARGET_WEBGL)
        using GL::AbstractShaderProgram::dispatchCompute;
        #endif

        Flags _flags;
        Int _transformationProjectionMatrixUniform{0},
            _textureMatrixUniform{1},
            _colorUniform{2},
            _alphaMaskUniform{3};
        #ifndef MAGNUM_TARGET_GLES2
        Int _objectIdUniform{4};
        #endif
        Int _clipPlanesUniform{5};
        
        UnsignedInt _clipPlaneCount{0};
    };

    MAGNUM_SHADERS_EXPORT Debug& operator<<(Debug& debug, Flat3D::Flag value);
    MAGNUM_SHADERS_EXPORT Debug& operator<<(Debug& debug, Flat3D::Flags value);
    CORRADE_ENUMSET_OPERATORS(Flat3D::Flags)

}

#endif // _SOURCE_SHADERS_TFFLAT3D_H_