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

#ifndef _SOURCE_SHADERS_TFPHONG_H_
#define _SOURCE_SHADERS_TFPHONG_H_

#include "Magnum/GL/AbstractShaderProgram.h"
#include "Magnum/Shaders/Generic.h"
#include "Magnum/Shaders/visibility.h"


using namespace Magnum;


namespace TissueForge::shaders {


    class Phong: public GL::AbstractShaderProgram {
    public:
        /**
         * @brief Vertex position
         */
        typedef Magnum::Shaders::Generic3D::Position Position;
        
        /**
         * @brief Normal direction
         */
        typedef Magnum::Shaders::Generic3D::Normal Normal;
        
        /**
         * @brief Tangent direction
         */
        typedef Magnum::Shaders::Generic3D::Tangent Tangent;
        
        /**
         * @brief Tangent direction with a bitangent sign
         */
        typedef typename Magnum::Shaders::Generic3D::Tangent4 Tangent4;
        
        /**
         * @brief Bitangent direction
         */
        typedef typename Magnum::Shaders::Generic3D::Bitangent Bitangent;
        
        /**
         * @brief 2D texture coordinates
         */
        typedef Magnum::Shaders::Generic3D::TextureCoordinates TextureCoordinates;
        
        /**
         * @brief Three-component vertex color
         */
        typedef Magnum::Shaders::Generic3D::Color3 Color3;
        
        /**
         * @brief Four-component vertex color
         */
        typedef Magnum::Shaders::Generic3D::Color4 Color4;
        
    #ifndef MAGNUM_TARGET_GLES2
        /**
         * @brief (Instanced) object ID
         */
        typedef Magnum::Shaders::Generic3D::ObjectId ObjectId;
    #endif
        
        /**
         * @brief (Instanced) transformation matrix
         */
        typedef Magnum::Shaders::Generic3D::TransformationMatrix TransformationMatrix;
        
        /**
         * @brief (Instanced) normal matrix
         */
        typedef Magnum::Shaders::Generic3D::NormalMatrix NormalMatrix;
        
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
        
        /**
         * @brief Flag
         */
        enum class Flag: UnsignedShort {
            /**
             * Multiply ambient color with a texture.
             */
            AmbientTexture = 1 << 0,
            
            /**
             * Multiply diffuse color with a texture.
             */
            DiffuseTexture = 1 << 1,
            
            /**
             * Multiply specular color with a texture.
             */
            SpecularTexture = 1 << 2,
            
            /**
             * Modify normals according to a texture. 
             */
            NormalTexture = 1 << 4,
            
            /**
             * Enable alpha masking. 
             */
            AlphaMask = 1 << 3,
            
            /**
             * Multiply diffuse color with a vertex color. 
             */
            VertexColor = 1 << 5,
            
            /**
             * Use the separate @ref Bitangent attribute for retrieving vertex bitangents. 
             */
            Bitangent = 1 << 11,
            
            /**
             * Enable texture coordinate transformation. 
             */
            TextureTransformation = 1 << 6,
            
    #ifndef MAGNUM_TARGET_GLES2
            /**
             * Enable object ID output. 
             */
            ObjectId = 1 << 7,
            
            /**
             * Instanced object ID. 
             */
            InstancedObjectId = (1 << 8)|ObjectId,
    #endif
            
            /**
             * Instanced transformation. 
             */
            InstancedTransformation = 1 << 9,
            
            /**
             * Instanced texture offset. 
             */
            InstancedTextureOffset = (1 << 10)|TextureTransformation
        };
        
        /**
         * @brief Flags
         */
        typedef Containers::EnumSet<Flag> Flags;
        
        /**
         * @brief Constructor
         * @param flags         Flags
         * @param lightCount    Count of light sources
         */
        explicit Phong(Flags flags = {}, unsigned lightCount = 1, unsigned clipPlaneCount = 0);
        
        /**
         * @brief Construct without creating the underlying OpenGL object
         */
        explicit Phong(NoCreateT) noexcept: GL::AbstractShaderProgram{NoCreate} {}
        
        /** @brief Copying is not allowed */
        Phong(const Phong&) = delete;
        
        /** @brief Move constructor */
        Phong(Phong&&) noexcept = default;
        
        /** @brief Copying is not allowed */
        Phong& operator=(const Phong&) = delete;
        
        /** @brief Move assignment */
        Phong& operator=(Phong&&) noexcept = default;
        
        /** @brief Flags */
        Flags flags() const { return _flags; }
        
        /** @brief Light count */
        UnsignedInt lightCount() const { return _lightCount; }
        
        /**
         * @brief Set ambient color
         * @return Reference to self (for method chaining)
         */
        Phong& setAmbientColor(const Magnum::Color4& color);
        
        /**
         * @brief Bind an ambient texture
         * @return Reference to self (for method chaining)
         */
        Phong& bindAmbientTexture(GL::Texture2D& texture);
        
        /**
         * @brief Set diffuse color
         * @return Reference to self (for method chaining)
         */
        Phong& setDiffuseColor(const Magnum::Color4& color);
        
        /**
         * @brief Bind a diffuse texture
         * @return Reference to self (for method chaining)
         */
        Phong& bindDiffuseTexture(GL::Texture2D& texture);
        
        /**
         * @brief Set normal texture scale
         * @return Reference to self (for method chaining)
         */
        Phong& setNormalTextureScale(Float scale);
        
        /**
         * @brief Bind a normal texture
         * @return Reference to self (for method chaining)
         */
        Phong& bindNormalTexture(GL::Texture2D& texture);
        
        /**
         * @brief Set specular color
         * @return Reference to self (for method chaining)
         */
        Phong& setSpecularColor(const Magnum::Color4& color);
        
        /**
         * @brief Bind a specular texture
         * @return Reference to self (for method chaining)
         */
        Phong& bindSpecularTexture(GL::Texture2D& texture);
        
        /**
         * @brief Bind textures
         * @return Reference to self (for method chaining)
         */
        Phong& bindTextures(GL::Texture2D* ambient, GL::Texture2D* diffuse, GL::Texture2D* specular, GL::Texture2D* normal
    #ifdef MAGNUM_BUILD_DEPRECATED
                            = nullptr
    #endif
        );
        
        /**
         * @brief Set shininess
         * @return Reference to self (for method chaining)
         */
        Phong& setShininess(Float shininess);
        
        /**
         * @brief Set alpha mask value
         * @return Reference to self (for method chaining)
         */
        Phong& setAlphaMask(Float mask);
        
    #ifndef MAGNUM_TARGET_GLES2
        /**
         * @brief Set object ID
         * @return Reference to self (for method chaining)
         */
        Phong& setObjectId(UnsignedInt id);
    #endif
        
        /**
         * @brief Set transformation matrix
         * @return Reference to self (for method chaining)
         */
        Phong& setTransformationMatrix(const Matrix4& matrix);
        
        /**
         * @brief Set normal matrix
         * @return Reference to self (for method chaining)
         */
        Phong& setNormalMatrix(const Matrix3x3& matrix);
        
        /**
         * @brief Set projection matrix
         * @return Reference to self (for method chaining)
         */
        Phong& setProjectionMatrix(const Matrix4& matrix);
        
        /**
         * @brief Set texture coordinate transformation matrix
         * @return Reference to self (for method chaining)
         */
        Phong& setTextureMatrix(const Matrix3& matrix);
        
        /**
         * @brief Set light positions
         * @return Reference to self (for method chaining)
        */
        Phong& setLightPositions(Containers::ArrayView<const Vector4> positions);
        
        Phong& setLightPositions(std::initializer_list<Vector4> positions);
        
    #ifdef MAGNUM_BUILD_DEPRECATED
    
        CORRADE_DEPRECATED("use setLightPositions(Containers::ArrayView<const Vector4>) instead") Phong& setLightPositions(Containers::ArrayView<const Vector3> positions);
        
        CORRADE_DEPRECATED("use setLightPositions(std::initializer_list<Vector4>) instead") Phong& setLightPositions(std::initializer_list<Vector3> positions);
    #endif
        
        /**
         * @brief Set position for given light
         * @return Reference to self (for method chaining)
         */
        Phong& setLightPosition(UnsignedInt id, const Vector4& position);
        
        
        /**
         * @brief Set clip plane equation for given clip plane
         * @return Reference to self (for method chaining)
         */
        Phong& setclipPlaneEquation(UnsignedInt id, const Vector4& position);
        
    #ifdef MAGNUM_BUILD_DEPRECATED

        CORRADE_DEPRECATED("use setLightPosition(UnsignedInt, const Vector4&) instead") Phong& setLightPosition(UnsignedInt id, const Vector3& position);
        
        CORRADE_DEPRECATED("use setLightPositions(std::initializer_list<Vector4>) instead") Phong& setLightPosition(const Vector3& position);
    #endif
        
        /**
         * @brief Set light colors
         * @return Reference to self (for method chaining)
         */
        Phong& setLightColors(Containers::ArrayView<const Magnum::Color3> colors);
        
        Phong& setLightColors(std::initializer_list<Magnum::Color3> colors);
        
    #ifdef MAGNUM_BUILD_DEPRECATED

        CORRADE_DEPRECATED("use setLightColors(Containers::ArrayView<const Magnum::Color3>) instead") Phong& setLightColors(Containers::ArrayView<const Magnum::Color4> colors);
        
        CORRADE_DEPRECATED("use setLightColors(std::initializer_list<Magnum::Color3>) instead") Phong& setLightColors(std::initializer_list<Magnum::Color4> colors);
    #endif
        
        /**
         * @brief Set position for given light
         * @return Reference to self (for method chaining)
         */
        Phong& setLightColor(UnsignedInt id, const Magnum::Color3& color);
        
    #ifdef MAGNUM_BUILD_DEPRECATED

        CORRADE_DEPRECATED("use setLightColor(UnsignedInt, const Magnum::Color3&) instead") Phong& setLightColor(UnsignedInt id, const Magnum::Color4& color);
        
        CORRADE_DEPRECATED("use setLightColor(std::initializer_list<Color3>) instead") Phong& setLightColor(const Magnum::Color4& color);
    #endif
        
        /**
         * @brief Set light specular colors
         * @return Reference to self (for method chaining)
         */
        Phong& setLightSpecularColors(Containers::ArrayView<const Magnum::Color3> colors);
        
        Phong& setLightSpecularColors(std::initializer_list<Magnum::Color3> colors);
        
        /**
         * @brief Set position for given light
         * @return Reference to self (for method chaining)
         */
        Phong& setLightSpecularColor(UnsignedInt id, const Magnum::Color3& color);
        
        /**
         * @brief Set light attenuation ranges
         * @return Reference to self (for method chaining)
         */
        Phong& setLightRanges(Containers::ArrayView<const Float> ranges);
        
        Phong& setLightRanges(std::initializer_list<Float> ranges);
        
        /**
         * @brief Set attenuation range for given light
         * @return Reference to self (for method chaining)
         */
        Phong& setLightRange(UnsignedInt id, Float range);
        
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
        UnsignedInt _lightCount{};
        Int _transformationMatrixUniform{0},
        _projectionMatrixUniform{1},
        _normalMatrixUniform{2},
        _textureMatrixUniform{3},
        _ambientColorUniform{4},
        _diffuseColorUniform{5},
        _specularColorUniform{6},
        _shininessUniform{7},
        _normalTextureScaleUniform{8},
        _alphaMaskUniform{9};
    #ifndef MAGNUM_TARGET_GLES2
        Int _objectIdUniform{10};
    #endif
        Int _lightPositionsUniform{11},
        _lightColorsUniform, /* 11 + lightCount, set in the constructor */
        _lightSpecularColorsUniform, /* 11 + 2*lightCount */
        _lightRangesUniform; /* 11 + 3*lightCount */
        
        
        UnsignedInt _clipPlaneCount{1};
        
        Int _clipPlanesUniform; /*_lightRangesUniform + 1, or  11 + 4*lightCount, set in the constructor */
    };

    MAGNUM_SHADERS_EXPORT Debug& operator<<(Debug& debug, Phong::Flag value);
    MAGNUM_SHADERS_EXPORT Debug& operator<<(Debug& debug, Phong::Flags value);
    CORRADE_ENUMSET_OPERATORS(Phong::Flags)

}

#endif // _SOURCE_SHADERS_TFPHONG_H_