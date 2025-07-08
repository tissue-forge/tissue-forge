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

#include "tfGlInfo.h"

#include <Corrade/Utility/Arguments.h>
#include <Corrade/Utility/Debug.h>
#include <Corrade/Utility/DebugStl.h>
#include <Corrade/Utility/String.h>

#include "Magnum/GL/AbstractShaderProgram.h"
#include "Magnum/GL/Buffer.h"
#if !defined(MAGNUM_TARGET_GLES2) && !defined(MAGNUM_TARGET_WEBGL)
#include "Magnum/GL/BufferTexture.h"
#endif
#include "Magnum/GL/Context.h"
#include "Magnum/GL/CubeMapTexture.h"
#if !defined(MAGNUM_TARGET_GLES2) && !defined(MAGNUM_TARGET_WEBGL)
#include "Magnum/GL/CubeMapTextureArray.h"
#endif
#ifndef MAGNUM_TARGET_WEBGL
#include "Magnum/GL/DebugOutput.h"
#endif
#include "Magnum/GL/Extensions.h"
#include "Magnum/GL/Framebuffer.h"
#include "Magnum/GL/Mesh.h"
#if !defined(MAGNUM_TARGET_GLES2) && !defined(MAGNUM_TARGET_WEBGL)
#include "Magnum/GL/MultisampleTexture.h"
#endif
#ifndef MAGNUM_TARGET_GLES
#include "Magnum/GL/RectangleTexture.h"
#endif
#include "Magnum/GL/Renderer.h"
#include "Magnum/GL/Renderbuffer.h"
#include "Magnum/GL/Shader.h"
#include "Magnum/GL/Texture.h"
#ifndef MAGNUM_TARGET_GLES2
#include "Magnum/GL/TextureArray.h"
#include "Magnum/GL/TransformFeedback.h"
#endif


#include "tfWindowless.h"

#include <sstream>


using namespace Magnum;
using namespace TissueForge;




std::string rendering::gl_info() {



    std::stringstream os;


    os << "";
    os << "  +---------------------------------------------------------+";
    os << "  |   Information about OpenGL capabilities   |";
    os << "  +---------------------------------------------------------+";
    os << "";
    
    


    #ifdef MAGNUM_WINDOWLESSEGLAPPLICATION_MAIN
    os << "Used application: Platform::WindowlessEglApplication" << std::endl;
    #elif defined(MAGNUM_WINDOWLESSIOSAPPLICATION_MAIN)
    os << "Used application: Platform::WindowlessIosApplication" << std::endl;
    #elif defined(MAGNUM_WINDOWLESSCGLAPPLICATION_MAIN)
    os << "Used application: Platform::WindowlessCglApplication" << std::endl;
    #elif defined(MAGNUM_WINDOWLESSGLXAPPLICATION_MAIN)
    os << "Used application: Platform::WindowlessGlxApplication" << std::endl;
    #elif defined(MAGNUM_WINDOWLESSWGLAPPLICATION_MAIN)
    os << "Used application: Platform::WindowlessWglApplication" << std::endl;
    #elif defined(MAGNUM_WINDOWLESSWINDOWSEGLAPPLICATION_MAIN)
    os << "Used application: Platform::WindowlessWindowsEglApplication" << std::endl;
    #else
    os << "No windowless application available on this platform" << std::endl;
    #endif
    os << "Compilation flags:";
    #ifdef CORRADE_BUILD_DEPRECATED
    os << "    CORRADE_BUILD_DEPRECATED" << std::endl;
    #endif
    #ifdef CORRADE_BUILD_STATIC
    os << "    CORRADE_BUILD_STATIC" << std::endl;
    #endif
    #ifdef CORRADE_BUILD_MULTITHREADED
    os << "    CORRADE_BUILD_MULTITHREADED" << std::endl;
    #endif
    #ifdef CORRADE_TARGET_UNIX
    os << "    CORRADE_TARGET_UNIX" << std::endl;
    #endif
    #ifdef CORRADE_TARGET_APPLE
    os << "    CORRADE_TARGET_APPLE" << std::endl;
    #endif
    #ifdef CORRADE_TARGET_IOS
    os << "    CORRADE_TARGET_IOS" << std::endl;
    #endif
    #ifdef CORRADE_TARGET_WINDOWS
    os << "    CORRADE_TARGET_WINDOWS" << std::endl;
    #endif
    #ifdef CORRADE_TARGET_WINDOWS_RT
    os << "    CORRADE_TARGET_WINDOWS_RT" << std::endl;
    #endif
    #ifdef CORRADE_TARGET_EMSCRIPTEN
    os << "    CORRADE_TARGET_EMSCRIPTEN (" << Debug::nospace
        << __EMSCRIPTEN_major__ << Debug::nospace << "." << Debug::nospace
        << __EMSCRIPTEN_minor__ << Debug::nospace << "." << Debug::nospace
       << __EMSCRIPTEN_tiny__ << Debug::nospace << ")" << std::endl;
    #endif
    #ifdef CORRADE_TARGET_ANDROID
    os << "    CORRADE_TARGET_ANDROID" << std::endl;
    #endif
    #ifdef CORRADE_TARGET_X86
    os << "    CORRADE_TARGET_X86" << std::endl;
    #endif
    #ifdef CORRADE_TARGET_ARM
    os << "    CORRADE_TARGET_ARM" << std::endl;
    #endif
    #ifdef CORRADE_TARGET_POWERPC
    os << "    CORRADE_TARGET_POWERPC" << std::endl;
    #endif
    #ifdef CORRADE_TARGET_LIBCXX
    os << "    CORRADE_TARGET_LIBCXX" << std::endl;
    #endif
    #ifdef CORRADE_TARGET_DINKUMWARE
    os << "    CORRADE_TARGET_DINKUMWARE" << std::endl;
    #endif
    #ifdef CORRADE_TARGET_LIBSTDCXX
    os << "    CORRADE_TARGET_LIBSTDCXX" << std::endl;
    #endif
    #ifdef CORRADE_PLUGINMANAGER_NO_DYNAMIC_PLUGIN_SUPPORT
    os << "    CORRADE_PLUGINMANAGER_NO_DYNAMIC_PLUGIN_SUPPORT" << std::endl;
    #endif
    #ifdef CORRADE_TESTSUITE_TARGET_XCTEST
    os << "    CORRADE_TESTSUITE_TARGET_XCTEST" << std::endl;
    #endif
    #ifdef CORRADE_UTILITY_USE_ANSI_COLORS
    os << "    CORRADE_UTILITY_USE_ANSI_COLORS" << std::endl;
    #endif
    #ifdef MAGNUM_BUILD_DEPRECATED
    os << "    MAGNUM_BUILD_DEPRECATED" << std::endl;
    #endif
    #ifdef MAGNUM_BUILD_STATIC
    os << "    MAGNUM_BUILD_STATIC" << std::endl;
    #endif
    #ifdef MAGNUM_TARGET_GLES
    os << "    MAGNUM_TARGET_GLES" << std::endl;
    #endif
    #ifdef MAGNUM_TARGET_GLES2
    os << "    MAGNUM_TARGET_GLES2" << std::endl;
    #endif
    #ifdef MAGNUM_TARGET_DESKTOP_GLES
    os << "    MAGNUM_TARGET_DESKTOP_GLES" << std::endl;
    #endif
    #ifdef MAGNUM_TARGET_WEBGL
    os << "    MAGNUM_TARGET_WEBGL" << std::endl;
    #endif
    #ifdef MAGNUM_TARGET_HEADLESS
    os << "    MAGNUM_TARGET_HEADLESS" << std::endl;
    #endif
    os << "" << std::endl;

    /* Create context here, so the context creation info is displayed at proper
       place */
    
    if(!GL::Context::hasCurrent()) {
        os << "N OpenGL Context";
        return os.str();
    }

    GL::Context& c = GL::Context::current();
    
    os << "vendor: " << c.vendorString().data();
    
    os << "version: " <<  c.versionString().data();
    
    os << "renderer: " <<  c.rendererString().data();
    
    os << "shading_language_version: " <<  c.shadingLanguageVersionString().data();
    
    {
        auto extensions = c.extensionStrings();
        
        os << "extensions: ";
        
        for (int i = 0; i < extensions.size(); ++i) {
            os << "\t  (" << i << ") " << extensions[i].data();
        }
    }

    os << "";


    /* Get first future (not supported) version */
    std::vector<GL::Version> versions{
        #ifndef MAGNUM_TARGET_GLES
        GL::Version::GL300,
        GL::Version::GL310,
        GL::Version::GL320,
        GL::Version::GL330,
        GL::Version::GL400,
        GL::Version::GL410,
        GL::Version::GL420,
        GL::Version::GL430,
        GL::Version::GL440,
        GL::Version::GL450,
        GL::Version::GL460,
        #else
        GL::Version::GLES300,
        #ifndef MAGNUM_TARGET_WEBGL
        GL::Version::GLES310,
        GL::Version::GLES320,
        #endif
        #endif
        GL::Version::None
    };
    
    std::size_t future = 0;

    while(versions[future] != GL::Version::None && c.isVersionSupported(versions[future]))
        ++future;

    /* Display supported OpenGL extensions from unsupported versions */
    for(std::size_t i = future; i != versions.size(); ++i) {
        if(versions[i] != GL::Version::None)
            Debug() << versions[i] << "extension support:";
        else Debug() << "Vendor extension support:";

        for(const auto& extension: GL::Extension::extensions(versions[i])) {
            std::string extensionName = extension.string();
            Debug d;
            d << "   " << extensionName << std::string(60-extensionName.size(), ' ');
            if(c.isExtensionSupported(extension))
                d << "SUPPORTED";
            else if(c.isExtensionDisabled(extension))
                d << " removed";
            else if(c.isVersionSupported(extension.requiredVersion()))
                d << "    -";
            else
                d << "   n/a";
        }

        Debug() << "";
    }


    /* Limits and implementation-defined values */
    #define _h(val) Debug() << "\n " << GL::Extensions::val::string() + std::string(":");
    #define _l(val) Debug() << "   " << #val << (sizeof(#val) > 64 ? "\n" + std::string(68, ' ') : std::string(64 - sizeof(#val), ' ')) << val;
    #define _lvec(val) Debug() << "   " << #val << (sizeof(#val) > 42 ? "\n" + std::string(46, ' ') : std::string(42 - sizeof(#val), ' ')) << val;

    Debug() << "Limits and implementation-defined values:";
    _lvec(GL::AbstractFramebuffer::maxViewportSize())
    _l(GL::AbstractFramebuffer::maxDrawBuffers())
    _l(GL::Framebuffer::maxColorAttachments())
    _l(GL::Mesh::maxVertexAttributeStride())
    #ifndef MAGNUM_TARGET_GLES2
    _l(GL::Mesh::maxElementIndex())
    _l(GL::Mesh::maxElementsIndices())
    _l(GL::Mesh::maxElementsVertices())
    #endif
    _lvec(GL::Renderer::lineWidthRange())
    _l(GL::Renderbuffer::maxSize())
    #if !(defined(MAGNUM_TARGET_WEBGL) && defined(MAGNUM_TARGET_GLES2))
    _l(GL::Renderbuffer::maxSamples())
    #endif
    _l(GL::Shader::maxVertexOutputComponents())
    _l(GL::Shader::maxFragmentInputComponents())
    _l(GL::Shader::maxTextureImageUnits(GL::Shader::Type::Vertex))
    #if !defined(MAGNUM_TARGET_GLES2) && !defined(MAGNUM_TARGET_WEBGL)
    _l(GL::Shader::maxTextureImageUnits(GL::Shader::Type::TessellationControl))
    _l(GL::Shader::maxTextureImageUnits(GL::Shader::Type::TessellationEvaluation))
    _l(GL::Shader::maxTextureImageUnits(GL::Shader::Type::Geometry))
    _l(GL::Shader::maxTextureImageUnits(GL::Shader::Type::Compute))
    #endif
    _l(GL::Shader::maxTextureImageUnits(GL::Shader::Type::Fragment))
    _l(GL::Shader::maxCombinedTextureImageUnits())
    _l(GL::Shader::maxUniformComponents(GL::Shader::Type::Vertex))
    #if !defined(MAGNUM_TARGET_GLES2) && !defined(MAGNUM_TARGET_WEBGL)
    _l(GL::Shader::maxUniformComponents(GL::Shader::Type::TessellationControl))
    _l(GL::Shader::maxUniformComponents(GL::Shader::Type::TessellationEvaluation))
    _l(GL::Shader::maxUniformComponents(GL::Shader::Type::Geometry))
    _l(GL::Shader::maxUniformComponents(GL::Shader::Type::Compute))
    #endif
    _l(GL::Shader::maxUniformComponents(GL::Shader::Type::Fragment))
    _l(GL::AbstractShaderProgram::maxVertexAttributes())
    #ifndef MAGNUM_TARGET_GLES2
    _l(GL::AbstractTexture::maxLodBias())
    #endif
    #ifndef MAGNUM_TARGET_GLES
    _lvec(GL::Texture1D::maxSize())
    #endif
    _lvec(GL::Texture2D::maxSize())
    #ifndef MAGNUM_TARGET_GLES2
    _lvec(GL::Texture3D::maxSize()) /* Checked ES2 version below */
    #endif
    _lvec(GL::CubeMapTexture::maxSize())

    #ifndef MAGNUM_TARGET_GLES
    if(c.isExtensionSupported<GL::Extensions::ARB::blend_func_extended>()) {
        _h(ARB::blend_func_extended)

        _l(GL::AbstractFramebuffer::maxDualSourceDrawBuffers())
    }
    #endif

    #if !defined(MAGNUM_TARGET_GLES2) && !defined(MAGNUM_TARGET_WEBGL)
    #ifndef MAGNUM_TARGET_GLES
    if(c.isExtensionSupported<GL::Extensions::ARB::compute_shader>())
    #endif
    {
        #ifndef MAGNUM_TARGET_GLES
        _h(ARB::compute_shader)
        #endif

        _l(GL::AbstractShaderProgram::maxComputeSharedMemorySize())
        _l(GL::AbstractShaderProgram::maxComputeWorkGroupInvocations())
        _lvec(GL::AbstractShaderProgram::maxComputeWorkGroupCount())
        _lvec(GL::AbstractShaderProgram::maxComputeWorkGroupSize())
    }

    #ifndef MAGNUM_TARGET_GLES
    if(c.isExtensionSupported<GL::Extensions::ARB::explicit_uniform_location>())
    #endif
    {
        #ifndef MAGNUM_TARGET_GLES
        _h(ARB::explicit_uniform_location)
        #endif

        _l(GL::AbstractShaderProgram::maxUniformLocations())
    }
    #endif

    #ifndef MAGNUM_TARGET_GLES
    if(c.isExtensionSupported<GL::Extensions::ARB::map_buffer_alignment>()) {
        _h(ARB::map_buffer_alignment)

        _l(GL::Buffer::minMapAlignment())
    }
    #endif

    #if !defined(MAGNUM_TARGET_GLES2) && !defined(MAGNUM_TARGET_WEBGL)
    #ifndef MAGNUM_TARGET_GLES
    if(c.isExtensionSupported<GL::Extensions::ARB::shader_atomic_counters>())
    #endif
    {
        #ifndef MAGNUM_TARGET_GLES
        _h(ARB::shader_atomic_counters)
        #endif

        _l(GL::Buffer::maxAtomicCounterBindings())
        _l(GL::Shader::maxAtomicCounterBuffers(GL::Shader::Type::Vertex))
        _l(GL::Shader::maxAtomicCounterBuffers(GL::Shader::Type::TessellationControl))
        _l(GL::Shader::maxAtomicCounterBuffers(GL::Shader::Type::TessellationEvaluation))
        _l(GL::Shader::maxAtomicCounterBuffers(GL::Shader::Type::Geometry))
        _l(GL::Shader::maxAtomicCounterBuffers(GL::Shader::Type::Compute))
        _l(GL::Shader::maxAtomicCounterBuffers(GL::Shader::Type::Fragment))
        _l(GL::Shader::maxCombinedAtomicCounterBuffers())
        _l(GL::Shader::maxAtomicCounters(GL::Shader::Type::Vertex))
        _l(GL::Shader::maxAtomicCounters(GL::Shader::Type::TessellationControl))
        _l(GL::Shader::maxAtomicCounters(GL::Shader::Type::TessellationEvaluation))
        _l(GL::Shader::maxAtomicCounters(GL::Shader::Type::Geometry))
        _l(GL::Shader::maxAtomicCounters(GL::Shader::Type::Compute))
        _l(GL::Shader::maxAtomicCounters(GL::Shader::Type::Fragment))
        _l(GL::Shader::maxCombinedAtomicCounters())
        _l(GL::AbstractShaderProgram::maxAtomicCounterBufferSize())
    }

    #ifndef MAGNUM_TARGET_GLES
    if(c.isExtensionSupported<GL::Extensions::ARB::shader_image_load_store>())
    #endif
    {
        #ifndef MAGNUM_TARGET_GLES
        _h(ARB::shader_image_load_store)
        #endif

        _l(GL::Shader::maxImageUniforms(GL::Shader::Type::Vertex))
        _l(GL::Shader::maxImageUniforms(GL::Shader::Type::TessellationControl))
        _l(GL::Shader::maxImageUniforms(GL::Shader::Type::TessellationEvaluation))
        _l(GL::Shader::maxImageUniforms(GL::Shader::Type::Geometry))
        _l(GL::Shader::maxImageUniforms(GL::Shader::Type::Compute))
        _l(GL::Shader::maxImageUniforms(GL::Shader::Type::Fragment))
        _l(GL::Shader::maxCombinedImageUniforms())
        _l(GL::AbstractShaderProgram::maxCombinedShaderOutputResources())
        _l(GL::AbstractShaderProgram::maxImageUnits())
        #ifndef MAGNUM_TARGET_GLES
        _l(GL::AbstractShaderProgram::maxImageSamples())
        #endif
    }

    #ifndef MAGNUM_TARGET_GLES
    if(c.isExtensionSupported<GL::Extensions::ARB::shader_storage_buffer_object>())
    #endif
    {
        #ifndef MAGNUM_TARGET_GLES
        _h(ARB::shader_storage_buffer_object)
        #endif

        _l(GL::Buffer::shaderStorageOffsetAlignment())
        _l(GL::Buffer::maxShaderStorageBindings())
        _l(GL::Shader::maxShaderStorageBlocks(GL::Shader::Type::Vertex))
        _l(GL::Shader::maxShaderStorageBlocks(GL::Shader::Type::TessellationControl))
        _l(GL::Shader::maxShaderStorageBlocks(GL::Shader::Type::TessellationEvaluation))
        _l(GL::Shader::maxShaderStorageBlocks(GL::Shader::Type::Geometry))
        _l(GL::Shader::maxShaderStorageBlocks(GL::Shader::Type::Compute))
        _l(GL::Shader::maxShaderStorageBlocks(GL::Shader::Type::Fragment))
        _l(GL::Shader::maxCombinedShaderStorageBlocks())
        /* AbstractShaderProgram::maxCombinedShaderOutputResources() already in shader_image_load_store */
        _l(GL::AbstractShaderProgram::maxShaderStorageBlockSize())
    }
    #endif

    #if !defined(MAGNUM_TARGET_GLES2) && !defined(MAGNUM_TARGET_WEBGL)
    #ifndef MAGNUM_TARGET_GLES
    if(c.isExtensionSupported<GL::Extensions::ARB::texture_multisample>())
    #endif
    {
        #ifndef MAGNUM_TARGET_GLES
        _h(ARB::texture_multisample)
        #endif

        _l(GL::AbstractTexture::maxColorSamples())
        _l(GL::AbstractTexture::maxDepthSamples())
        _l(GL::AbstractTexture::maxIntegerSamples())
        _lvec(GL::MultisampleTexture2D::maxSize())
        _lvec(GL::MultisampleTexture2DArray::maxSize())
    }
    #endif

    #ifndef MAGNUM_TARGET_GLES
    if(c.isExtensionSupported<GL::Extensions::ARB::texture_rectangle>()) {
        _h(ARB::texture_rectangle)

        _lvec(GL::RectangleTexture::maxSize())
    }
    #endif

    #ifndef MAGNUM_TARGET_GLES2
    #ifndef MAGNUM_TARGET_GLES
    if(c.isExtensionSupported<GL::Extensions::ARB::uniform_buffer_object>())
    #endif
    {
        #ifndef MAGNUM_TARGET_GLES
        _h(ARB::uniform_buffer_object)
        #endif

        _l(GL::Buffer::uniformOffsetAlignment())
        _l(GL::Buffer::maxUniformBindings())
        _l(GL::Shader::maxUniformBlocks(GL::Shader::Type::Vertex))
        #ifndef MAGNUM_TARGET_WEBGL
        _l(GL::Shader::maxUniformBlocks(GL::Shader::Type::TessellationControl))
        _l(GL::Shader::maxUniformBlocks(GL::Shader::Type::TessellationEvaluation))
        _l(GL::Shader::maxUniformBlocks(GL::Shader::Type::Geometry))
        _l(GL::Shader::maxUniformBlocks(GL::Shader::Type::Compute))
        #endif
        _l(GL::Shader::maxUniformBlocks(GL::Shader::Type::Fragment))
        _l(GL::Shader::maxCombinedUniformBlocks())
        _l(GL::Shader::maxCombinedUniformComponents(GL::Shader::Type::Vertex))
        #ifndef MAGNUM_TARGET_WEBGL
        _l(GL::Shader::maxCombinedUniformComponents(GL::Shader::Type::TessellationControl))
        _l(GL::Shader::maxCombinedUniformComponents(GL::Shader::Type::TessellationEvaluation))
        _l(GL::Shader::maxCombinedUniformComponents(GL::Shader::Type::Geometry))
        _l(GL::Shader::maxCombinedUniformComponents(GL::Shader::Type::Compute))
        #endif
        _l(GL::Shader::maxCombinedUniformComponents(GL::Shader::Type::Fragment))
        _l(GL::AbstractShaderProgram::maxUniformBlockSize())
    }

    #ifndef MAGNUM_TARGET_GLES
    if(c.isExtensionSupported<GL::Extensions::EXT::gpu_shader4>())
    #endif
    {
        #ifndef MAGNUM_TARGET_GLES
        _h(EXT::gpu_shader4)
        #endif

        _l(GL::AbstractShaderProgram::minTexelOffset())
        _l(GL::AbstractShaderProgram::maxTexelOffset())
    }

    #ifndef MAGNUM_TARGET_GLES
    if(c.isExtensionSupported<GL::Extensions::EXT::texture_array>())
    #endif
    {
        #ifndef MAGNUM_TARGET_GLES
        _h(EXT::texture_array)
        #endif

        #ifndef MAGNUM_TARGET_GLES
        _lvec(GL::Texture1DArray::maxSize())
        #endif
        _lvec(GL::Texture2DArray::maxSize())
    }
    #endif

    #ifndef MAGNUM_TARGET_GLES2
    #ifndef MAGNUM_TARGET_GLES
    if(c.isExtensionSupported<GL::Extensions::EXT::transform_feedback>())
    #endif
    {
        #ifndef MAGNUM_TARGET_GLES
        _h(EXT::transform_feedback)
        #endif

        _l(GL::TransformFeedback::maxInterleavedComponents())
        _l(GL::TransformFeedback::maxSeparateAttributes())
        _l(GL::TransformFeedback::maxSeparateComponents())
    }
    #endif

    #ifndef MAGNUM_TARGET_GLES
    if(c.isExtensionSupported<GL::Extensions::ARB::transform_feedback3>()) {
        _h(ARB::transform_feedback3)

        _l(GL::TransformFeedback::maxBuffers())
        _l(GL::TransformFeedback::maxVertexStreams())
    }
    #endif

    #if !defined(MAGNUM_TARGET_GLES2) && !defined(MAGNUM_TARGET_WEBGL)
    #ifndef MAGNUM_TARGET_GLES
    if(c.isExtensionSupported<GL::Extensions::ARB::geometry_shader4>())
    #else
    if(c.isExtensionSupported<GL::Extensions::EXT::geometry_shader>())
    #endif
    {
        #ifndef MAGNUM_TARGET_GLES
        _h(ARB::geometry_shader4)
        #else
        _h(EXT::geometry_shader)
        #endif

        _l(GL::AbstractShaderProgram::maxGeometryOutputVertices())
        _l(GL::Shader::maxGeometryInputComponents())
        _l(GL::Shader::maxGeometryOutputComponents())
        _l(GL::Shader::maxGeometryTotalOutputComponents())
    }
    #endif

    #if !defined(MAGNUM_TARGET_GLES2) && !defined(MAGNUM_TARGET_WEBGL)
    #ifndef MAGNUM_TARGET_GLES
    if(c.isExtensionSupported<GL::Extensions::ARB::tessellation_shader>())
    #else
    if(c.isExtensionSupported<GL::Extensions::EXT::tessellation_shader>())
    #endif
    {
        #ifndef MAGNUM_TARGET_GLES
        _h(ARB::tessellation_shader)
        #else
        _h(EXT::tessellation_shader)
        #endif

        _l(GL::Shader::maxTessellationControlInputComponents())
        _l(GL::Shader::maxTessellationControlOutputComponents())
        _l(GL::Shader::maxTessellationControlTotalOutputComponents())
        _l(GL::Shader::maxTessellationEvaluationInputComponents())
        _l(GL::Shader::maxTessellationEvaluationOutputComponents())
        _l(GL::Renderer::maxPatchVertexCount())
    }
    #endif

    #if !defined(MAGNUM_TARGET_GLES2) && !defined(MAGNUM_TARGET_WEBGL)
    #ifndef MAGNUM_TARGET_GLES
    if(c.isExtensionSupported<GL::Extensions::ARB::texture_buffer_object>())
    #else
    if(c.isExtensionSupported<GL::Extensions::EXT::texture_buffer>())
    #endif
    {
        #ifndef MAGNUM_TARGET_GLES
        _h(ARB::texture_buffer_object)
        #else
        _h(EXT::texture_buffer)
        #endif

        _l(GL::BufferTexture::maxSize())
    }

    #ifndef MAGNUM_TARGET_GLES
    if(c.isExtensionSupported<GL::Extensions::ARB::texture_buffer_range>())
    #else
    if(c.isExtensionSupported<GL::Extensions::EXT::texture_buffer>())
    #endif
    {
        #ifndef MAGNUM_TARGET_GLES
        _h(ARB::texture_buffer_range)
        #else
        /* Header added above */
        #endif

        _l(GL::BufferTexture::offsetAlignment())
    }

    #ifndef MAGNUM_TARGET_GLES
    if(c.isExtensionSupported<GL::Extensions::ARB::texture_cube_map_array>())
    #else
    if(c.isExtensionSupported<GL::Extensions::EXT::texture_cube_map_array>())
    #endif
    {
        #ifndef MAGNUM_TARGET_GLES
        _h(ARB::texture_cube_map_array)
        #else
        _h(EXT::texture_cube_map_array)
        #endif

        _lvec(GL::CubeMapTextureArray::maxSize())
    }
    #endif

    #ifndef MAGNUM_TARGET_GLES
    if(c.isExtensionSupported<GL::Extensions::ARB::texture_filter_anisotropic>()) {
        _h(ARB::texture_filter_anisotropic)

        _l(GL::Sampler::maxMaxAnisotropy())
    } else
    #endif
    if(c.isExtensionSupported<GL::Extensions::EXT::texture_filter_anisotropic>()) {
        _h(EXT::texture_filter_anisotropic)

        _l(GL::Sampler::maxMaxAnisotropy())
    }

    #ifndef MAGNUM_TARGET_WEBGL
    if(c.isExtensionSupported<GL::Extensions::KHR::debug>()) {
        _h(KHR::debug)

        _l(GL::AbstractObject::maxLabelLength())
        _l(GL::DebugOutput::maxLoggedMessages())
        _l(GL::DebugOutput::maxMessageLength())
        _l(GL::DebugGroup::maxStackDepth())
    }
    #endif

    #if defined(MAGNUM_TARGET_GLES2) && !defined(MAGNUM_TARGET_WEBGL)
    if(c.isExtensionSupported<GL::Extensions::OES::texture_3D>()) {
        _h(OES::texture_3D)

        _lvec(GL::Texture3D::maxSize())
    }
    #endif

    #undef _l
    #undef _h

    return os.str();
}


const std::unordered_map<std::string, std::string> rendering::GLInfo::getInfo() {

    std::unordered_map<std::string, std::string> o;

    if(Magnum::GL::Context::hasCurrent()) {
        Magnum::GL::Context& context = Magnum::GL::Context::current();
        context.detectedDriver();
        
        o["vendor"] = context.vendorString();
        o["version"] = context.versionString();
        o["renderer"] = context.rendererString();
        o["shading_language_version"] = context.shadingLanguageVersionString();
    }
    
    return o;

}

const std::vector<std::string> rendering::GLInfo::getExtensionsInfo() {
    std::vector<std::string> o;

    if(Magnum::GL::Context::hasCurrent()) {
        Magnum::GL::Context& context = Magnum::GL::Context::current();
        
        context.detectedDriver();

        for (auto s : context.extensionStrings())
            o.push_back(s);
        
    }

    return o;
}

std::unordered_map<std::string, std::string> rendering::glInfo() {
    std::unordered_map<std::string, std::string> result;
    
    if(Magnum::GL::Context::hasCurrent()) {
        auto info = rendering::GLInfo::getInfo();
        auto extInfo = rendering::GLInfo::getExtensionsInfo();

        for (auto &i : info)
            result[i.first] = i.second;

        std::string extInfoStr;
        if(extInfo.size() > 0) {
            extInfoStr = extInfo[0];
            if(extInfo.size() > 1) {
                for(unsigned int i = 1; i < extInfo.size(); ++i)
                    extInfoStr += ", " + extInfo[i];
            }
        }
        result["extensions"] = extInfoStr;
    }
    
    return result;
}
